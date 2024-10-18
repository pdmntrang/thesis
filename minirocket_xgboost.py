import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy.stats import ttest_rel

from sktime.transformations.panel.rocket import MiniRocket
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, LeaveOneGroupOut, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.signal import iirnotch, filtfilt  

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791

from numba import njit, prange, vectorize
import numpy as np

@njit("float32[:](float32[:,:],int32[:],int32[:],float32[:])", fastmath = True, parallel = False, cache = True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles):

    num_examples, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

    return biases

def _fit_dilations(input_length, num_features, max_dilations_per_kernel):

    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, num_features = 10_000, max_dilations_per_kernel = 32):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)

    return dilations, num_features_per_dilation, biases

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
@vectorize("float32(float32,float32)", nopython = True, cache = True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:])))", fastmath = True, parallel = True, cache = True)
def transform(X, parameters):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()

                feature_index_start = feature_index_end

    return features


# Load EEG data and segment it into 30-second windows
file_paths = glob.glob('eeg_label/*_eeg_label.csv')

data_list, labels_list, patient_ids = [], [], []

for file_path in file_paths:
    eeg_data = pd.read_csv(file_path)
    patient_id = file_path.split('/')[-1]

    segment_size = 128 * 30  # 30 seconds window at 128 Hz
    num_segments = len(eeg_data) // segment_size

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment = eeg_data['EEG'].iloc[start_idx:end_idx].values
        label = eeg_data['Label'].iloc[start_idx:end_idx].mode()[0]
        
        data_list.append(segment)
        labels_list.append(label)
        patient_ids.append(patient_id)

# Convert data to numpy arrays and move to the GPU
X = np.array(data_list).astype(np.float32)
y = np.array(labels_list)
patient_ids = np.array(patient_ids)

# Flatten data for MiniRocket and transform it
parameters = fit(X)  # Fit MiniRocket
X_transformed = transform(X, parameters)
X_transformed = X_transformed.reshape(X_transformed.shape[0], -1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize variables for predictions, labels, and reports
all_preds, all_labels = [], []
patient_reports, accuracies, f1_scores = [], [], []
patient_predictions = {}  
patient_true_labels = {}  

# Leave-One-Patient-Out Cross-Validation
logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X_transformed, y_encoded, groups=patient_ids):
    X_train, X_test = X_transformed[train_idx], X_transformed[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Initialise XGBoost with GPU support
    xgb_classifier = xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=500,
        gamma=0.1,
        min_child_weight=3,
        alpha=0.5,
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',
        random_state=42
    )

    # Train the XGBoost model
    xgb_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_classifier.predict(X_test)

    # Store predictions and true labels for the current patient
    patient_id = patient_ids[test_idx[0]]
    
    if patient_id not in patient_predictions:
        patient_predictions[patient_id] = []  # Initialize list for predictions
        patient_true_labels[patient_id] = []  # Initialize list for true labels
    
    patient_predictions[patient_id].extend(y_pred)
    patient_true_labels[patient_id].extend(y_test)

    # Store for overall metrics
    all_preds.extend(y_pred)
    all_labels.extend(y_test)

    # Generate and print classification report for this patient
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    patient_reports.append(f'Patient {patient_id} Report:\n{report}\n')
    print(f'Patient {patient_id} Report:\n{report}')

    # Calculate and store accuracy and F1 score for this patient
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

# Generate and print the overall classification report
overall_report = classification_report(all_labels, all_preds, zero_division=1)
print(overall_report)

# Calculate and print average accuracy and F1 score
average_accuracy = np.mean(accuracies)
average_f1 = np.mean(f1_scores)
print(f'Average LOPOCV Accuracy: {average_accuracy * 100:.2f}%')
print(f'Average LOPOCV F1 Score: {average_f1 * 100:.2f}%')

# Save classification reports
with open('minirocket_result/classification_report1.txt', 'w') as f:
    for report in patient_reports:
        f.write(report)
    f.write("\nOverall Classification Report:\n")
    f.write(overall_report)
    f.write(f"\nAverage LOPOCV Accuracy: {average_accuracy * 100:.2f}%\n")
    f.write(f"Average LOPOCV F1 Score: {average_f1 * 100:.2f}%\n")

# Plot and save the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('minirocket_result/confusion_matrix1.png')
plt.show()

# Save accuracies and F1 scores
with open('minirocket_result/metrics1.txt', 'w') as f:
    f.write("Accuracies:\n" + "\n".join(map(str, accuracies)) + "\n")
    f.write("F1 Scores:\n" + "\n".join(map(str, f1_scores)) + "\n")
    f.write(f"\nAverage LOPOCV Accuracy: {average_accuracy * 100:.2f}%\n")
    f.write(f"Average LOPOCV F1 Score: {average_f1 * 100:.2f}%\n")
    
predictions_df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'True_Labels': all_labels,
    'Predicted_Labels': all_preds
})
predictions_df.to_csv('minirocket_predictions_lopocv.csv', index=False)

####Visualisation and t-test
# Load BIS data
file_paths = glob.glob('bis_label/*_label.csv')  # Adjust this to the correct BIS data path

bis_list = []
patient_ids = []

for file_path in file_paths:
    bis_data = pd.read_csv(file_path)
    patient_id = file_path.split('/')[-1]
    
    # Segment the BIS data into non-overlapping 30-second windows
    segment_size = 128 * 30
    num_segments = len(bis_data) // segment_size
    
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        bis_segment = bis_data['BIS'].iloc[start_idx:end_idx].mean()  # Calculate the mean BIS for each segment
        bis_list.append(bis_segment)
        patient_ids.append(patient_id)  # Repeat patient ID for each segment

# Convert to numpy arrays
BIS = np.array(bis_list)
patient_ids = np.array(patient_ids)
all_preds = label_encoder.inverse_transform(all_preds)  # Decode predicted labels
all_labels = label_encoder.inverse_transform(all_labels) 

import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for predictions, true labels, and BIS values
minirocket_results_df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Predicted_Labels': all_preds,  # Assuming all_preds contains predictions from your model
    'True_Labels': all_labels,
    'BIS': BIS
})

palette = sns.color_palette('deep', n_colors=5)

# Plot distribution for each class (Predictions vs. BIS)
plt.figure(figsize=(12, 6))

# Add a title for the overall graph to indicate MiniRocket
plt.suptitle('MiniRocket Model: Comparison of Predicted and True Labels with BIS', fontsize=16)

# Boxplot for predictions
plt.subplot(1, 2, 1)
sns.boxplot(x='Predicted_Labels', y='BIS', data=minirocket_results_df, hue='Predicted_Labels', palette=palette, dodge=False)
plt.title('Predicted Class vs BIS Distribution')
plt.xlabel('Predicted Classes')
plt.ylabel('BIS')

# Boxplot for true labels
plt.subplot(1, 2, 2)
sns.boxplot(x='True_Labels', y='BIS', data=minirocket_results_df, hue='True_Labels', palette=palette, dodge=False)

plt.title('True Class vs BIS Distribution')
plt.xlabel('True Classes')
plt.ylabel('BIS')

plt.tight_layout()
plt.savefig('minirocket_result/distribution1.png')
plt.show()