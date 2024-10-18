import torch
import torch.optim as optim
import torch.nn as nn
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import iirnotch, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns

# Detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def apply_notch_filter(signal, fs=128, f0=50, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, signal)

# Load and preprocess data
file_paths = glob.glob('eeg_label/*_eeg_label.csv')
data_list, labels_list, patient_ids = [], [], []

for file_path in file_paths:
    eeg_data = pd.read_csv(file_path)
    patient_id = file_path.split('/')[-1]

    eeg_data['EEG'] = apply_notch_filter(eeg_data['EEG'].values)

    # Segment into non-overlapping 30-second windows
    segment_size = 128 * 30
    num_segments = len(eeg_data) // segment_size

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment = eeg_data['EEG'].iloc[start_idx:end_idx].values
        label = eeg_data['Label'].iloc[start_idx:end_idx].mode()[0]

        data_list.append(segment)
        labels_list.append(label)
        patient_ids.append(patient_id)

# Convert data to numpy arrays
X = np.array(data_list)
y = np.array(labels_list)
patient_ids = np.array(patient_ids)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
y = torch.tensor(y_encoded, dtype=torch.long).to(device)

import sys
sys.path.append('/home/dpha0015/pc88/dpha0015/s4')

from models.s4.s4d import S4D

# Model definition with LogSoftmax
class S4DClassificationModel(nn.Module):
    def __init__(self, num_layers, d_model, num_classes):
        super(S4DClassificationModel, self).__init__()
        self.s4d_layers = nn.ModuleList([S4D(d_model=d_model, dropout=0.1) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for layer in self.s4d_layers:
            x, _ = layer(x)
        x = x.mean(dim=-1)
        x = self.classifier(x)
        return self.log_softmax(x)

# Initialise LOPO cross-validator
logo = LeaveOneGroupOut()

# Store results
accuracies, f1_scores, all_preds, all_labels = [], [], [], []
patient_reports = []

# Loop over patients (LOPOCV)
for train_idx, test_idx in logo.split(X, y, groups=patient_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Create a model for this fold
    model = S4DClassificationModel(num_layers=2, d_model=128, num_classes=5).to(device)

    # Loss and optimizer with class weights
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

    # Train the model
    model.train()
    for epoch in range(20):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}, Average Loss: {epoch_loss / len(train_loader):.4f}')

    # Evaluate the model
    model.eval()
    all_preds_batch = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            all_preds_batch.append(predicted.cpu())
        all_preds_batch = torch.cat(all_preds_batch)

        # Convert to numpy
        y_test_numpy = y_test.cpu().numpy()
        all_preds_numpy = all_preds_batch.numpy()

        # Append predictions and labels
        all_preds.extend(all_preds_numpy)
        all_labels.extend(y_test_numpy)

        # Generate classification report for this patient
        report = classification_report(y_test_numpy, all_preds_numpy, target_names=label_encoder.classes_)
        patient_reports.append(f'Patient {patient_ids[test_idx[0]]} Report:\n{report}\n')
        print(f'Patient {patient_ids[test_idx[0]]} Report:\n{report}')

        # Calculate accuracy and F1 score using functions
        patient_accuracy = accuracy_score(y_test_numpy, all_preds_numpy)
        patient_f1 = f1_score(y_test_numpy, all_preds_numpy, average='weighted')

        accuracies.append(patient_accuracy)
        f1_scores.append(patient_f1)

# Overall classification report
overall_report = classification_report(all_labels, all_preds, zero_division=1)
print(overall_report)

# Calculate average accuracy and F1 score
average_accuracy = sum(accuracies) / len(accuracies)
average_f1 = sum(f1_scores) / len(f1_scores)

# Print the average accuracy and F1 score
print(f'Average LOPOCV Accuracy: {average_accuracy * 100:.2f}%')
print(f'Average LOPOCV F1 Score: {average_f1 * 100:.2f}%')

# Save results
with open('s4d_result/classification_report1.txt', 'w') as f:
    for report in patient_reports:
        f.write(report)
    f.write("\nOverall Classification Report:\n")
    f.write(overall_report)
    f.write(f"\nAverage LOPOCV Accuracy: {average_accuracy * 100:.2f}%\n")
    f.write(f"Average LOPOCV F1 Score: {average_f1 * 100:.2f}%\n")

with open('s4d_result/metrics.txt', 'w') as f:
    f.write("Accuracies:\n")
    f.write("\n".join(map(str, accuracies)) + "\n")
    f.write("F1 Scores:\n")
    f.write("\n".join(map(str, f1_scores)) + "\n")
    f.write(f"\nAverage LOPOCV Accuracy: {average_accuracy * 100:.2f}%\n")
    f.write(f"Average LOPOCV F1 Score: {average_f1 * 100:.2f}%\n")
    
# Save Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('s4d_result/confusion_matrix.png')
plt.show()


predictions_df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'True_Labels': all_labels,
    'Predicted_Labels': all_preds
})
predictions_df.to_csv('s4d_predictions_lopocv.csv', index=False)

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


s4d_results_df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Predicted_Labels': all_preds, 
    'True_Labels': all_labels,
    'BIS': BIS
})

palette = sns.color_palette('deep', n_colors=5)

# Plot distribution for each class (Predictions vs. BIS)
plt.figure(figsize=(12, 6))

# Add a title for the overall graph to indicate MiniRocket
plt.suptitle('S4D Model: Comparison of Predicted and True Labels with BIS', fontsize=16)

# Boxplot for predictions
plt.subplot(1, 2, 1)
sns.boxplot(x='Predicted_Labels', y='BIS', data=s4d_results_df, hue='Predicted_Labels', palette=palette, dodge=False)
plt.title('Predicted Class vs BIS Distribution')
plt.xlabel('Predicted Classes')
plt.ylabel('BIS')

# Boxplot for true labels
plt.subplot(1, 2, 2)
sns.boxplot(x='True_Labels', y='BIS', data=s4d_results_df, hue='True_Labels', palette=palette, dodge=False)
plt.title('True Class vs BIS Distribution')
plt.xlabel('True Classes')
plt.ylabel('BIS')

plt.tight_layout()
plt.savefig('s4d_result/distribution1.png')
plt.show()
