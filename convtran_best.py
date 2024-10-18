#47.45%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import random
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

import numpy as np
from torch import nn
from ConvTran.Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from ConvTran.Models.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec

from scipy.signal import iirnotch, filtfilt


def apply_notch_filter(data, fs=128, freq=50.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


def model_factory(config):
    if config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'CC-T':
        model = CasualConvTran(config, num_classes=config['num_labels'])
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    return model


class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x_src = self.embed_layer(x.permute(0, 2, 1))
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        # out = out.permute(1, 0, 2)
        # out = self.out(out[-1])
        return out


class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out


class CasualConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.causal_Conv1 = nn.Sequential(CausalConv1d(channel_size, emb_size, kernel_size=8, stride=2, dilation=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv2 = nn.Sequential(CausalConv1d(emb_size, emb_size, kernel_size=5, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv3 = nn.Sequential(CausalConv1d(emb_size, emb_size, kernel_size=3, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(nn.functional.pad(x, (self.__padding, 0)))
    
    
# Load data
file_paths = glob.glob('eeg_label/*_eeg_label.csv')

data_list = []
labels_list = []
patient_ids = []

for file_path in file_paths:
    eeg_data = pd.read_csv(file_path)
    patient_id = file_path.split('/')[-1].split('_')[0]

    # Segment the data into non-overlapping 30-second windows
    segment_size = 128 * 30  # Assuming a sampling rate of 128 Hz
    num_segments = len(eeg_data) // segment_size

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        
        # Extract and apply notch filter to the EEG segment
        segment = eeg_data['EEG'].iloc[start_idx:end_idx].values
        filtered_segment = apply_notch_filter(segment)

        label = eeg_data['Label'].iloc[start_idx:end_idx].mode()[0]
        
        data_list.append(filtered_segment)
        labels_list.append(label)
        patient_ids.append(patient_id)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels_list)

# Convert lists to numpy arrays and then to PyTorch tensors
X = torch.tensor(np.array(data_list), dtype=torch.float32)
y = torch.tensor(y_encoded, dtype=torch.long)
patient_ids = np.array(patient_ids)

# Reshape X to include a channel dimension
X = X.unsqueeze(1)  # Shape: (num_samples, 1, sequence_length)

# Define number of epochs and batch size
num_epochs = 10
batch_size = 8

# Model configuration
config = {
    'Data_shape': (X.shape[0], 1, X.shape[2]),  # Match data shape
    'emb_size': 32,  
    'num_heads': 4,  
    'dim_ff': 64,  
    'dropout': 0.1,
    'Fix_pos_encode': 'tAPE',
    'Rel_pos_encode': 'Scalar',
    'num_labels': 5
}

# Initialise LOPO cross-validator
logo = LeaveOneGroupOut()

# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
X, y = X.to(device), y.to(device)

patient_reports = []
accuracies = []
f1_scores = []

# Initialize lists for overall predictions and labels
all_preds = []
all_labels = []

# LOPOCV loop for each patient
for train_idx, test_idx in logo.split(X, y, groups=patient_ids):
    # Split the data into training and testing sets
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Convert to PyTorch dataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = ConvTran(config, num_classes=config['num_labels']).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        all_preds_batch = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds_batch.append(predicted.cpu())  # Move to CPU

        all_preds_batch = torch.cat(all_preds_batch)

        # Store predictions and labels for overall report
        all_preds.extend(all_preds_batch.numpy())
        all_labels.extend(y_test.cpu().numpy())

        # Calculate metrics for the current patient
        patient_accuracy = accuracy_score(y_test.cpu(), all_preds_batch)
        patient_f1 = f1_score(y_test.cpu(), all_preds_batch, average='weighted')
        patient_report = classification_report(y_test.cpu(), all_preds_batch, zero_division=1)

        # Print and store individual patient metrics
        patient_id = patient_ids[test_idx[0]]
        print(f'Patient ID: {patient_id}, Accuracy: {patient_accuracy * 100:.2f}%, F1 Score: {patient_f1 * 100:.2f}%')
        print(f'Classification Report for Patient ID: {patient_id}:\n{patient_report}')

        # Store results in memory for later saving
        accuracies.append(patient_accuracy)
        f1_scores.append(patient_f1)
        patient_reports.append(f'Patient ID: {patient_id}, Accuracy: {patient_accuracy * 100:.2f}%, F1 Score: {patient_f1 * 100:.2f}%\n{patient_report}\n')

# Overall classification report
overall_report = classification_report(all_labels, all_preds, zero_division=1)
print(overall_report)

# Calculate average accuracy and F1 score
average_accuracy = sum(accuracies) / len(accuracies)
average_f1 = sum(f1_scores) / len(f1_scores)

# Print the average accuracy and F1 score
print(f'Average LOPOCV Accuracy: {average_accuracy * 100:.2f}%')
print(f'Average LOPOCV F1 Score: {average_f1 * 100:.2f}%')

# Save individual reports and overall report to a file
with open('convtran_result/classification_report.txt', 'w') as f:
    for report in patient_reports:
        f.write(report)
    f.write("\nOverall Classification Report:\n")
    f.write(overall_report)
    f.write(f"\nAverage LOPOCV Accuracy: {average_accuracy * 100:.2f}%\n")
    f.write(f"Average LOPOCV F1 Score: {average_f1 * 100:.2f}%\n")

# Save Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('convtran_result/confusion_matrix.png')
plt.show()

# Save metrics to a separate file
with open('convtran_result/metrics.txt', 'w') as f:
    f.write("Accuracies:\n")
    f.write("\n".join(map(str, accuracies)) + "\n")
    f.write("F1 Scores:\n")
    f.write("\n".join(map(str, f1_scores)) + "\n")
    f.write(f"\nAverage LOPOCV Accuracy: {average_accuracy * 100:.2f}%\n")
    f.write(f"Average LOPOCV F1 Score: {average_f1 * 100:.2f}%\n")

    
predictions_df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'True_Labels': all_labels,
    'Predicted_Labels': all_preds
})
predictions_df.to_csv('convtran_predictions_lopocv.csv', index=False)

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

convtran_results_df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Predicted_Labels': all_preds,  # Assuming all_preds contains predictions from your model
    'True_Labels': all_labels,
    'BIS': BIS
})

palette = sns.color_palette('deep', n_colors=5)

# Plot distribution for each class (Predictions vs. BIS)
plt.figure(figsize=(12, 6))

# Add a title for the overall graph to indicate MiniRocket
plt.suptitle('ConvTran Model: Comparison of Predicted and True Labels with BIS', fontsize=16)

# Boxplot for predictions
plt.subplot(1, 2, 1)
sns.boxplot(x='Predicted_Labels', y='BIS', data=convtran_results_df, hue='Predicted_Labels', palette=palette, dodge=False)
plt.title('Predicted Class vs BIS Distribution')
plt.xlabel('Predicted Classes')
plt.ylabel('BIS')

# Boxplot for true labels
plt.subplot(1, 2, 2)
sns.boxplot(x='True_Labels', y='BIS', data=convtran_results_df, hue='True_Labels', palette=palette, dodge=False)
plt.title('True Class vs BIS Distribution')
plt.xlabel('True Classes')
plt.ylabel('BIS')

plt.tight_layout()
plt.savefig('convtran_result/distribution1.png')
plt.show()
