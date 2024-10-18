import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import random
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, LeaveOneGroupOut, train_test_split
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

X_train, X_temp, y_train, y_temp, train_ids, temp_ids = train_test_split(
    X.cpu(), y.cpu(), patient_ids, test_size=0.3, stratify=y.cpu(), random_state=42)

X_test, X_val, y_test, y_val, test_ids, val_ids = train_test_split(
    X_temp, y_temp, temp_ids, test_size=1/3, stratify=y_temp, random_state=42)


# Move data back to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

# Define batch size and number of epochs
batch_size = 8
num_epochs = 10

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

# Create datasets and data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = ConvTran(config, num_classes=config['num_labels']).to(device)

# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop with validation
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

    # Validation step
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())

    val_accuracy = accuracy_score(y_val.cpu().numpy(), val_preds)
    val_f1 = f1_score(y_val.cpu().numpy(), val_preds, average='weighted')

    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
    print(f'Validation F1 Score: {val_f1 * 100:.2f}%')

# Test the model
model.eval()
test_preds = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())

# Calculate test metrics
test_accuracy = accuracy_score(y_test.cpu().numpy(), test_preds)
test_f1 = f1_score(y_test.cpu().numpy(), test_preds, average='weighted')

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test F1 Score: {test_f1 * 100:.2f}%')

# Save test classification report
test_report = classification_report(y_test.cpu().numpy(), test_preds, zero_division=1)
print("\nTest Classification Report:\n", test_report)

with open('convtran_result/traintest_report.txt', 'w') as f:
    f.write(test_report)
    f.write(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%\n")
    f.write(f"Validation F1 Score: {val_f1 * 100:.2f}%\n")
    f.write(f"\nTest Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Test F1 Score: {test_f1 * 100:.2f}%\n")
    
test_preds = pd.DataFrame({
    'True_Labels': y_test.cpu().numpy(),
    'Predicted_Labels': test_preds
})
test_preds.to_csv('convtran_test_preds.csv', index=False)

print(f"Training Samples: {len(X_train)}, Validation Samples: {len(X_val)}, Test Samples: {len(X_test)}")