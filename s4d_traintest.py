import torch
import torch.optim as optim
import torch.nn as nn
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import iirnotch, filtfilt
from sklearn.utils.class_weight import compute_class_weight

# Detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Notch filter to remove 50Hz powerline noise
def apply_notch_filter(signal, fs=128, f0=50, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, signal)

# Load and preprocess data
file_paths = glob.glob('eeg_label/*_eeg_label.csv')
data_list, labels_list, patient_ids = [], [], []

for file_path in file_paths:
    eeg_data = pd.read_csv(file_path)
    patient_id = file_path.split('/')[-1]

    # Apply notch filter
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

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y = torch.tensor(y_encoded, dtype=torch.long)

# Train (70%), Test (15%), Validation (15%) split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

# Move data to GPU
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Import S4D model
import sys
sys.path.append('/home/dpha0015/pc88/dpha0015/s4')
from models.s4.s4d import S4D

# Define S4D Classification Model with LogSoftmax
class S4DClassificationModel(nn.Module):
    def __init__(self, num_layers, d_model, num_classes):
        super(S4DClassificationModel, self).__init__()
        self.s4d_layers = nn.ModuleList([S4D(d_model=d_model, dropout=0.1) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for layer in self.s4d_layers:
            x, _ = layer(x)
        x = x.mean(dim=-1)  # Global average pooling
        x = self.classifier(x)
        return self.log_softmax(x)

# Initialize the model, loss function, and optimizer
model = S4DClassificationModel(num_layers=2, d_model=128, num_classes=5).to(device)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.NLLLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop with Validation
print("Starting training...")
best_val_loss = float('inf')

for epoch in range(10):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_loader):.4f}')

    # Validation step
    model.eval()
    val_loss, val_preds = 0, []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(y_val.cpu().numpy(), val_preds)
    val_f1 = f1_score(y_val.cpu().numpy(), val_preds, average='weighted')

    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy * 100:.2f}%, F1 Score: {val_f1:.2f}')

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_s4d.pth')
        print("Model saved.")

# Load the best model for testing
model.load_state_dict(torch.load('best_model_s4d.pth'))

# Test the model
print("Testing the model...")
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
print(f'Test F1 Score: {test_f1:.2f}')

# Save test classification report
test_report = classification_report(y_test.cpu().numpy(), test_preds, zero_division=1)
print("\nTest Classification Report:\n", test_report)

with open('s4d_result/traintest_report.txt', 'w') as f:
    f.write(test_report)
    f.write(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%\n")
    f.write(f"Validation F1 Score: {val_f1:.2f}\n")
    f.write(f"\nTest Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Test F1 Score: {test_f1:.2f}\n")
    
    
test_preds = pd.DataFrame({
    'True_Labels': y_test.cpu().numpy(),
    'Predicted_Labels': test_preds
})
test_preds.to_csv('s4d_test_preds.csv', index=False)