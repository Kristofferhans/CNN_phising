import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the saved model
class CNNRNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim=128, num_layers=2, 
                 kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(CNNRNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.rnn = nn.LSTM(input_size=num_filters * len(kernel_sizes),
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        cnn_out = torch.cat(pooled, 1)
        rnn_input = cnn_out.unsqueeze(1)
        rnn_out, _ = self.rnn(rnn_input)
        rnn_out = self.dropout(rnn_out[:, -1, :])
        output = self.fc(rnn_out)
        return output

# Function to find the correct data path (same as in training)
def find_data_file(filename, search_paths):
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(f"Could not find {filename} in any of these locations: {search_paths}")

# Load the dataset
try:
    possible_paths = [
        r"C:\Users\krist\Data science\island\master",
        r"C:\Users\krist\Data science\island",
        r"C:\Users\krist\Data science\island\CNN_phising",
        os.path.dirname(os.path.abspath(__file__)),
    ]
    
    data_file = "Phishing_Email - Phishing_Email.csv"
    data_path = find_data_file(data_file, possible_paths)
    df = pd.read_csv(data_path)

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Preprocessing (same as in training)
df = df.dropna()
label_encoder = LabelEncoder()
df['Email Type'] = label_encoder.fit_transform(df['Email Type'])

# Split data (using same random_state for consistency)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer and vocabulary (same as in training)
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_df['Email Text']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

# Dataset class (same as in training)
class EmailDataset(Dataset):
    def __init__(self, df, text_pipeline):
        self.df = df
        self.text_pipeline = text_pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['Email Text']
        label = self.df.iloc[idx]['Email Type']
        text_indices = self.text_pipeline(text)
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Create test dataset and loader
test_dataset = EmailDataset(test_df, text_pipeline)

def collate_batch(batch):
    text_list, label_list = [], []
    for (text, label) in batch:
        text_list.append(text)
        label_list.append(label)
    return torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True), torch.tensor(label_list)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# Initialize and load the model
vocab_size = len(vocab)
embed_dim = 100
num_classes = len(label_encoder.classes_)
hidden_dim = 128
num_layers = 2
kernel_sizes = [3, 4, 5]
num_filters = 100
dropout = 0.5

model = CNNRNNTextClassifier(vocab_size, embed_dim, num_classes, hidden_dim, num_layers, 
                           kernel_sizes, num_filters, dropout)

model.load_state_dict(torch.load('phishing_email_cnn_rnn.pth'))
model.eval()

# Evaluation function
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return accuracy, precision, recall, f1

# Run evaluation
print("Evaluating model on test set...")
accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

# Print class-wise metrics if needed
print("\nDetailed Classification Report:")
from sklearn.metrics import classification_report
with torch.no_grad():
    all_preds = []
    all_labels = []
    for texts, labels in test_loader:
        outputs = model(texts)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))