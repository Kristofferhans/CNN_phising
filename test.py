import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import numpy as np

# Define the CNN model (same as during training)
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, seq_len - kernel_size + 1)]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, num_filters)]
        x = torch.cat(x, 1)  # (batch_size, num_filters * len(kernel_sizes))
        x = self.fc(x)  # (batch_size, num_classes)
        return x

# Load the dataset (same as during training)
data_path = r"C:\Users\krist\Data science\island\CNN_phising\CNN_phising\Phishing_Email - Phishing_Email.csv"
df = pd.read_csv(data_path)
df = df.dropna()

# Encode the labels
label_encoder = LabelEncoder()
df['Email Type'] = label_encoder.fit_transform(df['Email Type'])

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer and vocabulary (same as during training)
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_df['Email Text']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

# Dataset class (same as during training)
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

# Create test dataset
test_dataset = EmailDataset(test_df, text_pipeline)

# DataLoader for test dataset
def collate_batch(batch):
    text_list, label_list = [], []
    for (text, label) in batch:
        text_list.append(text)
        label_list.append(label)
    return torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True), torch.tensor(label_list)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# Load the saved model
vocab_size = len(vocab)
embed_dim = 100
num_classes = len(label_encoder.classes_)
kernel_sizes = [3, 4, 5]
num_filters = 100

# Load the saved model checkpoint
checkpoint = torch.load('phishing_email_cnn.pth')

# Initialize the model with the same parameters used during training
model = CNNTextClassifier(
    vocab_size=len(checkpoint['vocab']),
    embed_dim=checkpoint['config']['embed_dim'],
    num_classes=len(checkpoint['label_encoder'].classes_),
    kernel_sizes=checkpoint['config']['kernel_sizes'],
    num_filters=checkpoint['config']['num_filters']
)

# Load the model state dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Update your local variables to match the saved ones
vocab = checkpoint['vocab']
label_encoder = checkpoint['label_encoder']


# Evaluate the model
y_true = []
y_pred = []

with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()