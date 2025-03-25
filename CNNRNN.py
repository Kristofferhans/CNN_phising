import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
import os

# Function to find the correct data path
def find_data_file(filename, search_paths):
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(f"Could not find {filename} in any of these locations: {search_paths}")

# Try to locate the data file
try:
    # List of possible locations to search for the file
    possible_paths = [
        r"C:\Users\krist\Data science\island\master",
        r"C:\Users\krist\Data science\island",
        r"C:\Users\krist\Data science\island\CNN_phising",
        os.path.dirname(os.path.abspath(__file__)),  # Current script directory
    ]
    
    data_file = "Phishing_Email - Phishing_Email.csv"
    data_path = find_data_file(data_file, possible_paths)
    
    # Load the dataset
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nPlease ensure:")
    print("1. The data file exists in one of these locations:")
    for path in possible_paths:
        print(f"   - {path}")
    print("2. The file is named exactly: 'Phishing_Email - Phishing_Email.csv'")
    exit()

# Rest of your code remains the same from here...
# Drop rows with missing values
df = df.dropna()

# Encode the labels
label_encoder = LabelEncoder()
df['Email Type'] = label_encoder.fit_transform(df['Email Type'])

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_df['Email Text']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Convert text to indices
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

# Dataset class
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

# Create datasets
train_dataset = EmailDataset(train_df, text_pipeline)
test_dataset = EmailDataset(test_df, text_pipeline)

# DataLoader
def collate_batch(batch):
    text_list, label_list = [], []
    for (text, label) in batch:
        text_list.append(text)
        label_list.append(label)
    return torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True), torch.tensor(label_list)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

class CNNRNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim=128, num_layers=2, 
                 kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(CNNRNNTextClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # CNN part
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        
        # RNN part (using LSTM)
        self.rnn = nn.LSTM(input_size=num_filters * len(kernel_sizes),
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # CNN processing
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, seq_len - kernel_size + 1)]
        pooled = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]  # [(batch_size, num_filters)]
        cnn_out = torch.cat(pooled, 1)  # (batch_size, num_filters * len(kernel_sizes))
        
        # Prepare for RNN - expand cnn_out to have a sequence dimension
        rnn_input = cnn_out.unsqueeze(1)  # (batch_size, 1, num_filters * len(kernel_sizes))
        
        # RNN processing
        rnn_out, _ = self.rnn(rnn_input)  # rnn_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Get the final hidden state
        rnn_out = self.dropout(rnn_out[:, -1, :])  # Take the last output (batch_size, hidden_dim * 2)
        
        # Fully connected layer
        output = self.fc(rnn_out)  # (batch_size, num_classes)
        
        return output

# Hyperparameters
vocab_size = len(vocab)
embed_dim = 100
num_classes = len(label_encoder.classes_)
hidden_dim = 128
num_layers = 2
kernel_sizes = [3, 4, 5]
num_filters = 100
dropout = 0.5

# Initialize the model
model = CNNRNNTextClassifier(vocab_size, embed_dim, num_classes, hidden_dim, num_layers, 
                           kernel_sizes, num_filters, dropout)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

train(model, train_loader, criterion, optimizer, num_epochs=5)

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Loss: {total_loss/len(test_loader)}')
    print(f'Accuracy: {100 * correct / total}%')

evaluate(model, test_loader, criterion)

torch.save(model.state_dict(), 'phishing_email_cnn_rnn.pth')