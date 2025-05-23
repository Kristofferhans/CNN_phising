import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # <-- move this here
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple
import numpy as np


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_pipeline: callable):
        self.df = df
        self.text_pipeline = text_pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.df.iloc[idx]['feature']
        label = self.df.iloc[idx]['label']
        return torch.tensor(self.text_pipeline(text), dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    texts, labels = zip(*batch)
    return pad_sequence(texts, batch_first=True), torch.stack(labels)

class CNNRNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim, num_layers, kernel_sizes, num_filters, dropout):
        super(CNNRNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.rnn = nn.LSTM(
            input_size=num_filters * len(kernel_sizes),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        convs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pools = [torch.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        cnn_out = torch.cat(pools, dim=1).unsqueeze(1)
        rnn_out, _ = self.rnn(cnn_out)
        out = self.dropout(rnn_out[:, -1, :])
        return self.fc(out)

def load_model_and_vocab(model_path: str):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_cfg = checkpoint['config']
    vocab = checkpoint['vocab']
    label_encoder = checkpoint['label_encoder']
    
    model = CNNRNNTextClassifier(
        vocab_size=len(vocab),
        embed_dim=model_cfg['embed_dim'],
        num_classes=len(label_encoder.classes_),
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        kernel_sizes=model_cfg['kernel_sizes'],
        num_filters=model_cfg['num_filters'],
        dropout=model_cfg['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, vocab, label_encoder

def main():
    model_path = 'phishing_email_cnn_rnn.pth'
    data_path = 'Phishing_Email - Phishing_Email.csv'

    # Load model
    model, vocab, label_encoder = load_model_and_vocab(model_path)
    model.eval()

    # Tokenizer and text pipeline
    tokenizer = get_tokenizer("basic_english")
    def text_pipeline(text): return [vocab[token] for token in tokenizer(text)]

    # Load and prepare data
    df = pd.read_csv(data_path, index_col=0).dropna()
    print(df.columns.tolist())
    label_map = {'legitimate': 0, 'phishing': 1}
    df['label'] = df['label'].map(label_map)

    # Now you can safely stratify or split
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    test_dataset = EmailDataset(test_df, text_pipeline)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = label_encoder.classes_

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Phishing Email')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
