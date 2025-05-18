import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
from typing import List, Tuple, Iterator, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class EmailDataset(Dataset):
    """Custom Dataset for loading and processing email data."""
    
    def __init__(self, df: pd.DataFrame, text_pipeline: callable):
        """
        Args:
            df: DataFrame containing email data
            text_pipeline: Function to convert text to indices
        """
        self.df = df
        self.text_pipeline = text_pipeline

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.df.iloc[idx]['text_combined']
        label = self.df.iloc[idx]['label']        
        text_indices = self.text_pipeline(text)
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class CNNRNNTextClassifier(nn.Module):
    """Hybrid CNN-RNN text classification model."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, 
                 hidden_dim: int = 128, num_layers: int = 2,
                 kernel_sizes: List[int] = [3, 4, 5], 
                 num_filters: int = 100, dropout: float = 0.5):
        """
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of word embeddings
            num_classes: Number of output classes
            hidden_dim: Dimension of RNN hidden state
            num_layers: Number of RNN layers
            kernel_sizes: List of kernel sizes for convolutional layers
            num_filters: Number of filters per convolutional layer
            dropout: Dropout probability
        """
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
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #embedding layer
        embedded = self.embedding(x)  
        
        #CNN processing
        embedded = embedded.unsqueeze(1) 
        
        #applying each convolution and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded)).squeeze(3)  
            pooled_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  
            conv_outputs.append(pooled_out)
            
        cnn_out = torch.cat(conv_outputs, 1) 
        
        rnn_input = cnn_out.unsqueeze(1)  
        
        #RNN processing
        rnn_out, _ = self.rnn(rnn_input) 
        
        #getting the final hidden state with dropout
        rnn_out = self.dropout(rnn_out[:, -1, :])  
        
        #final classification
        output = self.fc(rnn_out)  
        
        return output


def find_data_file(filename: str, search_paths: List[str]) -> str:
    """Locate a data file in possible directories."""
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(f"Could not find {filename} in any of these locations: {search_paths}")


def yield_tokens(data_iter: Iterator[str], tokenizer: callable) -> Iterator[List[str]]:
    """Yield tokens from text data."""
    for text in data_iter:
        yield tokenizer(text)


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader to handle variable length sequences."""
    text_list, label_list = [], []
    for (text, label) in batch:
        text_list.append(text)
        label_list.append(label)
    return (
        torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True), 
        torch.stack(label_list)
    )


def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    num_epochs: int = 5,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> None:
    """Train the model with validation."""
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        #validating
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')


def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.Module,
    label_encoder: LabelEncoder,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the model and return accuracy and confusion matrix data."""
    model.eval()
    model.to(device)
    
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            #storing predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
    
    #generating confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = label_encoder.classes_
    
    #plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return accuracy, all_labels, all_preds


def main():
    #configuration
    DATA_FILE = "phishing_email.csv"
    SEARCH_PATHS = [
        r"C:\Users\krist\Data science\island\master",
        r"C:\Users\krist\Data science\island",
        r"C:\Users\krist\Data science\island\CNN_phising",
        os.path.dirname(os.path.abspath(__file__)),
    ]
    
    #model hyperparameters
    BATCH_SIZE = 32
    EMBED_DIM = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_FILTERS = 100
    KERNEL_SIZES = [3, 4, 5]
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    RANDOM_STATE = 42
    
    try:
        #loading and preprocess data
        data_path = find_data_file(DATA_FILE, SEARCH_PATHS)
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path).dropna()
        
        #printing columns for verification
        print("\nColumns in dataset:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        
        #encoding labels with correct column name
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
        
        #checking class distribution
        print("\nClass distribution:")
        print(df['label'].value_counts())
        
        #splitting data into 60% train, 20% validation, 20% test
        train_df, temp_df = train_test_split(
            df, test_size=0.4, random_state=RANDOM_STATE, stratify=df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df['label']
        )
        
        #tokenizer and vocabulary (built only from training data)
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(
            yield_tokens(train_df['text_combined'], tokenizer), 
            specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        
        #text processing pipeline
        def text_pipeline(text: str) -> List[int]:
            return [vocab[token] for token in tokenizer(text)]
        
        #creating datasets and data loaders
        train_dataset = EmailDataset(train_df, text_pipeline)
        val_dataset = EmailDataset(val_df, text_pipeline)
        test_dataset = EmailDataset(test_df, text_pipeline)
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
        )
        
        #initializing model
        model = CNNRNNTextClassifier(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            num_classes=len(label_encoder.classes_),
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            kernel_sizes=KERNEL_SIZES,
            num_filters=NUM_FILTERS,
            dropout=DROPOUT
        )
        
        #loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        #training with validation and evaluation
        print("\nStarting training...")
        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
        
        print("\nEvaluating model on test set...")
        accuracy, true_labels, pred_labels = evaluate_model(model, test_loader, criterion, label_encoder)
        
        # printing classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=label_encoder.classes_))
        
        #saving model with additional information
        model_info = {
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'label_encoder': label_encoder,
            'config': {
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'num_filters': NUM_FILTERS,
                'kernel_sizes': KERNEL_SIZES,
                'dropout': DROPOUT
            }
        }
        
        torch.save(model_info, 'phishing_email_cnn_rnn.pth')
        print(f"\nModel saved with test accuracy: {accuracy:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print(f"1. The data file '{DATA_FILE}' exists in one of these locations:")
        for path in SEARCH_PATHS:
            print(f"   - {path}")
        print("2. You have proper read permissions for the file")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()