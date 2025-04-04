import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, RocCurveDisplay)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Configuration
try:
    plt.style.use('seaborn-v0_8')  
except:
    plt.style.use('ggplot')  
    
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42
sns.set_palette("coolwarm")

def load_and_preprocess_data(filepath):
    """Load and preprocess the email dataset."""
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=["Email Text"]).copy()
        df["Email Text"] = df["Email Text"].astype(str).str.strip()
        df["Label"] = df["Email Type"].apply(lambda x: 1 if x == "Phishing Email" else 0)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def analyze_sentiment(text):
    """Perform sentiment analysis using TextBlob with enhanced text processing."""
    text = str(text).strip()
    analysis = TextBlob(text)
    
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity,
        'word_count': len(text.split()),
        'char_count': len(text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
    }

def visualize_sentiment(df):
    """Create visualizations for sentiment analysis results."""
    plt.figure(figsize=(15, 12))
    
    #scatter plot
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='polarity', y='subjectivity', hue='Label', alpha=0.6)
    plt.title("Sentiment Analysis: Phishing vs. Safe Emails")
    
    #distribution plots
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='polarity', hue='Label', kde=True, element='step')
    plt.title("Polarity Distribution")
    
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='subjectivity', hue='Label', kde=True, element='step')
    plt.title("Subjectivity Distribution")
    
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='Label', y='polarity')
    plt.xticks([0, 1], ['Safe', 'Phishing'])
    plt.title("Polarity by Email Type")
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(X, y):
    """Train and evaluate a classification model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
    
    #confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
               annot=True, fmt='d', cmap='Blues',
               xticklabels=['Safe', 'Phishing'], 
               yticklabels=['Safe', 'Phishing'])
    plt.title("Confusion Matrix")
    plt.show()
    
    # ROC Curve
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()
    
    return pipeline

def main():
    try:
        df = load_and_preprocess_data("Phishing_Email - Phishing_Email.csv")
        sentiment_features = df["Email Text"].apply(analyze_sentiment).apply(pd.Series)
        df = pd.concat([df, sentiment_features], axis=1)
        
        visualize_sentiment(df)
        
        features = ['polarity', 'subjectivity', 'word_count', 
                   'exclamation_count', 'question_count', 'uppercase_ratio']
        model = train_and_evaluate_model(df[features], df['Label'])
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()