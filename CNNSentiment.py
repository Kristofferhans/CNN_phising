import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, RocCurveDisplay,
                             precision_recall_curve, PrecisionRecallDisplay)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuration
try:
    plt.style.use('seaborn-v0_8')  
except:
    plt.style.use('ggplot')  
    
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42
sns.set_palette("coolwarm")

def load_and_preprocess_data(filepath):
    """Load and preprocess the email dataset with enhanced cleaning."""
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=["Email Text"]).copy()
        df["Email Text"] = df["Email Text"].astype(str).str.strip()
        df["Label"] = df["Email Type"].apply(lambda x: 1 if x == "Phishing Email" else 0)
        
        #basic text cleaning
        df["clean_text"] = df["Email Text"].str.lower()
        df["clean_text"] = df["clean_text"].str.replace(r'[^\w\s]', '', regex=True)
        df["clean_text"] = df["clean_text"].str.replace(r'\d+', '', regex=True)
        df["clean_text"] = df["clean_text"].str.strip()
        
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
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'url_count': text.count('http://') + text.count('https://'),
        'has_attachment': int('attachment' in text.lower())
    }

def visualize_sentiment(df):
    """Create enhanced visualizations for sentiment analysis results."""
    plt.figure(figsize=(18, 15))
    
    #scatter plot with regression lines
    plt.subplot(2, 3, 1)
    sns.scatterplot(data=df, x='polarity', y='subjectivity', hue='Label', alpha=0.6)
    sns.regplot(data=df[df['Label']==0], x='polarity', y='subjectivity', scatter=False, color='blue')
    sns.regplot(data=df[df['Label']==1], x='polarity', y='subjectivity', scatter=False, color='red')
    plt.title("Sentiment Analysis: Phishing vs. Safe Emails")
    
    #distribution plots with KDE
    plt.subplot(2, 3, 2)
    sns.histplot(data=df, x='polarity', hue='Label', kde=True, element='step', stat='density')
    plt.title("Polarity Distribution")
    
    plt.subplot(2, 3, 3)
    sns.histplot(data=df, x='subjectivity', hue='Label', kde=True, element='step', stat='density')
    plt.title("Subjectivity Distribution")
    
    #boxplots with swarmplot
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df, x='Label', y='polarity')
    sns.swarmplot(data=df, x='Label', y='polarity', size=2, color='black', alpha=0.3)
    plt.xticks([0, 1], ['Safe', 'Phishing'])
    plt.title("Polarity by Email Type")
    
    #uppercase ratio comparison
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='Label', y='uppercase_ratio')
    plt.xticks([0, 1], ['Safe', 'Phishing'])
    plt.title("Uppercase Ratio by Email Type")
    
    #URL count comparison
    plt.subplot(2, 3, 6)
    sns.countplot(data=df, x='url_count', hue='Label')
    plt.title("URL Count Distribution")
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(X, y, text_features=None):
    """Train and evaluate a gradient boosting model with hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    
    #defining the pipeline with SMOTE for handling class imbalance
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        ))
    ])
    
    #hyperparameter grid for tuning
    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }
    
    #performing grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    print("\nTraining model with hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    #getting best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\nBest Parameters:", grid_search.best_params_)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
    
    #creating feature importance plot
    try:
        plt.figure(figsize=(10, 6))
        feature_importances = best_model.named_steps['xgb'].feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
    
    #confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
               annot=True, fmt='d', cmap='Blues',
               xticklabels=['Safe', 'Phishing'], 
               yticklabels=['Safe', 'Phishing'])
    plt.title("Confusion Matrix")
    plt.show()
    
    #ROC Curve
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()
    
    #precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.show()
    
    return best_model

def main():
    try:
        df = load_and_preprocess_data("Phishing_Email - Phishing_Email.csv")
        
        #generating sentiment and text features
        sentiment_features = df["Email Text"].apply(analyze_sentiment).apply(pd.Series)
        df = pd.concat([df, sentiment_features], axis=1)
        
        #visualising the data
        visualize_sentiment(df)
        
        #defining features for modeling
        features = ['polarity', 'subjectivity', 'word_count', 
                   'exclamation_count', 'question_count', 
                   'uppercase_ratio', 'url_count', 'has_attachment']
        
        #training and evaluate model
        model = train_and_evaluate_model(df[features], df['Label'])
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()