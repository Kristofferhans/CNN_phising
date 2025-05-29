import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import nltk

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load dataset
df = pd.read_csv(r"C:\Users\krist\Data science\island\CNN_phising\phishing_email_no_duplicates.csv")

# Extract sentiment score
def get_sentiment_score(text):
    return sia.polarity_scores(str(text))['compound']

df['sentiment_score'] = df['text_combined'].apply(get_sentiment_score)

# Prepare features and labels
X = df[['sentiment_score']].values  # Only using sentiment score
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
