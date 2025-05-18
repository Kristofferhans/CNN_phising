import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from textblob import TextBlob
import scipy

#loading dataset
df = pd.read_csv('phishing_email.csv')
df.dropna(subset=['text_combined', 'label'], inplace=True)

#Including sentiment polarity as a feature
df['sentiment'] = df['text_combined'].apply(lambda x: TextBlob(x).sentiment.polarity)

#feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['text_combined'])
y = df['label']

X_combined = scipy.sparse.hstack((X_tfidf, df[['sentiment']].values))
#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

#training XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

#prediction on test data
y_pred = model.predict(X_test)

#eval model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Phishing"]))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phishing"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Legit vs Phishing Emails")
plt.show()
