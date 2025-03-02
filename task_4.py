import numpy as np
import pandas as pd
import re
import nltk
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Load training dataset
dataset = pd.read_csv('dataset.csv', header=None, names=['ID', 'Entity', 'Sentiment', 'Text'])
dataset = dataset[['Sentiment', 'Text']]
dataset = dataset.dropna(subset=['Text'])

# Clean text function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        text = text.lower()  # Convert to lowercase
        return text
    return ""

dataset['Cleaned_Text'] = dataset['Text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(dataset['Cleaned_Text']).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(processed_features, dataset['Sentiment'], test_size=0.2, random_state=0)

# Model training
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

# Predictions
predictions = text_classifier.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

# Sentiment distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=dataset, x='Sentiment', hue='Sentiment', palette='coolwarm', legend=False)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Word clouds for each sentiment
for sentiment in dataset['Sentiment'].unique():
    text = ' '.join(dataset[dataset['Sentiment'] == sentiment]['Cleaned_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# New input prediction
new_input = ["@virginamerica Well, I didn'tâ€¦but NOW I DO! :-D"]
new_input_vectorized = vectorizer.transform(new_input).toarray()
new_prediction = text_classifier.predict(new_input_vectorized)
print(f"Predicted class: {new_prediction[0]}")

# Save processed dataset
dataset.to_csv('processed_dataset.csv', index=False)
print("Processed dataset saved.")
