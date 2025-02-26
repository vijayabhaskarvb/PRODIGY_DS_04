import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import warnings
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.metrics import classification_report, accuracy_score

#Ignoring Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)


# Load training dataset
dataset = pd.read_csv('training_dataset.csv', header=None, names=['ID', 'Entity', 'Sentiment', 'Text'])

# Keep only relevant columns
dataset = dataset[['Sentiment', 'Text']]

# Remove null values
dataset = dataset.dropna(subset=['Text'])

# Clean text function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        return text
    else:
        return ""  # Return empty string for non-string values

# Apply text cleaning
dataset['Cleaned_Text'] = dataset['Text'].apply(clean_text)

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
dataset['Sentiment_Score'] = dataset['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment based on score
def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

dataset['Predicted_Sentiment'] = dataset['Sentiment_Score'].apply(classify_sentiment)

# Sentiment distribution plot
plt.figure(figsize=(6,4))
sns.countplot(data=dataset, x='Predicted_Sentiment', hue='Predicted_Sentiment', palette='coolwarm', legend=False)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Word clouds for each sentiment
for sentiment in ['Positive', 'Negative', 'Neutral']:
    text = ' '.join(dataset[dataset['Predicted_Sentiment'] == sentiment]['Cleaned_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Save processed training data to the local directory
dataset.to_csv('processed_training_dataset.csv', index=False)
print("Training Sentiment Analysis Complete. Processed data saved.")

# Load validation dataset
validation_dataset = pd.read_csv('validation_dataset.csv', header=None, names=['ID', 'Entity', 'Sentiment', 'Text'])

#Removing irrelevant Datas
validation_dataset = validation_dataset[validation_dataset['Sentiment'] != 'Irrelevant']

# Keep only relevant columns
validation_dataset = validation_dataset[['Sentiment', 'Text']]

# Remove null values
validation_dataset = validation_dataset.dropna(subset=['Text'])

# Apply the same text preprocessing function
validation_dataset['Cleaned_Text'] = validation_dataset['Text'].apply(clean_text)

# Sentiment analysis on validation dataset
validation_dataset['Sentiment_Score'] = validation_dataset['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment for validation data
validation_dataset['Predicted_Sentiment'] = validation_dataset['Sentiment_Score'].apply(classify_sentiment)

# Evaluate model performance
accuracy = accuracy_score(validation_dataset['Sentiment'], validation_dataset['Predicted_Sentiment'])
print(f'Validation Accuracy: {accuracy:.2f}')

# Generate classification report
print("\nClassification Report:\n")
print(classification_report(validation_dataset['Sentiment'], validation_dataset['Predicted_Sentiment'], zero_division=1))

# Save processed validation data
validation_dataset.to_csv('processed_validation_dataset.csv', index=False)
print("Validation Sentiment Analysis Complete. Processed data saved.")
