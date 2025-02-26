import numpy as np
import pandas as pd
import re
import nltk
import warnings
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning, module="nltk")
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

dataset = pd.read_csv('training_dataset.csv')
print(dataset.shape)
print(dataset.head())

X = dataset.iloc[:, 3].values
y = dataset.iloc[:, 2].values
print(y)

processed_features = []

for sentence in range(0, len(X)):
   
    processed_feature = re.sub(r'\W', ' ', str(X[sentence]))

    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()
print(processed_features)

X_train, X_test, y_train, y_test = train_test_split(processed_features, y, test_size=0.2, random_state=0)

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

new_input = ["@virginamerica Well, I didn'tâ€¦but NOW I DO! :-D"]
new_input_vectorized = vectorizer.transform(new_input).toarray()

new_prediction = text_classifier.predict(new_input_vectorized)

print(f"Predicted class: {new_prediction[0]}")
