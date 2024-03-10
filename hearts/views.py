from sklearn.model_selection import train_test_split
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from django.conf import settings
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier


# Text Preprocessing Function
def clean_text(sent):
    stop_words = set(stopwords.words('english'))
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()
    words = word_tokenize(sent.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    sent = ' '.join(words)
    return sent


# Text Lemmatization Function
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return text


# Load the data
csv_file = os.path.join(settings.BASE_DIR, 'Symptom2Disease.csv')
data = pd.read_csv(csv_file)

# Clean and lemmatize the text data
data['text'] = data['text'].apply(clean_text)
data['text'] = data['text'].apply(lemmatize_text)

# Shuffle the data
data = shuffle(data, random_state=42)

# Select features and target
X = data['text']
y = data['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Text feature extraction using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X_train = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test = tfidf_vectorizer.transform(X_test).toarray()

# Train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)


class ProcessInput(APIView):
    def post(self, request, *args, **kwargs):
        text = request.data.get('text')

        text = clean_text(text)
        text = lemmatize_text(text)

        text_vectorized = tfidf_vectorizer.transform([text]).toarray()

        predicted_label = dt_classifier.predict(text_vectorized)[0]

        return Response({'predicted': predicted_label}, status=status.HTTP_200_OK)
