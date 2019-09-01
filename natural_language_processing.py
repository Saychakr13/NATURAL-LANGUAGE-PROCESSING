# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk

# for downloading stopwords 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    # removing everything(like punctuations) other than english letters and replacing the removed items by space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # converting all characters into lower case
    review = review.lower()
    
    #converting a sentence into a list where each word is an element of the list
    review = review.split()
    
    # changes similar words to a single word. "loved","loving","love" will be changed to "love"
    ps = PorterStemmer()
    
    # removing irrelevant words(stopwords) like "this","that","the" etc
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #reconstructing the revievs from array back into a sentence
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
