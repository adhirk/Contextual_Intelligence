# Importing necessary libraries and packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE

import gensim
import numpy as np
from gensim.utils import simple_preprocess
import gensim.downloader


# Download the NLTK wordnet
nltk.download('wordnet')

# Instantiate a function for lemmatizing the text data
def lem_tokenizer(s):
    
    # remove punctuation from the string using string.punctuation
    for char in s:
        if char in punctuation:
            s = s.replace(char, '')
            
    # make the entire string lowercase
    s = s.lower()
    
    # split the string at each space to make the list of tokens (uncleaned)
    tokens = s.split()
    
    # save NLTK stop words to a variable
    stop_words = nltk.corpus.stopwords.words('english')
    
    # use list comprehension to create a list of the tokens that are NOT stop words
    tokens_new = [token for token in tokens if token not in stop_words]
    
    # create WordNetLemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()

    # list of part-of-speech tags
    pos_tags = ['v', 'n', 'a']
    
    # initiate empty list to collect lemmatized tokens
    tokens_lem = list()

    # loop through each token
    for token in tokens_new:

        # loop through each part-of-speech tag
        for pos_tag in pos_tags:

            # lemmatize each token using each part-of-speech tag
            token = wnl.lemmatize(word=token, pos=pos_tag)

        # append the lemmatized token to the new list
        tokens_lem.append(token)
    
    return tokens_lem


###############################################################################


# Instantiate a function for Stemming the text data

stemmer = nltk.stem.PorterStemmer()

# import the nltk stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords 

ENGLISH_STOP_WORDS = stopwords.words('english')

def remove_html_tags(text):
    pattern = re.compile(r'<.*?>')  
    return pattern.sub('', text)


def stem_tokenizer(sentence):
    # remove punctuation and set to lower case
    for punctuation_mark in string.punctuation:
        sentence = sentence.replace(punctuation_mark,'').lower()
        
    # remove digits using list comprehension
    sentence = ''.join([char for char in sentence if not char.isdigit()])
    
    # remove html tags
    sentence = remove_html_tags(sentence)

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []
    
    # remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)

    return listofstemmed_words

###############################################################################

# Instantiate a function for Sentence2Vec using Word2Vec

# Download the "word2vec-google-news-300" model
word2vec_model = gensim.downloader.load("word2vec-google-news-300")

# Access the model's word vectors
model = word2vec_model.vectors

def sentence2vec(text):
    """
    Embed a sentence by averaging the word vectors of the tokenized text. Out-of-vocabulary words are replaced by the zero-vector.
    -----
    
    Input: text (string)
    Output: embedding vector (np.array)
    """
    tokenized = simple_preprocess(text)
    
    word_embeddings = []
    for word in tokenized:
        # if the word is in the model then embed
        if word in word2vec_model:
            vector = word2vec_model[word]
        # add zeros for out-of-vocab words
        else:
            vector = np.zeros(300)
            
        word_embeddings.append(vector)
    
    # average the word vectors
    sentence_embedding = np.mean(word_embeddings, axis=0)
    
    return sentence_embedding
