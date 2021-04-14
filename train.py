from sklearn.linear_model import LogisticRegression
import argparse
import os
import nltk
import re
import spacy
import json
import numpy as np
import pandas as pd
#from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
import contractions  
from collections import Counter   # for getting freq of words
ds_path="https://raw.githubusercontent.com/Varsh18/IMDB-Moview-Review-Analysis/master/IMDB-Dataset.csv"
contractions_dict = contractions.contractions_dict

!pip install contractions
!pip install nltk
def install_dependancies():
    nltk.download('stopwords')

def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions_dict:
                new_text.append(contractions_dict[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)    
    # remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        
    return text

def clean_data(data):
    install_dependancies()
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df.sentiment = x_df.sentiment.apply(lambda s: 1 if s == "positive" else 0)
    x_df['review'] = x_df['review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
    x_df['review'] = x_df['review'].str.replace("[^a-zA-Z]", " ")
    x_df['review'] = list(map(clean_text, x_df.review))

    review_string = " ".join(review for review in x_df["review"])
    tweet_words = review_string.split(' ')
    tweet_word_freq = Counter(tweet_words)
    five_most_common_words = tweet_word_freq.most_common(5000)
    common_word_list =[]
    for i in range(len(five_most_common_words)):
      common_word_list.append(five_most_common_words[i][0])
    x_df['review'] = x_df['review'].apply(lambda x: ' '.join([word for word in x.split() if word in common_word_list]))
    X = x_df['review'].values
    y = x_df['sentiment'].values
    return X,y

ds= TabularDatasetFactory.from_delimited_files(path=ds_path)

x, y = clean_data(ds)

# TODO: Split data into train and test sets.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
vectorizer = CountVectorizer()
vectorizer.fit(x_train)

X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)
test_data = X_test[1,:]
test_data_array = test_data.toarray()
test_data_list = test_data_array.tolist()
print("len test_data_list",len(test_data_list))
print("len test_data_list 0",len(test_data_list[0]))
with open("test_data.txt", "w") as fp:
    json.dump(test_data_list, fp)

run = Run.get_context(allow_offline=True)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs("outputs",exist_ok=True)
    joblib.dump(value=model, filename="outputs/bestmodel.pkl")

if __name__ == '__main__':
    main()
