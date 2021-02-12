from sklearn.linear_model import LogisticRegression
import argparse
import os
import nltk
import re
import spacy
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
!pip install contractions
import contractions  
ds_path="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

def install_dependancies(){
    nltk.download('stopwords')
}

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
    contractions_dict = contractions.contractions_dict
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df.sentiment = x_df.sentiment.apply(lambda s: 1 if s == "positive" else 0)
    reviews = pd.get_dummies(x_df.review , prefix = "review")
    x_df.drop("reviews", inplace=True, axis=1)
    x_df = x_df.join(reviews)

    x_df['review'] = x_df['review'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text().strip())
    x_df['review'] = x_df['review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
    x_df['review'] = x_df['review'].str.replace("[^a-zA-Z]", " ")
    x_df['review'] = list(map(clean_text, x_df.review_cleaned))
    
    combined_reviews = ' '.join(x_df['review'])
    text_series = pd.Series(combined_reviews.split())
    freq_comm = text_series.value_counts()

    rare_words = freq_comm[-38000:-1]
    x_df['review'] = x_df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in rare_words]))
    y_df = x_df.pop("sentiment")
    return x_df, y_df

ds= TabularDatasetFactory.from_delimited_files(path=ds_path)

x, y = clean_data(ds)

# TODO: Split data into train and test sets.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)

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
    joblib.dump(value=model,filename="./outputs/model.joblib")

if __name__ == '__main__':
    main()
 