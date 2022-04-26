import os
import pandas as pd
import numpy as np
from .preprocess_text import preproc_text

import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

class Trainer:
    """
    The main class for train ML model
    """
    # (csv file with two columns: commenttext and spam)
    train_file_name = 'upload-train-data.csv'
    file_path = os.path.abspath(os.getcwd())+'/project/media/'

    df = None

    inf_spam_val_count = None
    inf_accuracy = None

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self,):
        self.df = pd.read_csv(self.file_path + self.train_file_name)

    def prepeare_data(self, steammer=True, lemmatizer=True):
        self.df['preproc_txt'] = self.df.commenttext.apply(lambda x: preproc_text(x, steamm=steammer, lemm=lemmatizer))
        # convert label to a numerical variable
        self.df['spam_num'] = self.df["spam"].astype(int)

        self.df = self.df[['preproc_txt', 'spam_num']]
        self.df['preproc_txt'].replace('', np.nan, inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)

        # get some info about df
        self.inf_spam_val_count = self.df['spam_num'].value_counts()

        self.df.to_csv(self.file_path+'preproc_data_res.csv')

        # define X and y (from the SMS data) for use with COUNTVECTORIZER
        X =  self.df.preproc_txt
        y =  self.df.spam_num

        # split X and y into training and testing sets
        # by default, it splits 75% training and 25% test
        # random_state=1 for reproducibility
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    def make_models(self, model_name):
        vect = CountVectorizer(
            # Dosn't work -> stop_words=ua_stop_words_list, # remove Ukranian stop words
            # ngram_range=(1, 3), # include 1-grams and 2-grams
            # min_df=2, # only keep terms that appear in at least 2 documents
        )
        vect.fit(self.X_train)

        # transform training data
        X_train_dtm = vect.transform(self.X_train)
        # equivalently: combine fit and transform into a single step
        # this is faster and what most people would do
        X_train_dtm = vect.fit_transform(self.X_train)
        X_test_dtm = vect.transform(self.X_test)

        # dump CountVectorizer model
        pickle.dump(vect, open(os.path.abspath(os.getcwd())+'/project/ml/models/cv-'+model_name, 'wb'))


        nb = MultinomialNB()
        nb.fit(X_train_dtm, self.y_train)

        # dump NB model
        pickle.dump(nb, open(os.path.abspath(os.getcwd())+'/project/ml/models/nb-'+model_name, 'wb'))

        y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
        # calculate AUC
        self.inf_accuracy = metrics.roc_auc_score(self.y_test, y_pred_prob)