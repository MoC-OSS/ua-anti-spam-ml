import os
import pickle
import pandas as pd
from .preprocess_text import preproc_text

from flask import render_template

class Predictor:
    """
    The main class for predictions
    """
    cv_model = None # CountVectorizer model
    nb_model = None # NaiveBayes model
    text_series = None # text after preprocessing
    text_dtm = None # input matrix for NaiveBayes model
    # for res
    pred_prob = None
    spam_class = None

    def __init__(self, ):
        #init Predictor with default models
        self.load_cv_model()
        self.load_nb_model()

    def load_cv_model(self, cvm_name='add-cv-model.pkl'):
        self.cv_model = pickle.load(open(os.path.abspath(os.getcwd())+'/project/ml/models/'+cvm_name, 'rb'))

    def load_nb_model(self, nbm_name='add-nb-model.pkl'):
        self.nb_model = pickle.load(open(os.path.abspath(os.getcwd())+'/project/ml/models/'+nbm_name, 'rb'))  

    def prepare_data(self, text_list, steammer=True, lemmatizer=True):
        pd_series = pd.Series(text_list)
        self.text_series = pd_series.apply(lambda x: preproc_text(x, steamm=steammer, lemm=lemmatizer))
    
        self.text_dtm = self.cv_model.transform(self.text_series)

    def spam_class_predict(self,):
        self.spam_class = self.nb_model.predict(self.text_dtm)

    def spam_score_predict(self,):
        self.pred_prob = self.nb_model.predict_proba(self.text_dtm)[:, 1]

    def get_html_result(self,):
        """return template for Flask with model predictionresult"""

        df = pd.DataFrame()
        df['text_tokens'] = self.text_series
        if self.spam_class is not None:
            df['predict_class'] = self.spam_class
        if self.pred_prob is not None:
            df['pred_prob'] = self.pred_prob

        return render_template('spam_res_df.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
