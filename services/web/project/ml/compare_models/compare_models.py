import imp
import os
import pandas as pd

from ..predictor import Predictor

class CompareModels():
    test_phradses_df = None

    file_path = os.path.abspath(os.getcwd())+'/project/ml/compare_models/test-data.csv'

    predictor_old = None
    predictor_new = None

    def __init__(self,):
        self.read_test_phrases()

    def read_test_phrases(self, ):
        self.test_phradses_df = pd.read_csv(self.file_path, dtype={'message': str, 'manual_class': bool,})
        self.test_phradses_df = self.test_phradses_df.reset_index()

    def load_models(self, old_cv_model, old_nb_model, new_cv_model, new_nb_model):
        # activate CV and NB old models 
        self.predictor_old = Predictor()
        self.predictor_old.load_nb_model(nbm_name=old_nb_model)
        self.predictor_old.load_cv_model(cvm_name=old_cv_model)
        # activate CV and NB new models 
        self.predictor_new = Predictor()
        self.predictor_new.load_nb_model(nbm_name=new_nb_model)
        self.predictor_new.load_cv_model(cvm_name=new_cv_model)

    def compeare(self,):
        res = dict()

        df_len = 0
        is_better_counter = 0

        for inex, row in self.test_phradses_df.iterrows():
            self.predictor_old.prepare_data(row['message'], steammer=True, lemmatizer=True)
            self.predictor_old.spam_score_predict()    
            old_score = list(self.predictor_old.pred_prob)[0]

            self.predictor_new.prepare_data(row['message'], steammer=True, lemmatizer=True)
            self.predictor_new.spam_score_predict() 
            new_score = list(self.predictor_new.pred_prob)[0]

            is_better = False
            if row['manual_class'] is True:
                # spam
                if old_score < new_score:
                    is_better = True
            else:
                # not spam
                if old_score > new_score:
                    is_better = True

            if is_better:
                is_better_counter +=1

            res[inex] = {
                'test_phrase': row['message'],
                'old_score': old_score,
                'new_score': new_score,
                'is_better': is_better,
                'manual_spam': row['manual_class'],
            }
            df_len +=1

        better_percentage = is_better_counter *100 / df_len
        print(df_len, is_better_counter)
        
        return res, '{:.2f}'.format(better_percentage)