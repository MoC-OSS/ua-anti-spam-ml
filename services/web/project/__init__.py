import os
import json
from datetime import datetime
import io
import csv

from werkzeug.utils import secure_filename
from flask import (
    Flask,
    jsonify,
    send_from_directory,
    request,
    jsonify,
    abort,
    redirect,
    url_for
)
from flask_sqlalchemy import SQLAlchemy
from .ml.predictor import Predictor
from .ml.trainer import Trainer
from .ml.src.translit_convertor import ru_translit as translit
from .ml.src.translit_convertor import special_characters as spec_ch

import pandas as pd

app = Flask(__name__)
app.config.from_object("project.config.Config")

db = SQLAlchemy(app)

predictor = Predictor()

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(128), unique=True, nullable=False)
    active = db.Column(db.Boolean(), default=True, nullable=False)

    def __init__(self, email):
        self.email = email


@app.route('/healthy')
def healthy():
    db.engine.execute('SELECT 1')
    return ''

@app.route("/static/<path:filename>")
def staticfiles(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["MEDIA_FOLDER"], filename))
    return """
    <!doctype html>
    <title>upload new File</title>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file><input type=submit value="Upload CSV">
    </form>
    <small>(csv file with two columns: <b>commenttext</b> and <b>spam</b>)</small>
    """

@app.route("/web/spam-predict", methods=['GET', 'POST'])
def predict():
    predictor = Predictor()

    if request.method == "POST":
        test_message = []
        test_message.append(request.form['text'])

        predictor.prepare_data(test_message, steammer=True, lemmatizer=True)
        predictor.spam_score_predict()
        predictor.spam_class_predict()

        return predictor.get_html_result()

    return """
    <!doctype html>
    <title>Put your message for test</title>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=text name=text><input type=submit value=Check>
    </form>
    """


# ===========
#API
# ===========
@app.route("/api/spam-predict", methods=['POST'])
def api_predict():
    """
    Waiting for:
    BODY
    { "message": "Я спав а він стріляв", "rate": 0.8 }
    Rerurn json like:
    { "result": { "deleteRank": 0.8, "isSpam": false, "spamRate": 0.251405040105228, "tensorToken": "я спат а стріл" }, "time": 4.634 }
    """
    time_start_req = datetime.now()
    test_message = []

    request_data = request.get_json()

    test_message.append(request_data['message'])

    predictor.prepare_data(test_message, steammer=True, lemmatizer=True)
    predictor.spam_score_predict()
    # predictor.spam_class_predict()

    is_spam = False
    if list(predictor.pred_prob)[0] >= request_data['rate']:
        is_spam = True

    req_sec = ((datetime.now() - time_start_req).total_seconds())*1000 # in milliseconds
    res_dict = dict(
        time=req_sec,
        result={
            "spamRate": list(predictor.pred_prob)[0],
            "deleteRank": request_data['rate'],
            "isSpam": is_spam,
            "tensorToken": predictor.text_series[0],
        }
    )

    return jsonify(res_dict)

@app.route("/api/set-models", methods=['POST'])
def api_set_model():
    """
    Waiting for:
    BODY
    { "cv_model_name": "add-cv-model.pkl", "nb_model_name": "add-nb-model.pkl" }
    Rerurn json like:
    { "status": 'OK', "model_type": 'nb' }
    """
    request_data = request.get_json()
    cv_model_name = request_data['cv_model_name']
    nb_model_name = request_data['nb_model_name']

    if cv_model_name and nb_model_name in os.listdir(app.config["ML_MODELS_FOLDER"]):
        # set nativeBayes model
        predictor.load_nb_model(nbm_name=nb_model_name)
        app.logger.info('New ML NativeBayes model was set ("%s") ', nb_model_name)

        # set countVectorize model
        predictor.load_cv_model(cvm_name=cv_model_name)
        app.logger.info('New ML CountVectorize model was set ("%s") ', cv_model_name)
            
        return jsonify({"status": 'OK', "cv_model_name": cv_model_name, "nb_model_name": nb_model_name})
    else:
        return abort(400, "Model name not found")

@app.route("/api/train-data-upload", methods=['POST'])
def api_train_data_upload():
    """
    (csv file with two columns: commenttext and spam)
    Waiting for:
    BODY binary file
    """
    upload_data = request.data.decode("utf-8")

    if request.content_type not in ['text/csv',]:
        return abort(400, "Unexpected Content-Type, expecting: text/csv")
    else:
        train_csv_name = 'upload-train-data.csv'
        f = open(os.path.join(app.config["MEDIA_FOLDER"], train_csv_name), "w", encoding="utf-8")
        f.write(upload_data)
        f.close()

        return jsonify({"status": 'OK', "info": 'The new file for train ML model was uploaded'})

@app.route("/api/train-models", methods=['POST'])
def api_train_models():
    """
    nativeBayes and countVectorize model will be trained
    Waiting for:
    BODY
    { "save_model_name": "new-model.pkl" }
    Rerurn json like:
    { "status": 'OK', "nb_model_name": 'nb-new-model.pkl', "cv_model_name": 'cv-new-model.pkl' }
    """
    app.logger.info('Start train new ML models')

    trainer = Trainer()
    trainer.train_file_name = 'upload-train-data.csv'

    time_start_req = datetime.now()

    request_data = request.get_json()
    save_model_name = request_data['save_model_name']

    trainer.prepeare_data(steammer=True, lemmatizer=True)
    trainer.make_models(model_name=save_model_name)

    app.logger.info('New ML Models was trained ("nb-%s", "cv-%s") ', save_model_name, save_model_name)
    req_sec = ((datetime.now() - time_start_req).total_seconds())*1000 # in milliseconds
    return jsonify({"status": 'OK', 
                    "info": 'The new Models was trained (nb-{}, cv-{})'.format(save_model_name, save_model_name),
                    "time": req_sec,
                    "accuracy": trainer.inf_accuracy,
                    "df_spam_val_count": "spam-{}, notSpam-{}".format(trainer.inf_spam_val_count[1], trainer.inf_spam_val_count[0])
                })

@app.route("/api/translit-convertor", methods=['POST'])
def api_translit_convertor():
    """
    Waiting for:
    BODY
    { "string": "Ці хлопці перші беруть на себе удар", "type": "ru" }
        or, for revers
    { "string": "Tsі hloptsі pershі berut' na sebe udar", "type": "cyr" }
    Rerurn json like:
    { "res": "Ці хлопці перші берутЬ на себе удар", "status": "OK", "type": "to_cyr" }
    """
    request_data = request.get_json()
    translit_type = request_data['type']
    text = request_data['string']

    res = None
    if translit_type == 'ru':
        res = translit.ru_to_cyrillic(text)
    elif translit_type == 'cyr':
        res = translit.cyrillic_to_ru(text)
    else:
        return abort(400, "Unexpected type {}, available: [ru, cyr]".format(translit_type))

    return jsonify({
            "status": 'OK', 
            "res": res,
            "type": translit_type,
        })

@app.route("/api/spec-symb", methods=['POST'])
def api_spec_symb():
    """
    Waiting for:
    BODY
    { "string": "Взрывается как мини яде₽ка. No πrivet"}
    Rerurn json like:
    { "res": "Взрывается как мини ядерка. Но привет", "status": "OK" }
    """
    request_data = request.get_json()
    text = request_data['string']

    # replacing special symbols (like ₽, @, 0, € etc..)
    res = spec_ch.replace_specsymb(text)
    # additional checking for eng symbols
    res = translit.cyrillic_to_ru(res)

    return jsonify({
            "status": 'OK', 
            "res": res,
        })
