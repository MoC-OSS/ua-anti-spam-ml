import os
import json
from datetime import datetime

from werkzeug.utils import secure_filename
from flask import (
    Flask,
    jsonify,
    send_from_directory,
    request,
    jsonify,
    redirect,
    url_for
)
from flask_sqlalchemy import SQLAlchemy
from .ml.predictor import Predictor


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


# @app.route("/")
# def hello_world():
#     return jsonify(hello="world")

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
      <p><input type=file name=file><input type=submit value=Upload>
    </form>
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
    { message: 'тест', rate: 0.8 }
    Rerurn json like:
    { time: 200, result: { spamRate: 0.988888, deleteRank: 0.8, isSpam: true, tensorToken: [0,1, 345,2,2,2] } }
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

@app.route("/api/set_model", methods=['POST'])
def api_set_model():
    """
    Waiting for:
    BODY
    { "model_name": "add-cv-model.pkl", "nb_model": false }
    Rerurn json like:
    { "status": 'OK', "model_type": 'nb' }
    """
    request_data = request.get_json()
    model_name = request_data['model_name']

    if model_name in os.listdir(app.config["ML_MODELS_FOLDER"]):
        if request_data['nb_model']: # check type of model
            # set nativeBayes model
            predictor.load_nb_model(nbm_name=model_name)
            app.logger.info('New ML NativeBayes model was set ("%s") ', model_name)
            return jsonify({"status": 'OK', "model_type": 'nb'})
        else:
            # set countVectorize model
            predictor.load_cv_model(cvm_name=model_name)
            app.logger.info('New ML CountVectorize model was set ("%s") ', model_name)
            return jsonify({"status": 'OK', "model_type": 'cv'})
    else:
        return jsonify({"status": 'Model {} not found'.format(model_name)})