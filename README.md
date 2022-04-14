<h1 align="center">Flask with ML spam prediction</h1>


This is a Flask App with ML **Naive Bayes** model for check message on the Spam statement, Ukrainian words oriented. 

Also we train and use here **Count Vectorizer**(so 2 different models  will be used for prediction).

# Installation

For project startup uses `docker-compose.yml`. In case with prod you can use `docker-compose.prod.yml`.

You can also use `python venv` for development. All dependent python packages in **requirements.txt**

# Configuration

*Docker-compose* use configurations from `.env.dev` or `.env.prod` files.

For *Flask* config uses `services/web/project/config.py`.

# App routes

- **/static/<path:filename>** (get static file)
- **/upload** (upload file to *media* folder)
- **/web/spam-predict** (simple UI form for checking Spam)
- **/api/spam-predict** (api request for checking Spam)
- **/api/set_model** (api request for setting other ml models)

# Usage Example

For example we will check spam staitment uses json request

```
POST http://host:5000/api/spam-predict
BODY
{
    "message": "Я спав а він стріляв",
    "rate": 0.8
}
```

the response will be like

```
{
    "result": {
        "deleteRank": 0.8,
        "isSpam": false,
        "spamRate": 0.2528499848085302,
        "tensorToken": "я спат а стріл"
    },
    "time": 6.2989999999999995
}
```