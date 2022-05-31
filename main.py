from SberOfSite import SberOfSite
from SmartLab import SmartLab
from interfax import Interfax
from primpres import Primpress
from bert_predictor import BertPredictor
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from flask import Flask, request, jsonify,make_response,abort
import pandas as pd
import numpy as np

from functions import transform_candles, inverse_transform_candles




app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

sber = SberOfSite()
smartlab = SmartLab()
interfax=Interfax()
primpres=Primpress()
bert = BertPredictor()


@app.route('/predict/<company_id>')
def predict(company_id):
    print(company_id)
    scl_prices = MinMaxScaler()
    scl_volume = RobustScaler()
    model = keras.models.load_model(f"model_{company_id}.h5")
    req = request.json
    print(req)
    candles = pd.DataFrame(req).reset_index(drop=True)
    candles = transform_candles(candles, scl_prices, scl_volume)
    prediction = model.predict(candles)
    prediction = inverse_transform_candles(prediction, scl_prices, scl_volume)

    return jsonify(
        {"open": str(round(prediction[0], 2)),
         "close": str(round(prediction[1], 2)),
         "high": str(round(prediction[2], 2)),
         "low": str(round(prediction[3], 2)),
         "volume": str(int(prediction[4]))}
    ),201


@app.route('/predict/<company_id>/<forward_candles>')
def predict_many(company_id, forward_candles):
    scl_prices = MinMaxScaler()
    scl_volume = RobustScaler()
    model = keras.models.load_model(f"model_{company_id}.h5")
    req = request.json
    candles = pd.DataFrame(req).reset_index(drop=True)
    candles = transform_candles(candles, scl_prices, scl_volume)
    prediction_list = []
    for _ in range(int(forward_candles)):
        prediction = model.predict(candles)
        prediction_inverse = inverse_transform_candles(prediction, scl_prices, scl_volume)
        prediction_list.append(
            {
                "open": str(round(prediction_inverse[0], 2)),
                "close": str(round(prediction_inverse[1], 2)),
                "high": str(round(prediction_inverse[2], 2)),
                "low": str(round(prediction_inverse[3], 2)),
                "volume": str(int(prediction_inverse[4]))
            }
        )
        candles = pd.DataFrame(candles[0]).reset_index(drop=True)
        candles = candles.iloc[1:, :]
        np.append(prediction[0], 1)
        prediction = prediction[0]
        prediction = np.append(prediction, [0])
        prediction = [prediction]
        candles = candles.append(pd.DataFrame(prediction).reset_index(drop=True))
        candles.iloc[-1, -1] = candles.iloc[-2, -1]
        candles = [candles]
        candles = np.array(candles)
    return jsonify(prediction_list),201

@app.route('/predict_bert', methods=['POST'])
def get_predict_bert():
    if not request.json:
        abort(400)
    result = bert.predict(request.json)

    return result, 201
    
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/data', methods=['POST'])
def get_data():
    if not request.json or not 'from' and 'id' in request.json:
        abort(400)
    id = int(request.json['id'])
    date = request.json['from']
    posts = []
    if request.json['source']=='all':
        print(request.json['source'])
        if request.json['id'] == '4':
            [posts.append(i) for i in sber.parse(date)]
        [posts.append(i) for i in smartlab.parse(id, date)]
        [posts.append(i) for i in interfax.parse(id, date)]
        [posts.append(i) for i in primpres.parse(id, date)]
    if request.json['source']=='smartlab':
        [posts.append(i) for i in smartlab.parse(id, date)]
    if request.json['source']=='interfax':
        [posts.append(i) for i in interfax.parse(id, date)]
    if request.json['source']=='primpres':
        [posts.append(i) for i in primpres.parse(id, date)]
    if request.json['source']=='sber':
        [posts.append(i) for i in sber.parse(id, date)]



    return jsonify(posts), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0')
