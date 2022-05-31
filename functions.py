import numpy as np


def transform_candles(candles, scl_prices, scl_volume):
    candles = candles.iloc[-120:, :]
    price_columns = candles.iloc[:, 0:4]
    volume_columns = candles.iloc[:, 4]
    sentiment_column = candles.iloc[:, -1]
    np_price_scales = scl_prices.fit_transform(price_columns)
    np_volume_scales = scl_volume.fit_transform(volume_columns.values.reshape(-1, 1))
    candles = np.c_[np_price_scales, np_volume_scales]
    candles = np.c_[candles, sentiment_column]
    candles = [candles]
    candles = np.array(candles)
    return candles


def inverse_transform_candles(prediction, scl_prices, scl_volume):
    prices = scl_prices.inverse_transform(prediction[:, 0:4].reshape(1, -1))
    volumes = scl_volume.inverse_transform(prediction[:, -1].reshape(1, -1))
    return np.c_[prices, volumes][0]
