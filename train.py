import pickle
import autogluon
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# Define the LSTM model architecture for CPU
from gluonts.dataset.common import ListDataset
from gluonts.mx import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

import numpy as np

def DeepAR_forecast(series, train_ratio=0.8):
    """
    Uses DeepAR from GluonTS to forecast time series.
    
    Input: 
        series: np.array of floats, shape [B, T]
        train_ratio: float, ratio of data to be used for training
    Output: 
        pred: np.array of floats, shape [B, T - 1]
    """
    b_pred = []
    for b in range(series.shape[0]):
        # Convert the series to a GluonTS ListDataset format
        history = series[b, :int(train_ratio * series.shape[1])].tolist()
        pred = []
        training_data = ListDataset(
            [{"start": "2020-01-01", "target": history}],
            freq = "D"
        )
        
        # Define DeepAR estimator and train
        estimator = DeepAREstimator(freq="D", prediction_length=1, trainer=Trainer(epochs=10))
        predictor = estimator.train(training_data=training_data)
        
        for t in range(series.shape[1] - int(train_ratio * series.shape[1])):
            # Forecast the next step using the current data
            forecast_it, _ = make_evaluation_predictions(
                dataset=ListDataset(
                    [{"start": "2020-01-01", "target": history}], 
                    freq = "D"
                ), 
                predictor=predictor, 
                num_samples=1000
            )
            
            forecasts = list(forecast_it)
            forecasted_value = forecasts[0].mean[0]
            pred.append(forecasted_value)
            
            # Add the true observation to the history for the next prediction
            history.append(series[b, int(train_ratio * series.shape[1]) + t])
        
        b_pred.append(pred)
    
    b_pred = np.array(b_pred)
    return b_pred

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :]) 
        return out

def LSTM_predictor(series, train_ratio=0.8):
    """
    Implements the LSTM prediction algorithm using PyTorch.
    
    Input: 
        series: np.array of floats, shape [B, T]
        train_ratio: float, ratio of data to be used for training
    Output: 
        pred: np.array of floats, shape [B, T - 1]
    """
    
    # Hyperparameters
    input_dim = 1
    hidden_dim = 64
    num_layers = 8
    output_dim = 1
    num_epochs = 100
    learning_rate = 0.01
    
    b_pred = []
    history = series[:, :int(train_ratio * series.shape[1])].tolist()

    train_data = [torch.FloatTensor(hist).view(-1, 1, 1) for hist in history]
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model on all batches
    for epoch in tqdm(range(num_epochs)):
        for b in range(series.shape[0]):
            optimizer.zero_grad()
            outputs = model(train_data[b][:-1])
            loss = loss_function(outputs.squeeze(), train_data[b][1:].squeeze())
            loss.backward()
            optimizer.step()
    
    for b in range(series.shape[0]):
        # Forecasting
        pred = []
        for t in range(series.shape[1] - int(train_ratio * series.shape[1])):
            with torch.no_grad():
                test_inputs = torch.FloatTensor(history[b][-1:]).view(-1, 1, 1)
                yhat = model(test_inputs)
                pred.append(yhat.item())
                history[b].append(series[b, int(train_ratio * series.shape[1]) + t])
            
        b_pred.append(pred)
    b_pred = np.array(b_pred)

    return b_pred
def ETS_model(series, train_ratio=0.8):
    """
    Implements the Exponential Smoothing algorithm.
    
    Input: 
        series: np.array of floats, shape [B, T]
        train_ratio: float, ratio of data to be used for training
    Output: 
        pred: np.array of floats, shape [B, T - 1]
    """
    
    b_pred = []
    history = series[:, :int(train_ratio * series.shape[1])].tolist()
    for b in range(series.shape[0]):
        pred = []
        for t in range(series.shape[1] - int(train_ratio * series.shape[1])):
            model = ExponentialSmoothing(history[b], trend="additive", seasonal=None)
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=1)
            pred.append(yhat)
            history[b].append(series[b, int(train_ratio * series.shape[1]) + t])
        b_pred.append(pred)
    b_pred = np.array(b_pred)
    return b_pred

def ARIMA_model(series, train_ratio=0.8):
    """
    Input: series: np.array of floats, shape [B, T]
    Output: pred: np.array of floats, shape [B, T - 1]
    """

    b_pred = []
    history = series[:, :int(train_ratio * series.shape[1])].tolist()
    for b in range(series.shape[0]):
        pred = []
        for t in range(series.shape[1] - int(train_ratio * series.shape[1])):
            model = ARIMA(history[b], order=(2,1,2))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            pred.append(yhat)
            history[b].append(series[b, int(train_ratio * series.shape[1]) + t])
        b_pred.append(pred)
    b_pred = np.array(b_pred)
    return b_pred

def persistent_model(series, train_ratio=0.8):
    """
    y_t+1 = y_t

    Input: series: np.array of floats, shape [B, T]
    Output: pred: np.array of floats, shape [B, T - 1]
    """
    train_idx = int(series.shape[1] * train_ratio)
    pred = series[:, train_idx - 1: -1]
    return pred

def constant_model(series, train_ratio=0.8):
    """
    y_t+1 = y_t

    Input: series: np.array of floats, shape [B, T]
    Output: pred: np.array of floats, shape [B, T - 1]
    """
    pred = np.ones((series.shape[0], series.shape[1] - int(train_ratio * series.shape[1])), dtype=np.float32) * np.mean(series[:, int(train_ratio * series.shape[1]) - 42:int(train_ratio * series.shape[1])]).reshape(-1, 1)
    return pred
def evaluate(b_train, b_pred, b_label, name):

    # get rid of B dimension
    train, pred, label = b_train[-1], b_pred[-1], b_label[-1]
    plt.figure(figsize=(15, 7))

    # Plotting the training data
    plt.plot(train, label="Training Data", color="blue")

    # Plotting the actual test data
    plt.plot(np.arange(len(train), len(train) + len(label)), label, label="Actual Test Data", color="green")

    # Plotting the naive one-step ahead predictions using the fixed approach
    plt.plot(np.arange(len(train), len(train) + len(label)), pred, label="Naive Fixed Model Predictions", color="red", linestyle="--")

    plt.title("Naive Fixed Model One-Step Ahead Predictions vs Actual Test Data")
    plt.xlabel("Data Points")
    plt.ylabel("Bandwidth")
    plt.legend()
    plt.grid(True)
    # score = np.mean(np.abs(b_pred - b_label) / b_label)
    score = np.mean(np.abs(b_pred.squeeze() - b_label.squeeze()))
    plt.text(0.5, 0.5, 'Mean Absolute Error (MAE): {:.6f}'.format(score), transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5), fontsize=14)
    plt.savefig("%s.png" % name)
    return score

def autogluon_model(series, train_ratio=0.8):
    """
    Input: series: np.array of floats, shape [B, T]
    Output: pred: np.array of floats, shape [B, T - 1]
    """
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    data = [{"target": series[0], "start": pd.Period("01-01-2019", freq='D')}]
    data = TimeSeriesDataFrame(data)
    train_data = data[:int(train_ratio * data.shape[0])]
    test_data = data[int(train_ratio * data.shape[0]):]
    predictor = TimeSeriesPredictor(target='target', prediction_length=len(test_data), cache_predictions=True, eval_metric="MASE")
    predictor = predictor.fit(train_data)

    pred = predictor.predict(train_data)
    print(pred.info())
    print(pred.head())
    pred = pred['mean'].to_numpy()
    print(pred.shape)
    raise Exception("break")


    return pred
if __name__ == "__main__":
    # load data
    concat_data = []
    for i in range(27):
        if os.path.exists('bw_l_%02d.pkl' % i):
            with open('bw_l_%02d.pkl' % i, 'rb') as f:
                data = pickle.load(f)
                concat_data.extend(data)
    plt.figure(figsize=(15, 6))
    plt.plot(concat_data, markersize=4, linewidth=2, color='purple')
    plt.title("Bandwidth Data")
    plt.xlabel("Data Points")
    plt.ylabel("Bandwidth")
    plt.grid(True)
    plt.savefig("Bandwith.png")

    # divide data into 4 non-overlapping parts
    concat_data = np.array(concat_data)
    data_1 = concat_data[range(0, len(concat_data), 4)]
    data_2 = concat_data[range(1, len(concat_data), 4)]
    data_3 = concat_data[range(2, len(concat_data), 4)]
    data_4 = concat_data[range(3, len(concat_data), 4)]
    # TODO cast to same shape
    min_len = min(data_1.shape[0], data_2.shape[0], data_3.shape[0], data_4.shape[0])
    data = np.concatenate((data_1[np.newaxis, :min_len], data_2[np.newaxis, :min_len], data_3[np.newaxis, :min_len], data_4[np.newaxis, :min_len]), axis=0)

    # 0 - int(train_ratio * T) - 1: train
    # int(train_ratio * T): -1: test

    

    # data = data_2[np.newaxis, :]
    # data = concat_data[np.newaxis, :]

    # normalize data to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # pickle.dump(data[:, :int(0.8 * data.shape[1])], open('train.pkl', 'wb'))
    # pickle.dump(data[:, int(0.8 * data.shape[1]):], open('test.pkl', 'wb'))
    pred = DeepAR_forecast(data)
    print(pred.shape)
    persistent_score = evaluate(data[:, :int(0.8 * data.shape[1])], pred, data[:, int(0.8 * data.shape[1]):], "plt_persistent_model")
    print("Persistent model score: ", persistent_score)

    pred = constant_model(data)
    print(pred.shape)
    persistent_score = evaluate(data[:, :int(0.8 * data.shape[1])], pred, data[:, int(0.8 * data.shape[1]):], "plt_constant_model")
    print("Constant model score: ", persistent_score)

    pred = ARIMA_model(data)
    print(pred.shape)
    persistent_score = evaluate(data[:, :int(0.8 * data.shape[1])], pred, data[:, int(0.8 * data.shape[1]):], "plt_ARIMA_model")
    print("ARIMA model score: ", persistent_score)

    pred = ETS_model(data)
    print(pred.shape)
    persistent_score = evaluate(data[:, :int(0.8 * data.shape[1])], pred, data[:, int(0.8 * data.shape[1]):], "plt_ETS_model")
    print("ETS model score: ", persistent_score)

    pred = LSTM_predictor(data)
    print(pred.shape)
    persistent_score = evaluate(data[:, :int(0.8 * data.shape[1])], pred, data[:, int(0.8 * data.shape[1]):], "plt_LSTM_model")
    print("LSTM model score: ", persistent_score)

    raise Exception("break")
    print(data.shape)
    print(data_1[np.newaxis, :].shape, data_2.shape, data_3.shape, data_4.shape)
    raise Exception("break")

    # save data
    pickle.dump(data_1, open('data_1.pkl', 'wb'))
    pickle.dump(data_2, open('data_2.pkl', 'wb'))
    pickle.dump(data_3, open('data_3.pkl', 'wb'))
    pickle.dump(data_4, open('data_4.pkl', 'wb'))

    


    # Load data
    # with open('bw_l_00.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     print(data)
    # Train model
    # use autogluon time series prediction, data is a list of floats
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    # load pickle data
    for i in range(1, 5):
        with open('bw_l_%02d.pkl' % i, 'rb') as f:
            data = pickle.load(f)

            data = [{"target": data, "start": pd.Period("01-01-2019", freq='D')}]
            data = TimeSeriesDataFrame(data)
            break
            
    predictor = TimeSeriesPredictor.load('/home/kt19/bw_pred/AutogluonModels/ag-20230917_031650/predictor.pkl')

    # predictor = TimeSeriesPredictor(target='target', prediction_length=48, cache_predictions=True, eval_metric="MAPE").fit(data)
    predictions = predictor.predict(data)
    print(type(predictions))
    
