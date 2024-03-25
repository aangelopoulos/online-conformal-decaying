import os
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from darts import TimeSeries
from darts.models.forecasting.theta import FourTheta
from darts.models.forecasting.nbeats import NBEATSModel
from tqdm import tqdm
import multiprocessing as mp
from warnings import simplefilter

if __name__ == "__main__":
    # Parameters
    horizon = 14
    total_length = 9920+1 # Determined by original dataframe
    modelname = "Theta"

    # Load data
    data = pd.read_csv('Daily-train.csv')
    data.rename({"V1": "M4id"}, axis=1, inplace=True)
    info = pd.read_csv('M4-info.csv')
    info = info[info.M4id.str.contains("D")]
    data = data.merge(info, on='M4id')

    # Compile TimeSeries objects
    series = {}
    value_columns = [f"V{j}" for j in range(2, 9921)]
    for _i, _row in data.iterrows():
        _y = _row[value_columns]
        # Remove NaNs
        _y = _y.dropna()
        # Make x a list of dates starting at January 1, 2024 and incrementing daily
        _x = pd.date_range(start=_row['StartingDate'], periods=len(_y))
        _ts = TimeSeries.from_times_and_values(_x,_y)
        series[_row["M4id"]] = _ts
        if _i > 20:
            break

    # Make predictions
    if modelname == "Theta":
        model = FourTheta(2)
    elif modelname == "NBEATSModel":
        model = NBEATSModel(input_chunk_length=horizon, output_chunk_length=horizon, n_epochs=100)
    else:
        raise ValueError(f"Model {modelname} not recognized.")
    print("Making predictions...")
    outputs = {_k : model.historical_forecasts(series[_k], forecast_horizon=horizon) for _k in tqdm(series.keys())}
    _df_list = []
    for _k in outputs.keys():
        _yhat = outputs[_k].values().squeeze()
        _padded_output = np.concatenate([[np.NaN]*2, _yhat, (total_length-2-len(_yhat))*[np.NaN]])
        _df = {f"Vhat{j}" : _padded_output[j] for j in range(2, 9921)}
        _df["M4id"] = _k
        _df = pd.DataFrame([_df])
        _df_list += [_df]
    _pred_df = pd.concat(_df_list)
    data = data.merge(_pred_df, on='M4id')
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    for j in range(2, 9921):
        data[f"S{j}"] = (data[f"V{j}"] - data[f"Vhat{j}"]).abs()


    # Save predictions
    os.makedirs('.cache', exist_ok=True)
    data.to_csv(f".cache/{modelname}.csv", index=False)
