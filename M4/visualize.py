import marimo

__generated_with = "0.3.4"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from warnings import simplefilter 
    return mo, np, os, pd, plt, simplefilter, sns, sys


@app.cell
def __(pd):
    data = pd.read_csv('.cache/theta.csv')
    data
    return data,


@app.cell
def __():
    y_columns = [f"V{j}" for j in range(2, 9921)]
    pred_columns = [f"Vhat{j}" for j in range(2, 9921)]
    score_columns = [f"S{j}" for j in range(2, 9921)]
    q_adaptive_columns = [f"q_adaptive{j}" for j in range(2, 9921)]
    q_decaying_columns = [f"q_decaying{j}" for j in range(2, 9921)]
    q_fixed_columns = [f"q_fixed{j}" for j in range(2, 9921)]
    q_dtACI_columns = [f"q_dtACI{j}" for j in range(2, 9921)]
    return (
        pred_columns,
        q_adaptive_columns,
        q_decaying_columns,
        q_dtACI_columns,
        q_fixed_columns,
        score_columns,
        y_columns,
    )


@app.cell
def __(
    data,
    pd,
    q_adaptive_columns,
    q_decaying_columns,
    q_dtACI_columns,
    q_fixed_columns,
    score_columns,
):
    # Metrics
    metrics = []
    for _i, _row in data.iterrows():
        _id = _row["M4id"]
        _scores = _row[score_columns].dropna().values
        _q_adaptive = _row[q_adaptive_columns].dropna().values
        _q_decaying = _row[q_decaying_columns].dropna().values
        _q_fixed = _row[q_fixed_columns].dropna().values
        _q_dtACI = _row[q_dtACI_columns].dropna().values
        metrics += [pd.DataFrame([{
            "id" : _id,
            "method" : "adaptive",
            "coverage" : (_scores <= _q_adaptive).mean(),
            "variance" : (_q_adaptive.var())/(_scores.var()),
        }])]
        metrics += [pd.DataFrame([{
            "id" : _id,
            "method" : "decaying",
            "coverage" : (_scores <= _q_decaying).mean(),
            "variance" : (_q_decaying.var())/(_scores.var()),
        }])]
        metrics += [pd.DataFrame([{
            "id" : _id,
            "method" : "fixed",
            "coverage" : (_scores <= _q_fixed).mean(),
            "variance" : (_q_fixed.var())/(_scores.var()),
        }])]
        metrics += [pd.DataFrame([{
            "id" : _id, 
            "method" : "DTACI",
            "coverage" : (_scores <= _q_dtACI).mean(),
            "variance" : (_q_dtACI.var())/(_scores.var()),
        }])]

    metrics = pd.concat(metrics)
    return metrics,


@app.cell
def __(metrics):
    metrics.to_markdown(index=False)
    return


@app.cell
def __(
    data,
    pd,
    plt,
    pred_columns,
    q_adaptive_columns,
    q_decaying_columns,
    q_dtACI_columns,
    q_fixed_columns,
    score_columns,
    sns,
    y_columns,
):
    # for each column in the dataframe, loop through and plot a rolling average over 30 points
    horizon=14
    rolling=30
    sns.set()
    sns.set_palette(sns.color_palette("pastel"))
    sns.set_context("talk")
    sns.set_style("white")

    for _i, _row in data.iterrows():
        print(_i)
        fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(10,5), sharex=True)
        fig.autofmt_xdate(rotation=45)
        _y = _row[y_columns]; _yhat = _row[pred_columns]; _s = _row[score_columns]; _q_decaying = _row[q_decaying_columns]; _q_fixed = _row[q_fixed_columns]; _q_dtACI = _row[q_dtACI_columns]; _q_adaptive = _row[q_adaptive_columns]
        # Remove NaNs
        _y = _y.dropna(); _yhat = _yhat.dropna(); _s = _s.dropna(); _q_decaying = _q_decaying.dropna(); _q_fixed = _q_fixed.dropna(); _q_dtACI = _q_dtACI.dropna(); _q_adaptive = _q_adaptive.dropna()
        # Make x a list of dates starting at January 1, 2024 and incrementing daily
        _x_y = pd.date_range(start=_row['StartingDate'], periods=len(_y))
        _x_yhat = pd.date_range(start=_row['StartingDate'], periods=len(_yhat)).shift(2+horizon)
        _x_s = pd.date_range(start=_row['StartingDate'], periods=len(_s))
        sns.lineplot(ax=axs[0], x=_x_y, y=_y, alpha=0.5, label=r'$Y_t$')
        sns.lineplot(ax=axs[0], x=_x_yhat, y=_yhat, alpha=0.5, label=r'$\hat{f}_t(X_t)$')
        #sns.lineplot(ax=axs[1], x=_x_s, y=_s.rolling(rolling, center=True).quantile(0.9), alpha=0.5, label='scores (rolling)')
        sns.lineplot(ax=axs[1], x=_x_s, y=_s, color='k', alpha=0.2, label='scores')
        sns.lineplot(ax=axs[1], x=_x_s, y=_q_adaptive, alpha=0.8, label='adaptive')
        sns.lineplot(ax=axs[1], x=_x_s, y=_q_decaying, alpha=0.8, label='decaying')
        sns.lineplot(ax=axs[1], x=_x_s, y=_q_fixed, alpha=0.8, label='fixed')
        sns.lineplot(ax=axs[1], x=_x_s, y=_q_dtACI, alpha=0.2, label='DTACI')
        
        sns.despine(top=True, right=True)
        # Tilt x axis labels by 30 degrees
        axs[0].set_ylabel(f"y")
        axs[0].legend(loc="upper left")
        axs[1].set_ylabel(f"quantile")
        axs[1].legend(loc="upper right", bbox_to_anchor=(1.2,1.2))
        for ax in axs:
            ax.set_xlabel(f"Time")
            ax.locator_params(axis='y', nbins=3)
        plt.tight_layout()
        plt.show()
    return ax, axs, fig, horizon, rolling


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
