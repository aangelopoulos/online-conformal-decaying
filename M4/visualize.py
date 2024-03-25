import marimo

__generated_with = "0.3.4"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from warnings import simplefilter 
    return mo, np, pd, plt, simplefilter, sns


@app.cell
def __(pd):
    data = pd.read_csv('.cache/theta.csv')
    data
    return data,


@app.cell
def __(data, pd, plt, sns):
    # for each column in the dataframe, loop through and plot a rolling average over 30 points
    horizon=14
    rolling=30
    sns.set()
    sns.set_palette(sns.color_palette("pastel"))
    sns.set_context("talk")
    sns.set_style("white")
    y_columns = [f"V{j}" for j in range(2, 9921)]
    pred_columns = [f"Vhat{j}" for j in range(2, 9921)]
    score_columns = [f"S{j}" for j in range(2, 9921)]

    for _i, _row in data.iterrows():
        print(_i)
        fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(10,5), sharex=True)
        fig.autofmt_xdate(rotation=45)
        _y = _row[y_columns]; _yhat = _row[pred_columns]; _s = _row[score_columns]
        # Remove NaNs
        _y = _y.dropna(); _yhat = _yhat.dropna(); _s = _s.dropna()
        # Make x a list of dates starting at January 1, 2024 and incrementing daily
        _x_y = pd.date_range(start=_row['StartingDate'], periods=len(_y))
        _x_yhat = pd.date_range(start=_row['StartingDate'], periods=len(_yhat)).shift(2+horizon)
        _x_s = pd.date_range(start=_row['StartingDate'], periods=len(_s))
        sns.lineplot(ax=axs[0], x=_x_y, y=_y, alpha=0.5, label=r'$Y_t$')
        sns.lineplot(ax=axs[0], x=_x_yhat, y=_yhat, alpha=0.5, label=r'$\hat{f}_t(X_t)$')
        sns.lineplot(ax=axs[1], x=_x_s, y=_s.rolling(rolling, center=True).mean(), alpha=0.5)
        sns.despine(top=True, right=True)
        # Tilt x axis labels by 30 degrees
        axs[0].set_ylabel(f"y")
        axs[0].legend(loc="upper left")
        axs[1].set_ylabel(f"score (rolling {rolling})")
        for ax in axs:
            ax.set_xlabel(f"Time")
            ax.locator_params(axis='y', nbins=3)
        plt.tight_layout()
        plt.show()
    return (
        ax,
        axs,
        fig,
        horizon,
        pred_columns,
        rolling,
        score_columns,
        y_columns,
    )


if __name__ == "__main__":
    app.run()
