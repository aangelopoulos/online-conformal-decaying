import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from warnings import simplefilter
from core import get_online_quantile, dtACI
import pdb

if __name__ == "__main__":
    alpha=0.1
    data = pd.read_csv('.cache/theta.csv')
    y_columns = [f"V{j}" for j in range(2, 9921)]
    pred_columns = [f"Vhat{j}" for j in range(2, 9921)]
    score_columns = [f"S{j}" for j in range(2, 9921)]
    scores = data[score_columns].iloc[0].dropna().values
    etas = np.ones_like(scores)
    q_quantile_tracker = get_online_quantile(scores, scores[0], etas, alpha)
    q_dtACI = dtACI(scores, alpha)
