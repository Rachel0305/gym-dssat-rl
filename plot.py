import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gym_dssat_pdi.envs.utils import utils as dssat_utils
import numpy as np
import os

def plot_results(df):
    ax = sns.boxplot(data=df)
    ax = sns.stripplot(data=df, color=".2", ax=ax, size=4, alpha=.5)
    ax.set_xlabel("policy")
    ax.set_ylabel("reward")
    plt.savefig('./figures/evaluation.pdf')

if __name__ == '__main__':
    assert os.path.exists('./output/evaluation_rewards.pkl')

    for dir in ['./figures']:
        dssat_utils.make_folder(dir)

    numpy_seed = 123
    np.random.seed(numpy_seed)  # for reproducible strip plot
    results_path = './output/evaluation_rewards.pkl'
    with open(results_path, 'rb') as handle:
        results = pickle.load(handle)

    results_df = pd.DataFrame.from_dict(results)
    plot_results(results_df)