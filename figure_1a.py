"""
figure_1a



Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 23 March 2023
"""

import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


from scipy.optimize import minimize_scalar
from sklearn.linear_model import Lasso, lasso_path

import solve_utils as sr


def srLASSO(A, b, lamda, verbose=False):
    m, n = A.shape
    b = sr._column_vectorize(b)

    primal_value, x_bar = sr._solve_primal(A, b, lamda)
    return x_bar


def sr_finetune_lamda(A, b, x0, lamda_bracket=None):
    if lamda_bracket is None:
        lamda_bracket = np.array([0.1, 10]) * sr.lamda_sr_best(A.shape[0])
    else:
        assert len(lamda_bracket) == 2

    def _objective(lamda):
        primal_value, x_bar = sr._solve_primal(A, b.reshape(-1, 1), lamda)
        h = x_bar.ravel() - x0.ravel()
        return np.log(h.dot(h))

    res = minimize_scalar(_objective, bounds=lamda_bracket, method="bounded")
    if not res.success:
        print("Did not succeed")
    return res.x, np.exp(_objective(res.x))


def _sr_coarse_tune_lamda(A, b, x0, lamda_c, width=None, num_alphas=None):
    if num_alphas is None:
        num_alphas = 11
    if width is None:
        width = 100
    alphas = np.geomspace(1 / width, width, num_alphas)

    lamdas = lamda_c * alphas

    x_bars = np.column_stack([srLASSO(A, b, lamda) for lamda in lamdas])
    h = x_bars - x0.reshape(-1, 1)
    errors = (h**2).sum(axis=0)
    idx_best = np.argmin(errors)
    lamda_best = lamdas[idx_best]
    return idx_best, lamda_best, lamdas


def _uc_coarse_tune_lamda(A, b, x0, lamda_c, width=None, num_alphas=None):
    if num_alphas is None:
        num_alphas = 11
    if width is None:
        width = 100
    alphas = np.geomspace(1 / width, width, num_alphas)

    lamdas = lamda_c * alphas

    lamdas_, x_bars = lasso_path(A, b, alphas=lamdas / m)[:2]
    x_bars = x_bars[:, ::-1]

    h = x_bars - x0.reshape(-1, 1)
    errors = (h**2).sum(axis=0)
    idx_best = np.argmin(errors)
    lamda_best = lamdas[idx_best]
    return idx_best, lamda_best, lamdas


def _get_width(lamda, lamdas, idx):
    width1 = lamda / lamdas[int(np.maximum(0, idx - 1))]
    width2 = lamdas[int(np.minimum(lamdas.size - 1, idx + 1))] / lamda
    width = np.maximum(width1, width2)
    return width


def sr_coarse_tune_lamda(A, b, x0, lamda_init):
    idx0, lamda0, lamdas0 = _sr_coarse_tune_lamda(A, b, x0, lamda_init)
    width0 = _get_width(lamda0, lamdas0, idx0)
    idx1, lamda1, lamdas1 = _sr_coarse_tune_lamda(A, b, x0, lamda0, width0, 31)
    width1 = _get_width(lamda1, lamdas1, idx1)
    idx2, lamda2, lamdas2 = _sr_coarse_tune_lamda(A, b, x0, lamda1, width1, 51)
    width2 = _get_width(lamda2, lamdas2, idx2)
    return lamda2, width2


def uc_coarse_tune_lamda(A, b, x0, lamda_init):
    idx0, lamda0, lamdas0 = _uc_coarse_tune_lamda(A, b, x0, lamda_init)
    width0 = _get_width(lamda0, lamdas0, idx0)
    idx1, lamda1, lamdas1 = _uc_coarse_tune_lamda(A, b, x0, lamda0, width0, 31)
    width1 = _get_width(lamda1, lamdas1, idx1)
    idx2, lamda2, lamdas2 = _uc_coarse_tune_lamda(A, b, x0, lamda1, width1, 51)
    width2 = _get_width(lamda2, lamdas2, idx2)
    return lamda2, width2


def sr_tune_lamda(A, b, x0, lamda_init):
    lamda2, width2 = sr_coarse_tune_lamda(A, b, x0, lamda_init)
    lamda_best, error_best = sr_finetune_lamda(
        A, b, x0, lamda_bracket=(lamda2 / width2, lamda2 * width2)
    )
    x_best = srLASSO(A, b, lamda_best)
    return lamda_best, error_best, x_best


def uc_finetune_lamda(A, b, x0, lamda_bracket):
    assert len(lamda_bracket) == 2

    def _objective(lamda):
        lasso = Lasso(alpha=lamda / A.shape[0], fit_intercept=False)
        lasso.fit(A, b.ravel())
        h = lasso.coef_.ravel() - x0.ravel()
        return h.dot(h)

    res = minimize_scalar(_objective, bounds=lamda_bracket, method="bounded")
    if not res.success:
        print("Did not succeed")
    return res.x, _objective(res.x)


def uc_tune_lamda(A, b, x0, lamda_init):
    lamda2, width2 = uc_coarse_tune_lamda(A, b, x0, lamda_init)
    lamda_best, error_best = uc_finetune_lamda(
        A, b, x0, lamda_bracket=(lamda2 / width2, lamda2 * width2)
    )
    x_best = (
        Lasso(alpha=lamda_best / A.shape[0], fit_intercept=False)
        .fit(A, b.ravel())
        .coef_
    )
    return lamda_best, error_best, x_best


def noisescale_experiment(A, x0, gammas=None):
    if gammas is None:
        gammas = np.geomspace(1e-3, 10, 11)
    lamda_init = sr.lamda_sr_best(m)
    results_sr = []
    results_uc = []
    for gamma in tqdm(gammas):
        b = A.dot(x0) + gamma * npr.randn(m)
        results_sr.append(sr_tune_lamda(A, b, x0, lamda_init))
        results_uc.append(uc_tune_lamda(A, b, x0, lamda_init))
    lamda_best_sr, error_best_sr, x_best_sr = zip(*results_sr)
    lamda_best_uc, error_best_uc, x_best_uc = zip(*results_uc)
    return (
        gammas,
        lamda_best_sr,
        error_best_sr,
        x_best_sr,
        lamda_best_uc,
        error_best_uc,
        x_best_uc,
    )


def run_trial(m, n, s, trial_number=None):
    A, _, x0 = sr.generate_data(m, n, s, gamma)
    (
        gammas,
        lamda_best_sr,
        error_best_sr,
        x_best_sr,
        lamda_best_uc,
        error_best_uc,
        x_best_uc,
    ) = noisescale_experiment(A, x0)
    df = pd.DataFrame(
        {
            "gammas": gammas,
            "lamda_best_sr": lamda_best_sr,
            "error_best_sr": error_best_sr,
            "x_best_sr": x_best_sr,
            "lamda_best_uc": lamda_best_uc,
            "error_best_uc": error_best_uc,
            "x_best_uc": x_best_uc,
        }
    )
    if trial_number:
        df["trial"] = trial_number
    return df


def make_plot(df, savefig=None):

    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 20
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(
        df.gammas,
        df.lamda_best_sr,
        ".",
        color="tab:red",
        alpha=0.5,
        markersize=15,
        label=r"$\rm{SR}$",
    )
    ax.plot(
        df.gammas,
        df.lamda_best_uc,
        ".",
        color="tab:blue",
        alpha=0.5,
        markersize=15,
        label=r"$\rm{UC}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma$", fontsize=24)
    ax.set_ylabel(r"$\lambda_{\rm{best}}$", fontsize=24)
    ax.legend(loc="upper left")
    fig.tight_layout()
    if isinstance(savefig, str):
        fig.savefig(savefig, bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")
    del fig, ax


if __name__ == "__main__":
    RUN_TRIAL = False
    m, n, s, gamma = 50, 100, 5, 0.25

    if RUN_TRIAL:
        df = pd.concat(
            [run_trial(m, n, s, i) for i in range(5)], axis=0
        ).reset_index(drop=True)

        df.to_csv(
            "data/example_introduction_sr_uc/parameter_noisescale_dependence.csv"
        )
    else:
        df = pd.read_csv(
            "data/example_introduction_sr_uc/parameter_noisescale_dependence.csv",
            index_col=0,
        )

    fig_dir = "fig/example_introduction_sr_uc"
    make_plot(
        df, savefig=os.path.join(fig_dir, "parameter_noisescale_dependence.pdf")
    )
