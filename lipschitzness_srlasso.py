"""
lipschitzness_srlasso

For the final figure in Section 7 of the paper

Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 24 March 2023
"""
import os
import pickle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

from lasso import srLASSO
from utils import get_tstamp, mkdir

from solve_utils import lipschitz_bound_lamda, lipschitz_bound_b_lamda


DO_GENERATE_DATA = False


def basic_test_plot(lamdas, lipschitz_ub, errors):
    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 14
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(lamdas[lipschitz_ub > 0], lipschitz_ub[lipschitz_ub > 0])
    ax.plot(lamdas, errors)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()
    plt.close("all")
    del fig, ax


def make_lipschitzness_plot(
    lamdas,
    lipsch_empir,
    lamda_ref,
    L_ub1,
    L_ub2=None,
    savefig=None,
    ax=None,
    lamda_nmz=None,
):

    AX_NONE = ax is None

    if AX_NONE:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if lamda_nmz is None:
        xvar = lamdas
    else:
        xvar = lamda_nmz

    ax.plot(
        xvar,
        lipsch_empir,
        linestyle="dashed",
        label=r"$\|\bar{x}(\lambda) - \bar{x}(\bar{\lambda})\|$",
    )
    ax.plot(
        xvar,
        np.abs(lamdas - lamda_ref) * L_ub1,
        label=r"$L(\bar{\lambda}) \cdot |\lambda - \bar{\lambda}|$",
    )
    if L_ub2 is not None:
        ax.plot(
            xvar,
            np.abs(lamdas - lamda_ref) * L_ub2,
            label=r"$L(\bar{b}, \bar{\lambda}) \cdot |\lambda - \bar{\lambda}|$",
        )

    if lamda_nmz is None:
        ax.axvline(
            lamda_ref,
            ls="dashed",
            c="tab:purple",
            label=r"$\lambda = \bar{\lambda}$",
        )
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_xlabel(r"$\lambda$")
    # ax.legend()

    if AX_NONE:
        fig.tight_layout()
        if isinstance(savefig, str):
            fig.savefig(savefig, bbox_inches="tight")
        else:
            plt.show()
        return fig, ax
    return ax


def generate_results_data(
    m=100, n=200, s=3, gamma=0.1, seed=2023, directory=None, tstamp=None
):
    if directory is None:
        directory = "data/srlasso_lipschitzness"
    mkdir(directory)
    if tstamp is None:
        tstamp = get_tstamp()

    # Problem data
    if isinstance(seed, int):
        np.random.seed(2023)

    A = np.random.randn(m, n) / m**0.5
    x0 = np.zeros(n)
    x0[:s] = m + np.random.randn(s) * m**0.5
    noise = np.random.randn(m)
    b = A.dot(x0) + gamma * noise

    problem_data = {
        "m": m,
        "n": n,
        "s": s,
        "gamma": gamma,
        "A": A,
        "x0": x0,
        "b": b,
    }

    lamda_sr = 1.1 * norm.ppf(1 - 0.05 / (2 * n))
    alphas = np.logspace(-2, 2, 301)
    lamda_vec = lamda_sr * alphas
    x_bars, errors, lamdas = srLASSO(A, b, x0, lamda_vec)

    opt_data = {
        "x_bars": x_bars,
        "errors": errors,
        "lamdas": lamdas,
    }

    L_pw_lamda = np.zeros_like(lamdas)
    L_pw_b_lamda = np.zeros_like(lamdas)
    for j in range(lamdas.size):
        L_pw_lamda[j] = lipschitz_bound_lamda(x_bars[:, j], A, b, lamdas[j])
        L_pw_b_lamda[j] = lipschitz_bound_b_lamda(x_bars[:, j], A, b, lamdas[j])

    lip_ub_pointwise = {
        "L_pw_lamda": L_pw_lamda,
        "L_pw_b_lamda": L_pw_b_lamda,
    }

    idx_best = np.argmin(errors)
    lamda_best = lamdas[idx_best]
    x_bar_best = x_bars[:, idx_best]
    L_lamda = lipschitz_bound_lamda(x_bar_best, A, b, lamda_best)
    L_b_lamda = lipschitz_bound_b_lamda(x_bar_best, A, b, lamda_best)
    L_empir = la.norm(x_bars - x_bar_best.reshape(-1, 1), axis=0)

    lip_ub_best = {
        "idx_best": idx_best,
        "lamda_best": lamda_best,
        "x_bar_best": x_bar_best,
        "L_lamda": L_lamda,
        "L_b_lamda": L_b_lamda,
        "L_empir": L_empir,
    }

    fname = (
        f"srlasso_lipschitzness_{tstamp}_m{m}_n{n}_s{s}_gamma{gamma:.1g}.pkl"
    )
    with open(os.path.join(directory, fname), "wb") as fp:
        pickle.dump(
            {
                "problem_data": problem_data,
                "opt_data": opt_data,
                "lip_ub_pointwise": lip_ub_pointwise,
                "lip_ub_best": lip_ub_best,
            },
            fp,
        )


def load_data(tstamp_ms=None, tstamp_mg=None, directory=None):
    """Need matching m_vec and gamma_vec for given tstamps"""
    # LOAD DATA
    if directory is None:
        directory = "data/srlasso_lipschitzness"
    if tstamp_ms is None:
        tstamp_ms = "20230202-223711-961013"
    if tstamp_mg is None:
        tstamp_mg = "20230202-223927-066589"

    data_ms = {}
    data_mg = {}
    for m in m_vec:
        for s in s_vec:
            fname = (
                f"srlasso_lipschitzness_{tstamp_ms}_m{m}_n{n}_s{s}_gamma0.1.pkl"
            )
            with open(os.path.join(directory, fname), "rb") as fp:
                data_ms[(m, s)] = pickle.load(fp)

    for m in m_vec:
        for gamma in gamma_vec:
            fname = f"srlasso_lipschitzness_{tstamp_mg}_m{m}_n{n}_s7_gamma{gamma:.1g}.pkl"
            with open(os.path.join(directory, fname), "rb") as fp:
                data_mg[(m, gamma)] = pickle.load(fp)
    return data_ms, data_mg


def make_ms_plot(data_ms, savefig=None):
    """for (m, s)."""
    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 18
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.formatter.min_exponent"] = 2

    fig, ax = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for jj, m in enumerate(m_vec):
        for ii, s in enumerate(s_vec):
            lamdas = data_ms[(m, s)]["opt_data"]["lamdas"]
            L_empir = data_ms[(m, s)]["lip_ub_best"]["L_empir"]
            lamda_best = data_ms[(m, s)]["lip_ub_best"]["lamda_best"]
            L_ub1 = data_ms[(m, s)]["lip_ub_best"]["L_lamda"]
            lamda_nmz = lamdas / lamda_best
            make_lipschitzness_plot(
                lamdas,
                L_empir,
                lamda_best,
                L_ub1,
                ax=ax[ii, jj],
                lamda_nmz=lamda_nmz,
            )
            ax[ii, jj].set_title(r"$(s, m) = (" + f"{s}, {m}" + r")$", size=22)
            ax[ii, jj].axis("tight")
            ax[ii, jj].set_xlim(0.5, 2)
            ax[ii, jj].set_ylim(-1e-3, 2)
    ax[-1, 0].legend(loc="upper right")
    fig.tight_layout()
    if isinstance(savefig, str):
        fig.savefig(savefig, bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")
    del fig, ax


def make_mg_plot(data_mg, savefig=None):
    """for (m, gamma)"""
    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 18
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
    for jj, m in enumerate(m_vec):
        for ii, gamma in enumerate(gamma_vec):
            if ii > 3:
                continue
            lamdas = data_mg[(m, gamma)]["opt_data"]["lamdas"]
            L_empir = data_mg[(m, gamma)]["lip_ub_best"]["L_empir"]
            lamda_best = data_mg[(m, gamma)]["lip_ub_best"]["lamda_best"]
            L_ub1 = data_mg[(m, gamma)]["lip_ub_best"]["L_lamda"]
            lamda_nmz = lamdas / lamda_best
            make_lipschitzness_plot(
                lamdas,
                L_empir,
                lamda_best,
                L_ub1,
                ax=ax[ii, jj],
                lamda_nmz=lamda_nmz,
            )
            ax[ii, jj].set_title(
                r"$(\gamma, m) = (" + f"{gamma}, {m}" + r")$", size=18
            )
            ax[ii, jj].axis("tight")
            ax[ii, jj].set_xlim(0.5, 2)
            ax[ii, jj].set_ylim(-1e-3, 2)
        ax[3, 0].set_yticks([0, 1, 2], ["0", "1", "2"])
    ax[-1, 0].legend(loc="lower right")
    fig.tight_layout()
    if isinstance(savefig, str):
        fig.savefig(savefig, bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")
    del fig, ax


if __name__ == "__main__":

    seed = 2023
    n = 200
    s_vec = [3, 7, 15]
    m_vec = [50, 100, 150, 200]
    gamma_vec = [0.1, 0.5, 1, 5, 10]

    if DO_GENERATE_DATA:
        tstamp_ms = get_tstamp()
        for m in tqdm(m_vec, desc="m"):
            for s in tqdm(s_vec, desc="s"):
                generate_results_data(m, n, s, seed=seed, tstamp=tstamp_ms)

        tstamp_mg = get_tstamp()
        for m in tqdm(m_vec, desc="m"):
            for gamma in tqdm(gamma_vec, desc="gamma"):
                generate_results_data(
                    m, n, 7, gamma=gamma, seed=seed, tstamp=tstamp_mg
                )
    data_ms, data_mg = load_data(
        tstamp_ms="20230202-223711-961013", tstamp_mg="20230202-223927-066589"
    )

    m_string = "_".join([str(x) for x in m_vec])
    s_string = "_".join([str(x) for x in s_vec])
    gamma_string = "_".join([str(x) for x in gamma_vec])
    gamma_string = gamma_string.replace(".", "")

    fig_dir = "fig/srlasso_lipschitzness/"
    savefig_ms = os.path.join(
        fig_dir, f"srlasso_lipschitzness_m_{m_string}_s_{s_string}.pdf"
    )
    savefig_mg = os.path.join(
        fig_dir, f"srlasso_lipschitzness_m_{m_string}_gamma_{gamma_string}.pdf"
    )

    make_ms_plot(data_ms, savefig_ms)
    make_mg_plot(data_mg, savefig_mg)
