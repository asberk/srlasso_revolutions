"""
viz

Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 16 March 2023
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from utils import mkdir


def _compute_mean_aggregated_statistic(
    dframe, column="uniqueness_sufficiency", aggfunc=None
):
    if aggfunc is None:
        aggfunc = "mean"
    aggdf = pd.pivot_table(
        dframe,
        index="lamda",
        columns="gamma",
        values=column,
        aggfunc=aggfunc,
        dropna=False,
    )
    lamdas = aggdf.index.values
    gammas = aggdf.columns.values
    return aggdf, lamdas, gammas


def _pcolormesh_coords(gammas, lamdas):
    d_gamma = np.mean(np.diff(np.log10(gammas)))
    d_lamda = np.mean(np.diff(np.log10(lamdas)))
    log_gamma_coords = (
        np.log10(gammas[0]) + (np.arange(gammas.size + 1) - 0.5) * d_gamma
    )
    log_lamda_coords = (
        np.log10(lamdas[0]) + (np.arange(lamdas.size + 1) - 0.5) * d_lamda
    )
    gamma_coords = 10**log_gamma_coords
    lamda_coords = 10**log_lamda_coords
    return np.meshgrid(lamda_coords, gamma_coords)


def plot_empirical_inexact_uniqueness_sufficiency(
    dframe, gamma_subset=None, savefig=None, fig_dir=None
):
    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 20
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["mathtext.fontset"] = "cm"

    cmap = sns.color_palette("mako", as_cmap=True)

    if fig_dir is None:
        fig_dir = "fig/srlasso_uniqueness_sufficiency"
    mkdir(fig_dir)

    dframe2 = dframe.copy()
    dframe2.loc[~dframe.inexact, "uniqueness_sufficiency"] = np.nan

    (
        uniqueness_sufficiency_mean,
        lamdas,
        gammas,
    ) = _compute_mean_aggregated_statistic(
        dframe2, "uniqueness_sufficiency", np.nanmean
    )
    Lamda_Coords, Gamma_Coords = _pcolormesh_coords(gammas, lamdas)

    if gamma_subset is None:
        gamma_subset = [gammas.min(), np.median(gammas), gammas.max()]
    if not isinstance(gamma_subset, (list, tuple, np.ndarray)):
        gamma_subset = [gamma_subset]
    subset = dframe.loc[dframe.gamma.isin(gamma_subset)]
    error_mean, lamdas_, gammas_ = _compute_mean_aggregated_statistic(
        subset, "error_value"
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.05},
        layout="constrained",
    )

    ax0, ax1 = ax[0], ax[1]
    ax2 = ax0.twinx()  #
    pcm = ax2.pcolormesh(
        Lamda_Coords,
        Gamma_Coords,
        uniqueness_sufficiency_mean.values.T,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\gamma$", fontsize=24)
    ax2.yaxis.grid(False, which="both")
    ax2.yaxis.set_minor_locator(NullLocator())
    plt.colorbar(pcm, cax=ax1)

    vline_vals = {}
    vline_colors = {}
    for gamma in gammas_:
        line = ax0.plot(error_mean.index, error_mean.loc[:, gamma], label=gamma)
        vline_vals[gamma] = error_mean.loc[:, gamma].idxmin()
        vline_colors[gamma] = line[0].get_color()
        ax0.axvline(
            vline_vals[gamma],
            color=vline_colors[gamma],
            linestyle="dashed",
        )
    ax0.set_yscale("log")
    ax0.set_xlabel(r"$\lambda$", fontsize=24)
    ax0.set_ylabel(r"$\|\bar{x}(\lambda) - x^{\sharp}\|$", fontsize=24)
    ax0.autoscale(enable=True, axis="x", tight=True)
    ax0.legend()
    ax0.set_zorder(ax2.get_zorder() + 1)
    ax0.patch.set_visible(False)
    if isinstance(savefig, str):
        fig.savefig(os.path.join(fig_dir, savefig), bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")
    del fig, ax


def plot_uniqueness_sufficiency_crosssection(
    results, xlim=None, ylim=None, save_dir=None, savefig=None
):
    results.loc[results.aux_value == np.inf, "aux_value"] = np.nan

    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 18
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(
        results.lamda,
        results.lamda,
        ls="dashed",
        label=r"$\lambda$",
        color="tab:red",
    )
    ax.plot(
        results.lamda,
        results.aux_value,
        label=r"$\|A_{I^C}^{\top}(\bar{y} + \bar{z})\|_{\infty}$",
        color="tab:blue",
    )
    ax.plot(
        results.lamda,
        results.error_value,
        label=r"$\|\bar{x} - x^{\sharp}\|_{2}$",
        color="tab:purple",
    )
    ax.fill_between(
        results.lamda,
        0,
        1,
        where=np.isclose(results.lamda.values, results.aux_value.values)
        | results.aux_value.isna(),
        color="#EEEEEE",
        alpha=1.0,
        transform=ax.get_xaxis_transform(),
    )
    ax.axvline(
        results.lamda.loc[results.error_value.idxmin()],
        linestyle="dashed",
        color="tab:purple",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.autoscale(enable=True, axis="x", tight=True)
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save_dir is None:
        save_dir = "fig/srlasso_uniqueness_sufficiency"
    mkdir(save_dir)

    if isinstance(savefig, str):
        fig.savefig(
            os.path.join(save_dir, savefig),
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close("all")
    del fig, ax
