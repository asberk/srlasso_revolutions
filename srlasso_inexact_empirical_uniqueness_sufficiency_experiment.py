"""
srlasso_inexact_empirical_uniqueness_sufficiency_experiment

Creates Figure 3 in the paper.

Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 16 March 2023
"""
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from solve import srLASSO_uniqueness
from solve_utils import generate_data, lamda_sr_best
from utils import get_tstamp, mkdir
from viz import plot_empirical_inexact_uniqueness_sufficiency


def run_trial(m, n, s, gamma, alphas=None, data_dir=None):
    if data_dir is None:
        data_dir = "data/srlasso_uniqueness_sufficiency"
    mkdir(data_dir)

    A, b, x0 = generate_data(m, n, s, gamma)
    lamda = lamda_sr_best(m) / m**0.5

    tstamp = get_tstamp()
    (
        results,
        x_bars,
        y_bars,
        z_bars,
    ) = srLASSO_uniqueness(A, b, lamda, x0, alphas=alphas)

    results["m"] = m
    results["n"] = n
    results["s"] = s
    results["gamma"] = gamma

    data = [A, b, x0]

    data_string = f"{tstamp}_m{m}_n{n}_s{s}_gamma{gamma}"
    data_string = data_string.replace(".", "")
    results.to_csv(os.path.join(data_dir, f"results_{data_string}.csv"))
    np.savez_compressed(
        os.path.join(data_dir, f"objects_{data_string}.npz"),
        x_bars,
        y_bars,
        z_bars,
    )
    np.savez_compressed(
        os.path.join(data_dir, f"objects2_{data_string}.npz"), *data
    )
    return results, x_bars, y_bars, z_bars, data


def big_data_experiment(
    m, n, s, n_trials=20, gammas=None, alphas=None, data_dir=None
):
    if gammas is None:
        gammas = np.logspace(-2, 1, 7, endpoint=True)
    if alphas is None:
        alphas = np.logspace(-1, 1, 31, endpoint=True)

    tstamp = get_tstamp()
    if data_dir is None:
        data_dir = f"data/srlasso_uniqueness_sufficiency/{tstamp}/"

    for i in tqdm(range(n_trials), desc="trials"):
        for gamma in tqdm(gammas, desc="gamma"):
            run_trial(m, n, s, gamma, alphas, data_dir)


def load_big_data_experiment(load_dir=None):
    """
    Returns
    -------
    dframe: pd.DataFrame
        All tabular results data in a dataframe, for trials, gammas and lamdas.
    solutions: list of (x_bars, y_bars, z_bars) tuples
        Each tuple member is a 2D np.ndarray
    solution_parms: pd.DataFrame
        The index for each tuple element in solutions corresponds with the same
        loc in solution_parms.
    """
    if load_dir is None:
        load_dir = "data/srlasso_uniqueness_sufficiency/"
        load_dirs = sorted(
            [
                dirpath
                for subdir in os.listdir(load_dir)
                if os.path.isdir(dirpath := os.path.join(load_dir, subdir))
            ]
        )
        load_dir = load_dirs[-1]
    dframe_fpaths = sorted(glob(os.path.join(load_dir, "results*csv")))
    npz_fpaths = sorted(glob(os.path.join(load_dir, "objects_*npz")))
    npz2_fpaths = sorted(glob(os.path.join(load_dir, "objects2_*npz")))

    def _parse_name(fpath):
        fname = os.path.basename(fpath)
        fname, ext = os.path.splitext(fname)
        fparts = fname.split("_")
        tstamp = fparts[1]
        return tstamp

    tstamps = [_parse_name(dframe_fpath) for dframe_fpath in dframe_fpaths]

    def _load_dframe_with_parms(fpath, tstamp):
        dframe = pd.read_csv(fpath, index_col=0)
        dframe["tstamp"] = tstamp
        return dframe

    def _load_npz(fpath):
        fpath = npz_fpaths[0]
        obj = np.load(fpath)
        x_bars, y_bars, z_bars = obj.values()
        return x_bars, y_bars, z_bars

    dframes = [
        _load_dframe_with_parms(fpath, parms)
        for fpath, parms in zip(dframe_fpaths, tstamps)
    ]
    dframe = pd.concat(dframes, axis=0)
    solutions = [_load_npz(fpath) for fpath in npz_fpaths]
    initial_data = [_load_npz(fpath) for fpath in npz2_fpaths]

    return dframe, solutions, initial_data


if __name__ == "__main__":

    run_big_data_experiment = True

    m, n, s = 100, 200, 2
    gammas = np.logspace(-2, 1, 7, endpoint=True)
    alphas = np.logspace(-1, 1, 31, endpoint=True)
    gamma_subset = gammas[::2]

    # generate lots of data if flag is True
    if run_big_data_experiment:
        big_data_experiment(m, n, s, gammas=gammas, alphas=alphas)

    # load the data that was most recently generated
    dframe, solutions, initial_data = load_big_data_experiment(load_dir=None)

    # create the plot
    plot_empirical_inexact_uniqueness_sufficiency(
        dframe,
        gamma_subset=gamma_subset,
        savefig="empirical_inexact_uniqueness_sufficiency_heatmap_with_error.pdf",
    )
