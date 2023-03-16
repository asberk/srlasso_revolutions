"""
srlasso_inexact_empirical_uniqueness_sufficiency_crosssection

Creates Figure 2 in the paper.

Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 16 March 2023
"""
import numpy as np
import numpy.random as npr

from solve import srLASSO_uniqueness
from solve_utils import generate_data, lamda_sr_best
from viz import plot_uniqueness_sufficiency_crosssection

seed = 2023
npr.seed(seed)

m, n, s, gamma = 100, 200, 2, 0.1


if __name__ == "__main__":
    A, b, x0 = generate_data(m, n, s, gamma)
    lamda = lamda_sr_best(m) / m**0.5
    lamdas = np.geomspace(1e-2, 1.0, 201)
    (
        results,
        x_bars,
        y_bars,
        z_bars,
    ) = srLASSO_uniqueness(A, b, 1.0, x0, alphas=lamdas)

    max_primal_dual_difference = (
        ((results.primal_value - results.dual_value) / results.primal_value)
        .abs()
        .max()
    )
    print(max_primal_dual_difference)

    plot_uniqueness_sufficiency_crosssection(
        results,
        xlim=(0.05, 1.0),
        ylim=(0.05, 200),
        savefig=f"uniqueness_sufficiency_crosssection_m{m}_n{n}_s{s}_gamma1e-1.pdf",
    )
