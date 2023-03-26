"""academic_example

Example 3.11 in SR LASSO paper. This code is not robust for other examples. It
was built to work for the examples given in the paper.

Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 26 March 2023
"""
import numpy as np
import numpy.linalg as la


def in_l1_subdifferential(v, soften=True):
    strict = np.abs(v) <= 1
    if not soften:
        return strict.all()
    soft = np.isclose(np.abs(v), 1.0)
    return np.all(strict | soft)


def support(v):
    mask = np.isclose(v, 0)
    supp = ~mask
    return supp.ravel()


def optimality_conditions(x_bar, A, b, lamda):
    r = b - A.dot(x_bar)
    R = la.norm(r)
    q = A.T.dot(r / R)
    return in_l1_subdifferential(q / lamda)


def equicorrelation(x_bar, A, b, lamda):
    r = b - A.dot(x_bar)
    R = la.norm(r)
    q = A.T.dot(r)
    return np.isclose(q, lamda * R).ravel()


def weak_assumption(x_bar, A, b, lamda, z):
    if np.allclose(A.dot(x_bar), b):
        print("A.x_bar = b")
        return False
    supp = support(x_bar)
    beta, _, rank, _ = la.lstsq(A[:, supp], b, rcond=None)
    if rank < supp.sum():
        print("deficient rank")
        return False
    r = b - A.dot(x_bar)
    R = la.norm(r)
    if not np.isclose(np.dot(r.ravel() / R, z.ravel()), 0):
        raise ValueError("invalid z (1)")
    if (supp.sum() > 0) and not np.allclose(A[:, supp].T.dot(z), 0):
        raise ValueError("invalid z (2)")
    zeta = A[:, ~supp].T.dot(r / R + z)
    Zeta = la.norm(zeta, np.inf)
    if Zeta >= lamda:
        print(f"invalid z: {zeta}, {Zeta}")
        return False
    return True


def intermediate_assumption(x_bar, A, b, lamda):
    if np.allclose(A.dot(x_bar), b):
        print("A.x_bar = b")
        return False
    J = equicorrelation(x_bar, A, b, lamda)
    beta, _, rank, _ = la.lstsq(A[:, J], b, rcond=None)
    if rank < J.sum():
        print("deficient rank ==>")
        return False
    if np.allclose(A[:, J].dot(beta), b):
        print("b in rge A_J")
        return False
    return True


def strong_assumption(x_bar, A, b, lamda):
    if np.allclose(A.dot(x_bar), b):
        print("A.x_bar = b")
        return False
    supp = support(x_bar)
    J = equicorrelation(x_bar, A, b, lamda)
    if (supp ^ J).sum() > 0:
        print("I != J ==>")
        return False
    beta, _, rank, _ = la.lstsq(A[:, supp], b, rcond=None)
    if rank < supp.sum():
        print("deficient rank")
        return False
    if np.allclose(A[:, supp].dot(beta), b):
        print("b in rge A_J")
        return False
    r = b - A.dot(x_bar)
    R = la.norm(r)
    q = la.norm(A[:, ~supp].T.dot(r), np.inf)
    if np.isclose(q, lamda * R) or (q >= lamda * R):
        print("I != J")
        return False
    return True


# Example (a)
print("Example (a)\n" + "-" * 11)
A = np.array([[1, 0, 0], [0, 1, 1]])
b = np.array([[1, 2]]).T
lamda = 2 / np.sqrt(5)
x_bar = np.array([[0, 0, 0]]).T
z = lamda / 6 * np.array([[2, -1]]).T

print("optimal:", optimality_conditions(x_bar, A, b, lamda))
print("weak:", weak_assumption(x_bar, A, b, lamda, z))
print("intermediate:", intermediate_assumption(x_bar, A, b, lamda))


# Example (b)
print("\nExample (b)\n" + "-" * 11)
A = np.array([[1, 0, 2], [0, 2, -2]])
b = np.array([[1, 1]]).T
lamda = 2**0.5
x_bar = np.array([[0, 0, 0]]).T
J = equicorrelation(x_bar, A, b, lamda)
print("optimal:", optimality_conditions(x_bar, A, b, lamda))
print("intermediate:", intermediate_assumption(x_bar, A, b, lamda))
print("strong:", strong_assumption(x_bar, A, b, lamda))
