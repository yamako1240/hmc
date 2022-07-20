"""Hamiltonian monte carlo."""

import numpy as np
from numpy import linalg as LA
import dataclasses
import tqdm

DeltaMax = 1000


@dataclasses.dataclass(frozen=True)
class Fvals:
    """Dataclass for storing function values `(fun, grad)`."""

    y: any
    x: any
    d: any

    @classmethod
    def from_fun(cls, fun, x_):
        """Rturn the function values."""
        y, d = fun(x_)
        return cls(y, x_, d)


def mc(fun, x0, method=None, options={}):
    """
    Execute the mc simulation.

    Paramters
    ---------
    fun : callable
        Objective function. The function returns a tuple of a value and a gradient vector.
    x0 : ndarray
        Initial value.
    method : str
        Calculation method.
    options : list, optional
        Options passed to MC subroutines.

    """
    x0 = np.asarray(x0)

    if method is None:
        method = "hmc"

    if method == "hmc":
        return _mc_hmc(fun, x0, **options)
    elif method == "enuts":
        return _mc_enuts(fun, x0, **options)
    elif method == "nutsda":
        return _mc_nutsda(fun, x0, **options)
    else:
        raise ValueError("Unknown solver %s" % method)


def _leapfrog(fun, f, r, e):
    r_ = r + 0.5 * e * f.d
    f_ = Fvals.from_fun(fun, f.x + e * r_)
    r_ += 0.5 * e * f_.d
    return f_, r_


def _mc_hmc(fun, x0, e=0.1, L=10, M=1000, **unknown_options):
    """
    Naive MC.

    Parameters
    ----------
    e : float
    L : int
    M : int

    """
    res = np.empty((M, x0.shape[0]))
    res[0] = x0
    f0 = Fvals.from_fun(fun, x0)
    for i in tqdm.tqdm(range(1, M)):
        r0 = np.random.randn(x0.shape[0])
        res[i] = res[i - 1]
        f1, r1 = _leapfrog(fun, f0, r0, e)
        for _ in range(1, L):
            f1, r1 = _leapfrog(fun, f1, r1, e)
        if np.random.rand() < min(
            1.0, np.exp(f1.y - 0.5 * LA.norm(r1) - f0.y + 0.5 * LA.norm(r0))
        ):
            res[i] = f1.x
            f0 = f1
    return res


def _mc_enuts(fun, x0, e=1.0, M=1000, all_vals=False, **unknown_options):
    """
    Effective No-U-turn sampler.

    Parameters
    ----------
    e : float
    M : int
        Max iteration.

    """
    def buildtree(f, r, log_u, v, j, e, tree=None):
        if j == 0:
            ft, rt = _leapfrog(fun, f, r, v * e)
            nt = 1 if log_u <= ft.y - 0.5 * np.dot(rt, rt) else 0
            st = 1 if ft.y - 0.5 * np.dot(rt, rt) > log_u - DeltaMax else 0
            if tree is not None and nt == 1:
                tree.append(ft.x)
            return ft, rt, ft, rt, ft, nt, st
        else:
            fi, ri, ff, rf, ft, nt, st = buildtree(f, r, log_u, v, j - 1, e, tree)
            if st == 1:
                if v == -1:
                    fi, ri, _, _, ftt, ntt, stt = buildtree(
                        fi, ri, log_u, v, j - 1, e, tree
                    )
                else:
                    _, _, ff, rf, ftt, ntt, stt = buildtree(
                        ff, rf, log_u, v, j - 1, e, tree
                    )
                if nt + ntt != 0 and np.random.rand() < ntt / (nt + ntt):
                    ft = ftt
                st = (
                    stt
                    * (np.dot(ff.x - fi.x, ri) >= 0)
                    * (np.dot(ff.x - fi.x, rf) >= 0)
                )
                nt = nt + ntt
            return fi, ri, ff, rf, ft, nt, st

    res = np.empty((M, x0.shape[0]))
    if all_vals:
        trees = [[x0]]
    res[0] = x0
    f0 = Fvals.from_fun(fun, x0)
    for m in tqdm.tqdm(range(1, M)):
        if all_vals:
            trees.append([res[m - 1]])
            tree = trees[m]
        else:
            tree = None
        r0 = np.random.randn(x0.shape[0])
        log_u = np.log(np.random.rand()) + f0.y - 0.5 * np.dot(r0, r0)
        fi = ff = f0
        ri = rf = r0
        j = 0
        res[m] = res[m - 1]
        n = 1
        s = 1
        while s == 1:
            v = np.random.randint(2) * 2 - 1
            if v == -1:
                fi, ri, _, _, ft, nt, st = buildtree(fi, ri, log_u, v, j, e, tree)
            else:
                _, _, ff, rf, ft, nt, st = buildtree(ff, rf, log_u, v, j, e, tree)
            if st == 1:
                if np.random.rand() < nt / (n + nt):
                    res[m] = ft.x
                    f0 = ft
            n += nt
            s = st * (np.dot(ff.x - fi.x, ri) >= 0) * (np.dot(ff.x - fi.x, rf) >= 0)
            j = j + 1
    if all_vals:
        trees = [np.array(tree) for tree in trees]
        return res, trees
    else:
        return res


def _find_reasonable_epsilon(fun, x0):
    e = 1.0
    r0 = np.random.randn(x0.shape[0])
    f0 = Fvals.from_fun(fun, x0)
    f1, r1 = _leapfrog(fun, f0, r0, e)
    a = (
        2 * (np.exp(f1.y - 0.5 * np.dot(r1, r1) - f0.y + 0.5 * np.dot(r0, r0)) > 0.5)
        - 1
    )
    while a * (f1.y - 0.5 * np.dot(r1, r1) - f0.y + 0.5 * np.dot(r0, r0)) > -a * np.log(
        2.0
    ):
        e *= np.power(2.0, a)
        f1, r1 = _leapfrog(fun, f0, r0, e)
    return e


def _mc_nutsda(
    fun,
    x0,
    delta=0.5,
    M=1000,
    Madapt=100,
    e0=None,
    mu=None,
    e0_=1.0,
    H0=0.0,
    gamma=0.05,
    t0=10,
    kappa=0.75,
    all_vals=False,
    **unknown_options
):
    """
    Effective No-U-turn sampler dual averaging.

    Parameters
    ----------
    e : float
    M : int
        Max iteration.
    Madapt : int
        Iterations for adaptation.
    e0 : float
    mu : float
    e0 : float
    H0 : float
    gamma : float
    t0 : float
    kappa : float
    all_vals : bool
        Return all values of each iteration.

    """
    def buildtree(f, r, log_u, v, j, e, f0, r0, tree=None):
        if j == 0:
            ft, rt = _leapfrog(fun, f, r, v * e)
            nt = 1 if log_u <= ft.y - 0.5 * np.dot(rt, rt) else 0
            st = 1 if ft.y - 0.5 * np.dot(rt, rt) > log_u - DeltaMax else 0
            if tree is not None and nt == 1:
                tree.append(ft.x)
            return (
                ft, rt, ft, rt, ft, nt, st, 
                min(
                    1.0,
                    np.exp(ft.y - 0.5 * LA.norm(rt) - f0.y + 0.5 * LA.norm(r0)),
                ),
                1,
            )
        else:
            fi, ri, ff, rf, ft, nt, st, at, nat = buildtree(
                f, r, log_u, v, j - 1, e, f0, r0, tree
            )
            if st == 1:
                if v == -1:
                    fi, ri, _, _, ftt, ntt, stt, att, natt = buildtree(
                        fi, ri, log_u, v, j - 1, e, f0, r0, tree
                    )
                else:
                    _, _, ff, rf, ftt, ntt, stt, att, natt = buildtree(
                        ff, rf, log_u, v, j - 1, e, f0, r0, tree
                    )
                if nt + ntt != 0 and np.random.rand() < ntt / (nt + ntt):
                    ft = ftt
                at += att
                nat += natt
                st = (
                    stt
                    * (np.dot(ff.x - fi.x, ri) >= 0)
                    * (np.dot(ff.x - fi.x, rf) >= 0)
                )
                nt += ntt
            return fi, ri, ff, rf, ft, nt, st, at, nat

    res = np.empty((M, x0.shape[0]))
    if all_vals:
        trees = [[x0]]
        dlt = np.zeros(M)
        es = np.zeros(M)
    res[0] = x0
    f0 = Fvals.from_fun(fun, x0)
    log_e0_ = np.log(e0_)
    if e0 is None:
        print("find reasonable epsilon")
        e0 = _find_reasonable_epsilon(fun, x0)
    if mu is None:
        mu = np.log(10 * e0)
    print("iteration", end="")
    for m in tqdm.tqdm(range(1, M)):
        if all_vals:
            trees.append([res[m - 1]])
            tree = trees[m]
        else:
            tree = None
        r0 = np.random.randn(x0.shape[0])
        log_u = np.log(np.random.rand()) + f0.y - 0.5 * np.dot(r0, r0)
        fi = ff = f0
        ri = rf = r0
        j = 0
        res[m] = res[m - 1]
        n = 1
        s = 1
        while s == 1:
            v = np.random.randint(2) * 2 - 1
            if v == -1:
                fi, ri, _, _, ft, nt, st, a, na = buildtree(
                    fi, ri, log_u, v, j, e0, f0, r0, tree
                )
            else:
                _, _, ff, rf, ft, nt, st, a, na = buildtree(
                    ff, rf, log_u, v, j, e0, f0, r0, tree
                )
            if st == 1:
                if np.random.rand() < nt / (n + nt):
                    res[m] = ft.x
                    f0 = ft
            n = n + nt
            s = st * (np.dot(ff.x - fi.x, ri) >= 0) * (np.dot(ff.x - fi.x, rf) >= 0)
            j = j + 1
        if m <= Madapt:
            H0 = (1 - 1 / (m + t0)) * H0 + 1 / (m + t0) * (delta - a / na)
            log_e0 = mu - np.sqrt(m) / gamma * H0
            log_e0_ = np.power(m, -kappa) * log_e0 + (1 - np.power(m, -kappa)) * log_e0_
            e0 = np.exp(log_e0)
        else:
            e0 = np.exp(log_e0_)
        if all_vals:
            dlt[m] = a / na
            es[m] = e0
    if all_vals:
        trees = [np.array(tree) for tree in trees]
        return res, (trees, dlt, es)
    else:
        return res
