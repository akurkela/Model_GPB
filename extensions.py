import warnings
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.special import lambertw
from scipy.stats import triang
from pQCD import pQCD


n_end_diffusion = 40 * 0.16
n_start_pQCD = 30 * 0.16
hierarchical_min, hierarchical_max = 0.2, 0.4
n_layers = 6
grid_size_diffused = 500



def extension_generation(mu, n, p, n_eoss_requested):
    e = -p + n * mu
    x_list = np.linspace(1, 4)
    iqcd = np.zeros(len(x_list))

    for j, x in enumerate(x_list):
        temp_mu = np.linspace(2.2, 2.8)
        temp_n = pQCD(x).number_density(temp_mu)
        iqcd[j] = tension_id(e, p, n, muQCD=np.interp(n_start_pQCD, temp_n, temp_mu), X=x)

    inside = (0 <= iqcd) & (iqcd <= 1)

    if not inside.any():
        raise ValueError("There are no valid extensions for the given point to pQCD at 30ns.")

    d = np.diff(inside.astype(int))
    enter = np.flatnonzero(d == 1)[0] + 1 if (d == 1).any() else 0
    exit_ = np.flatnonzero(d == -1)[0] + 1 if (d == -1).any() else len(iqcd) - 1

    print(f"Point is connectible for X in range [{x_list[enter]:.3f},{x_list[exit_]:.3f}]")

    df = pd.DataFrame([_generate_fractal_eos(mu, n, p) for _ in range(n_eoss_requested)])
    df["correlation"] = np.random.uniform(hierarchical_min, hierarchical_max, size=n_eoss_requested)
    _diffuse_fractal(df, n)
    return df

def tension_id(e0, p0, n0, muQCD=2.4, X=2):
    qcd = pQCD(X)

    p_qcd = qcd.pressure(muQCD) / 1000
    n_qcd = qcd.number_density(muQCD)

    mu0 = (e0 + p0) / n0
    deltaP = p_qcd - p0

    pMin = 0.5 * (muQCD * (muQCD / mu0) - mu0) * n0
    pMax = 0.5 * (muQCD - mu0 * (mu0 / muQCD)) * n_qcd

    return (deltaP - pMin) / (pMax - pMin)


class _PQCDConstraints:
    def __init__(self, cet, qcd, cs2):
        self.muCET, self.nCET, self.pCET = cet
        self.muQCD, self.nQCD, self.pQCD = qcd
        self.cs2 = cs2
        self.muc = ((self.muQCD ** (1 / cs2) * (-self.pCET + self.pQCD + cs2 * (self.muCET * self.nCET - self.muQCD * self.nQCD - self.pCET + self.pQCD))) / (cs2 * ((self.muQCD / self.muCET) ** (1 / cs2) * self.nCET - self.nQCD))) ** (cs2 / (1 + cs2))

    def nmin(self, mu):
        if mu < self.muc:
            return (mu / self.muCET) ** (1 / self.cs2) * self.nCET
        return ((mu / self.muCET) ** (1 / self.cs2) * (-self.pCET + self.pQCD + self.cs2 * (mu * (mu / self.muQCD) ** (1 / self.cs2) * self.nQCD - self.muQCD * self.nQCD - self.pCET + self.pQCD))) / (self.cs2 * (mu * (mu / self.muCET) ** (1 / self.cs2) - self.muCET))

    def nmax(self, mu):
        if mu < self.muc:
            return ((mu / self.muQCD) ** (1 / self.cs2) * (self.pCET - self.pQCD + self.cs2 * (mu * (mu / self.muCET) ** (1 / self.cs2) * self.nCET - self.muCET * self.nCET + self.pCET - self.pQCD))) / (self.cs2 * (mu * (mu / self.muQCD) ** (1 / self.cs2) - self.muQCD))
        return (mu / self.muQCD) ** (1 / self.cs2) * self.nQCD

    def pmin(self, mu):
        return self.pCET + (self.cs2 / (1 + self.cs2)) * (mu - self.muCET * (self.muCET / mu) ** (1 / self.cs2)) * self.nmin(mu)

    def pmax(self, mu, n):
        nc = (mu / self.muCET) ** (1 / self.cs2) * self.nmax(self.muCET)
        if n < nc:
            return self.pCET + (self.cs2 / (1 + self.cs2)) * (mu - self.muCET * (self.muCET / mu) ** (1 / self.cs2)) * n
        return self.pQCD + (self.cs2 / (1 + self.cs2)) * (mu - self.muQCD * (self.muQCD / mu) ** (1 / self.cs2)) * n


def _add_point(P0, P1, cs2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            muc = _mu_c(P0, P1)
            constraints = _PQCDConstraints(P0, P1, cs2)
            mu_new = np.sqrt(_select_mu(P0[0] ** 2, muc ** 2, P1[0] ** 2))
            n_min, n_max = constraints.nmin(mu_new), constraints.nmax(mu_new)
            n_c = mu_new * constraints.nmax(P0[0]) / P0[0]
            p_min = constraints.pmin(mu_new)
            p_max_nc = constraints.pmax(mu_new, n_c)
            n_new = _sample_triangle(n_min, n_max, n_c, p_max_nc)[0]
            p_new = np.random.uniform(p_min, constraints.pmax(mu_new, n_new))
            return [mu_new, n_new, p_new]
        except (OverflowError, ValueError, FloatingPointError):
            return P0


def _mu_c(P0, P1):
    mu0, n0, p0 = P0
    mu1, n1, p1 = P1
    return ((mu1 * (mu0 * n0 - mu1 * n1 - 2 * p0 + 2 * p1)) / (((mu1 / mu0) * n0 - n1))) ** 0.5


def _select_mu(mu_l2, mu_c2, mu_h2):
    if _side_select(mu_l2, mu_c2, mu_h2):
        x = np.random.uniform(0, _area_integral(mu_c2, mu_l2, mu_c2, mu_h2))
        return mu_h2 + (mu_h2 - mu_l2) * lambertw(-np.exp(-(x * (mu_c2 - mu_l2) / (mu_h2 - mu_c2) + mu_h2 - mu_l2) / (mu_h2 - mu_l2))).real
    x = np.random.uniform(0, _area_integral(mu_h2, mu_l2, mu_c2, mu_h2))
    return mu_l2 - (mu_h2 - mu_l2) * lambertw((mu_l2 - mu_c2) / (mu_h2 - mu_l2) * np.exp((x * (mu_h2 - mu_c2) / (mu_c2 - mu_l2) + mu_l2 - mu_c2) / (mu_h2 - mu_l2))).real


def _side_select(mu_l2, mu_c2, mu_h2):
    left = _area_integral(mu_c2, mu_l2, mu_c2, mu_h2)
    right = _area_integral(mu_h2, mu_l2, mu_c2, mu_h2)
    return np.random.uniform() <= left / (left + right)


def _area_integral(mu2, mu_l2, mu_c2, mu_h2):
    if mu2 <= mu_c2:
        return (-mu2 + mu_l2 - (mu_h2 - mu_l2) * np.log((mu_h2 - mu2) / (mu_h2 - mu_l2))) * (mu_h2 - mu_c2) / (mu_c2 - mu_l2)
    return (-mu2 + mu_c2 - (mu_l2 - mu_h2) * np.log((mu2 - mu_l2) / (mu_c2 - mu_l2))) * (mu_c2 - mu_l2) / (mu_h2 - mu_c2)


def _sample_triangle(x0, x1, xc, yc=None, size=1):
    return triang.rvs((xc - x0) / (x1 - x0), loc=x0, scale=x1 - x0, size=size)


def _fractal_area_dist(P0, P1, layers=n_layers, cs2=1):
    points = [P0, P1]
    for _ in range(layers):
        points = _make_layer(points, cs2)
    return points


def _make_layer(points, cs2=1):
    out = []
    for i in range(len(points) - 1):
        out.append(points[i])
        out.append(_add_point(points[i], points[i + 1], cs2))
    out.append(points[-1])
    return np.array(out)


def _generate_fractal_eos(mu, n, p):
    while True:
        x = np.exp(np.random.uniform(np.log(1), np.log(4)))
        mu_grid = np.linspace(2.0, 3.5, 500)
        qcd = pQCD(x)
        n_grid = qcd.number_density(mu_grid)

        iqcd = tension_id(
            -p + mu * n,
            p,
            n,
            muQCD=np.interp(n_start_pQCD, n_grid, mu_grid),
            X=x
        )

        if 0 <= iqcd <= 1:
            break

    p_grid = qcd.pressure(mu_grid) / 1000
    e_grid = qcd.edens(mu_grid) / 1000
    valid_idx = np.where(n_grid >= n_start_pQCD)[0]

    pqcd_mu = mu_grid[valid_idx]
    pqcd_n = n_grid[valid_idx]
    pqcd_p = p_grid[valid_idx]
    pqcd_e = e_grid[valid_idx]

    eos = _fractal_area_dist(
        [mu, n, p],
        [pqcd_mu[0], pqcd_n[0], pqcd_p[0]],
        layers=n_layers,
        cs2=1
    )

    mu_koku = eos[:, 0]
    n_koku = eos[:, 1]
    p_koku = eos[:, 2]
    e_koku = mu_koku * n_koku - p_koku

    mu_all = np.concatenate([mu_koku, pqcd_mu[1:]])
    n_all = np.concatenate([n_koku, pqcd_n[1:]])
    p_all = np.concatenate([p_koku, pqcd_p[1:]])
    e_all = np.concatenate([e_koku, pqcd_e[1:]])

    with np.errstate(divide="ignore", invalid="ignore"):
        cs2 = np.log(mu_all[:-1] / mu_all[1:]) / np.log(n_all[:-1] / n_all[1:])

    cs2 = np.append(cs2, cs2[-1])

    return {
        "mu_koku": mu_all,
        "n_koku": n_all,
        "p_koku": p_all,
        "e_koku": e_all,
        "cs2_koku": cs2,
        "pQCD_X": x,
    }


def _smooth_df(df, n_start_diffusion):
    x_grid = np.linspace(np.log(n_start_diffusion), np.log(n_end_diffusion), grid_size_diffused)
    n_grid = np.exp(x_grid)
    dx = x_grid[1] - x_grid[0]

    T = np.zeros((grid_size_diffused, grid_size_diffused))
    np.fill_diagonal(T, -2)
    T += np.diag(np.ones(grid_size_diffused - 1), 1) + np.diag(np.ones(grid_size_diffused - 1), -1)
    T /= dx ** 2
    T[0, :] = 0
    T[-1, :] = 0

    F = np.diag(np.ones(grid_size_diffused)) + np.diag(-np.ones(grid_size_diffused - 1), -1)
    F /= dx
    F[0, :] = 0
    F[-1, :] = 0

    evals, psi_right = np.linalg.eig(T + F)
    evals = np.real_if_close(evals)
    psi_left = np.linalg.inv(psi_right).T
    idx = np.argsort(evals)
    evals, psi_right, psi_left = evals[idx], psi_right[:, idx], psi_left[:, idx]

    mu_list, n_list = [], []
    for i in range(len(df)):
        u0 = np.interp(x_grid, np.log(df.n_koku[i]), df.mu_koku[i])
        c_t = (psi_left.T @ u0) * np.exp(evals * df.correlation[i] ** 2 / 4)
        idx1 = np.where(df.n_koku[i] < n_start_diffusion)[0]
        idx2 = np.where(df.n_koku[i] > n_end_diffusion)[0]
        mu_list.append(np.concatenate((df.mu_koku[i][idx1], np.real(psi_right @ c_t), df.mu_koku[i][idx2])))
        n_list.append(np.concatenate((df.n_koku[i][idx1], n_grid, df.n_koku[i][idx2])))

    return mu_list, n_list


def _get_e_p_cs2(df):
    e_list, p_list, cs2_list = [], [], []
    for i in range(len(df)):
        e = df.e_koku[i][0] + integrate.cumulative_trapezoid(df.mu[i], df.n[i], initial=0)
        p = -e + df.mu[i] * df.n[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            cs2 = np.log(df.mu[i][:-1] / df.mu[i][1:]) / np.log(df.n[i][:-1] / df.n[i][1:])
        e_list.append(e)
        p_list.append(p)
        cs2_list.append(np.append(cs2, cs2[-1]))
    return e_list, p_list, cs2_list


def _diffuse_fractal(df, n_start_diffusion):
    df["mu"], df["n"] = _smooth_df(df, n_start_diffusion)
    df["e"], df["p"], df["cs2"] = _get_e_p_cs2(df)

