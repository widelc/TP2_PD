import os
import sys
from typing import Tuple, List, Union, Any
import pandas as pd
import math
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np


if os.getcwd().startswith("/Users/christian/"):
    sys.path.append("../..")
from jupyter_notebook import *
import black_merton_scholes as bms
from monte_carlo import antithetic_normal


def load_zero_curve():
    """Load rates. Makes sure that within a date, rates are sorted by maturity."""
    rf = pd.read_csv("zerocd.csv", index_col=0).reset_index(drop=True)
    rf["date"] = pd.to_datetime(rf["date"])
    return rf.sort_values(by=["date", "days"])


def load_dividend():
    y = pd.read_csv("distrd_108105.csv", index_col=0).reset_index(drop=True)
    y["date"] = pd.to_datetime(y["date"])
    return y.sort_values(by=["date"])[["date", "rate"]]


def load_price():
    p = pd.read_csv("secprd_108105.csv", index_col=0).reset_index(drop=True)
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values(by=["date"])[["date", "close"]]


def read_and_concatenate_options(file1: str, file2: str) -> pd.DataFrame:
    """
    Lit et concatène deux fichiers CSV d'options dans un seul DataFrame.

    Parameters
    ----------
    file1 : str
        Le chemin du premier fichier CSV d'options.
    file2 : str
        Le chemin du deuxième fichier CSV d'options.

    Returns
    -------
    pd.DataFrame
        Un DataFrame contenant les données des deux fichiers CSV d'options concaténés.
    """

    # Lit les fichiers CSV d'options
    options1 = pd.read_csv(file1, index_col=0).reset_index(drop=True)
    options2 = pd.read_csv(file2, index_col=0).reset_index(drop=True)

    # Concatène les deux DataFrames d'options
    option_info = pd.concat([options1, options2])

    return option_info


def get_dividend_rate(time_t):
    y = load_dividend()

    date_ref = min([min(y.date), min(time_t)])
    interp = interp1d(
        (y.date - date_ref).apply(lambda x: x.days), y.rate / 100, kind="linear"
    )

    return interp((time_t - date_ref).apply(lambda x: x.days))


def get_price(time_t):
    p = load_price()
    date_ref = min([min(p.date), min(time_t)])
    interp = interp1d(
        (p.date - date_ref).apply(lambda x: x.days), p.close, kind="linear"
    )

    return interp((time_t - date_ref).apply(lambda x: x.days))


def get_risk_free_rate(time_t, dtm):
    rf = load_zero_curve()
    rf = rf[rf.date == time_t]

    # Interpolate between the 2 closest maturities
    interp = interp1d(rf.days, rf.rate / 100, kind="linear")
    return interp(dtm)


def get_log_excess_returns(days_in_year):
    # First, read the data on the underlying (108105 is the secid of the SP500 in OptionMetrics)
    spx = pd.read_csv("secprd_108105.csv", index_col=0).reset_index(drop=True)
    spx["date"] = pd.to_datetime(spx["date"])  # convert str to actual dates
    assert len(spx.cfadj.unique()) == 1  # Make sure there are no splits unaccounted for
    spx["log_ret"] = np.log(spx["close"] / spx["close"].shift(1))  # compute log-returns

    # Then, load rates. The function makes sure that within a date, rates are sorted by maturity
    rf = load_zero_curve()

    # Then, select the shortest-maturity rate on each day
    #  ffill(): when a date in the resampling is not in rf, use the last observed value
    short_term = rf.resample("D", on="date").agg(["first"]).ffill()
    short_term.columns = short_term.columns.get_level_values(
        0
    )  # get rid of 'first' in column names
    short_term.date = short_term.index  # Get rid of the ffill'ed value on date
    short_term = short_term.reset_index(
        drop=True
    )  # avoids conflicts in the merge below

    # And re-express it in daily log-returns
    #  Formally, we could account for 1/365 day of interest rate between 2 weekdays,
    #  and 3/365 between the Friday close and the Monday close... We'll keep things simple here.
    short_term["rf"] = short_term["rate"] / 100 / days_in_year

    # Finally, get *excess* log-returns (log_retx)
    spx = spx.merge(short_term, left_on="date", right_on="date", how="inner")
    spx["log_xret"] = spx["log_ret"] - spx["rf"].shift(
        1
    )  # rf is known at the beginning of the (1-day) period
    assert np.abs(spx.log_ret.iloc[1] - spx.rf.iloc[0] - spx.log_xret.iloc[1]) < 1e-16
    spx = spx.set_index("date")
    return spx["log_xret"]


# Placeholder for simulation results under different measures
class measure(struct):
    def __init__(self, ex_r: np.array, h: np.array, z: np.array = None):
        super().__init__(ex_r=ex_r, h=h, z=z)


# Subclassing struct allows for easy dict-like construction, defines str and repr.
class model(struct):
    pass


class ngarch(model):
    @classmethod
    def initialize_at(cls, time_t, log_xreturns, days_in_year):
        log_xret = log_xreturns[log_xreturns.index <= time_t]

        ng = (
            cls()
        )  # calls the constructor of the class (i.e. ngarch(), but robust to inheritance...)
        ng.lmbda = 0.01049  # lambda is a reserved word in Python   ##\simeq np.log(1.06) / (ng.days_in_year*np.sqrt(ng.uncond_var()))
        ng.omega = np.nan
        ng.alpha = 6.2530e-2
        ng.beta = 0.90825
        ng.gamma = 0.5972

        ng.days_in_year = days_in_year
        Dt = 1 / days_in_year

        ng.omega = ng.variance_targeting(log_xret.var())
        # print(ng)
        # print('Persistence:', ng.persistence())
        # print('Unconditional volatility:', np.sqrt(ng.uncond_var()/Dt))

        ng.log_xret = log_xret
        return ng

    def variance_targeting(self, var_target):
        omega = (1 - self.persistence()) * var_target
        return omega

    def persistence(self):
        return self.alpha * (1 + self.gamma**2) + self.beta

    def uncond_var(self):
        return self.omega / (1 - self.persistence())

    def cond_var(self):
        h2 = self.P_predict_h() ** 2
        return 2 * (self.alpha**2) * h2 * (1 + 2 * (self.gamma**2))

    def corr_ret_var(self):
        ht = self.P_predict_h()
        return -2 * self.gamma / np.sqrt(2 + 4 * self.gamma**2)

    def P_predict_h(self):
        theta = [self.lmbda, self.alpha, self.beta, self.gamma]
        h_t, eps = f_ht_ngarch(theta, self)
        return (
            self.omega
            + self.alpha * h_t[-1] * ((eps[-1] - self.gamma) ** 2)
            + self.beta * h_t[-1]
        )

    def Q_predict_h(self):
        theta = [self.lmbda, self.alpha, self.beta, self.gamma]
        h_t, eps = f_ht_ngarch(theta, self)
        return (
            self.omega
            + self.alpha * h_t[-1] * ((eps[-1] - self.gamma - self.lmbda) ** 2)
            + self.beta * h_t[-1]
        )

    def simulateP(self, S_t0, n_days, n_paths, h_tp1, z=None):
        """Simulate excess returns and their variance under the P measure

        We consider that the simulation is starting at t0, and tp0 is a shorthand for "time
        t0+1" where p in tp1 stands for plus."

        This method simulates *excess* log-returns; the risk-free rate must be added outside
        this function to get the full log-return. This allows using different risk-free rates
        to price options at different horizons with the same core simulations.

        Args:
            S_t0:    Spot price at the beginning of the simulation
            n_days:  Length of the simulation
            n_paths: Number of paths in the simulation
            h_tp1:   Measurable at t0. Note that
            z:       The N(0,1) shocks for the simulation (optional)

        Returns:
            ex_r:    Excess log-returns of the underlying (np.array: n_days x n_paths)
            h:       Corresponding variance (np.array: n_days+1 x n_paths)
            z:       The shocks used in the simulation
        """
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days + 1, n_paths), np.nan)
        h[0, :] = h_tp1  # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)

        # Going from 1 day to n_days ahead
        #  t0+1      is tn==0 because indices start at 0 in Python
        #  t0+n_days is consequently tn==n_days-1
        for tn in range(0, n_days):
            # Simulate returns
            sig = np.sqrt(h[tn, :])
            ex_r[tn, :] = self.lmbda * sig - 0.5 * h[tn, :] + sig * z[tn, :]

            # Update the variance paths
            h[tn + 1, :] = (
                self.omega
                + self.alpha * h[tn, :] * (z[tn, :] - self.gamma) ** 2
                + self.beta * h[tn, :]
            )

        return ex_r, h, z

    def simulateQ(self, S_t0, n_days, n_paths, h_tp1, z=None):
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days + 1, n_paths), np.nan)
        h[0, :] = h_tp1

        if z is None:
            z = antithetic_normal(n_days, n_paths)

        for tn in range(0, n_days):
            # Simulate returns
            sig = np.sqrt(h[tn, :])
            ex_r[tn, :] = -0.5 * h[tn, :] + sig * z[tn, :]

            # Update the variance paths
            h[tn + 1, :] = (
                self.omega
                + self.alpha * h[tn, :] * (z[tn, :] - self.gamma - self.lmbda) ** 2
                + self.beta * h[tn, :]
            )

        return ex_r, h, z

    def option_price(self, cum_ex_r, F_t0_T, K, rf, dtm, is_call):
        cp = [1 if cond else -1 for cond in [is_call]][0]
        disc = np.exp(-rf * dtm / self.days_in_year)
        payoff = np.maximum(0, ((F_t0_T * cum_ex_r) - K) * cp)
        option_price = np.mean(disc * payoff)

        return option_price


def plot_excess_return_forecasts(horizon, P, Q, annualized=False):
    ann = [1.0]
    y_prefix = ""
    if annualized:
        ann = horizon
        y_prefix = "Annualized "

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = plt.subplot(2, 1, 1)
    ax.plot(horizon, P.expected_ex_r / ann[0], label="P ex. returns forecasts (sim)")
    ax.plot(horizon, Q.expected_ex_r / ann[0], label="Q ex. returns forecasts (sim)")
    ax.legend(loc="upper right")
    ax.set_ylabel(y_prefix + "Excess Returns")

    ax = plt.subplot(2, 1, 2)
    # ERP is accumulated between t0 and horizon h (think about a buy and hold position)
    erp = (
        np.cumsum(P.expected_ex_r - Q.expected_ex_r) / ann
    )  # Theoretically, the "- Q.expected_ex_r" part is useless here
    ax.plot(horizon, erp)
    ax.set_ylabel(y_prefix + "Equity Risk Premium")
    ax.set_xlabel("Years to Maturity")

    return axes


def plot_var_forecasts2(
    horizon: Union[int, List[int]],
    P: List[ngarch],
    Q: List[ngarch],
    annualized: bool = False,
) -> Tuple:
    """Plot variance forecasts and variance risk premium.

    Args:
        horizon: Maturities to compute forecasts.
        P: List of ngarch models of P series.
        Q: List of ngarch models of Q series.
        annualized: Whether to annualize the variance.

    Returns:
        Tuple containing the figure and axes.

    """
    title = ["1996-12-03", "2020-02-03"]
    ann = [1.0]
    y_prefix = ""
    if annualized:
        ann = horizon
        y_prefix = "Annualized "

    fig, axes = plt.subplots(2, len(P), figsize=(14, 10))
    j = 0
    for i in range(len(P)):
        Pi = P[i]
        Qi = Q[i]

        # ax = plt.subplot(2,2, j)
        axes[j, i].plot(
            horizon, np.mean(Pi.h, axis=1) / ann[0], label="P variance forecasts (sim)"
        )
        axes[j, i].plot(
            horizon, np.mean(Qi.h, axis=1) / ann[0], label="Q variance forecasts (sim)"
        )
        axes[j, i].legend(loc="best")
        axes[j, i].set_ylabel(y_prefix + "Variance")
        axes[j, i].set_title(title[i])

        # ax = plt.subplot(2,2, j)
        j += 1
        vrp = np.cumsum(np.mean(Pi.h, axis=1) - np.mean(Qi.h, axis=1)) / ann
        axes[j, i].plot(horizon, vrp)
        axes[j, i].set_ylabel(y_prefix + "Variance Risk Premium")
        axes[j, i].set_xlabel("Years to Maturity")
        j -= 1

    return fig, axes


def plot_var_forecasts(
    horizon: List[int], P: List[ngarch], Q: List[ngarch], annualized: bool = False
) -> Tuple:
    """Plot variance forecasts for P and Q.

    Args:
        horizon: List of maturities to compute forecasts.
        P: List of ngarch models for P.
        Q: List of ngarch models for Q.
        annualized: Whether to annualize the variance.

    Returns:
        Tuple containing the figure and axes.

    """
    title = ["1996-12-03", "2020-02-03"]
    fig, axes = plt.subplots(len(P), 1, figsize=(14, 10))
    for i in range(len(P)):
        p = P[i]
        q = Q[i]

        p.expected_h = np.mean(p.h, axis=1)
        q.expected_h = np.mean(q.h, axis=1)

        if annualized:
            days_in_year = 1 / horizon[0]
            p.expected_h = p.expected_h * days_in_year
            q.expected_h = q.expected_h * days_in_year
            y_prefix = "Annualized "

        ax = plt.subplot(2, 1, i + 1)
        ax.plot(horizon, p.expected_h, label="P Var forecasts (sim)")
        ax.plot(horizon, q.expected_h, label="Q Var forecasts (sim)")
        ax.legend(loc="upper right")
        ax.set_ylabel(y_prefix + "Variance")
        ax.set_title(title[i])

    ax.set_xlabel("Years to Maturity")

    return fig, axes


def plot_iv_surface3d(option_info: pd.DataFrame, date: str = "1996-12-03") -> None:
    """Plot a 3D implied volatility surface for a given date.

    Args:
        option_info (pd.DataFrame): A DataFrame containing the option information.
        date (str): The date for which to plot the surface. Default is '1996-12-03'.

    Returns:
        None.

    """
    df_date = option_info[option_info["date"] == date][
        ["K/S", "DTM", "impl_volatility"]
    ]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        df_date["K/S"],
        df_date["DTM"],
        df_date["impl_volatility"],
        cmap=plt.cm.jet,
        linewidth=0.2,
        edgecolor="gray",
        alpha=0.8,
    )

    ax.set_ylabel("Time to maturity")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_zlabel("Implied volatility")

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

    ax.view_init(elev=30, azim=-45)
    plt.title(f"Implied Volatility Surface ({date})", fontsize=16)
    plt.tight_layout()
    plt.show()

    return None


def f_ht_ngarch(theta: List[float], ng: ngarch) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conditional variances and innovations from the parameters of the ngarch model and the data from `ng`.

    Parameters
    ----------
    theta : List[float]
        Parameters of the ngarch model.
        theta[0] is lmbda.
        theta[1] is alpha.
        theta[2] is beta.
        theta[3] is gamma.
    ng : ngarch
        ngarch model containing the data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two NumPy arrays.
        The first array contains the conditional variances h_t.
        The second array contains the innovations eps.

    """
    # Extract data from ng
    lmbda = theta[0]
    omega = ng.omega
    alpha = theta[1]
    beta = theta[2]
    gamma = theta[3]

    # Get rid of the NA at the beggining
    log_xreturns = ng.log_xret[1:]

    # Initialize the conditional variances
    T = len(log_xreturns)
    h_t = np.full(T, np.nan)
    eps = np.full(T, np.nan)

    # Start with unconditional variances
    h_t_ini = omega / (1 - alpha * (1 + (gamma**2)) - beta)
    eps_ini = (log_xreturns[0] - lmbda * np.sqrt(h_t_ini) + 0.5 * h_t_ini) / np.sqrt(
        h_t_ini
    )

    h_t[0] = h_t_ini
    eps[0] = eps_ini

    # Compute conditional variance and innovations at each step
    for t in range(T - 1):
        h_t[t + 1] = omega + alpha * h_t[t] * ((eps[t] - gamma) ** 2) + beta * h_t[t]
        eps[t + 1] = (
            log_xreturns[t + 1] - lmbda * np.sqrt(h_t[t + 1]) + 0.5 * h_t[t + 1]
        ) / np.sqrt(h_t[t + 1])

    return h_t, eps


def f_nll_ngarch(theta: List[float], ng: ngarch) -> float:
    """
    Fonction de vraisemblance négative (negative log-likelihood) pour le modèle ngarch.

    Paramètres :
    -----------
    theta : list[float]
        Liste des paramètres.
    ng : ngarch
        Instance de la classe ngarch contenant les données et les paramètres.

    Renvoie :
    ---------
    nll : float
        Vraisemblance négative pour les paramètres donnés.
    """
    h, eps = f_ht_ngarch(theta, ng)
    nll = -0.5 * np.sum(np.log(2 * np.pi * h) + eps**2)

    return -nll


def f_ngarch(ng: ngarch) -> ngarch:
    """
    Fonction pour estimer les paramètres du modèle ngarch.

    Paramètres :
    -----------
    ng : ngarch
        Instance de la classe ngarch contenant les données et les paramètres initiaux.

    Renvoie :
    ---------
    ng : ngarch
        Instance de la classe ngarch contenant les données et les paramètres estimés.
    """
    # Initial guess
    theta_0 = [ng.lmbda, ng.alpha, ng.beta, ng.gamma]

    # Optimization (and ignore warnings)
    warnings.simplefilter("ignore")
    bounds = [(0, None), (0, 1), (0.5, 1), (0, None)]
    ineq_cons = {
        "type": "ineq",
        "fun": lambda x: np.array([1 - x[1] * (1 + (x[3] ** 2)) - x[2]]),
    }
    opt = minimize(
        fun=f_nll_ngarch,
        x0=theta_0,
        args=ng,
        method="SLSQP",
        bounds=bounds,
        constraints=ineq_cons,
        options={"maxiter": 5000},
    )
    warnings.resetwarnings()

    # Update parameters value in ng
    param = opt.x
    ng.lmbda = param[0]
    ng.alpha = param[1]
    ng.beta = param[2]
    ng.gamma = param[3]

    return ng


def f_out_format_Q1(ng_vec):
    # Output table
    days_in_year = ng_vec[0].days_in_year
    data = {
        "λ": ["{:.4e}".format(ng.lmbda) for ng in ng_vec],
        "ω": ["{:.4e}".format(ng.omega) for ng in ng_vec],
        "α": ["{:.4e}".format(ng.alpha) for ng in ng_vec],
        "β": ["{:.4e}".format(ng.beta) for ng in ng_vec],
        "γ": [ng.gamma for ng in ng_vec],
        "Vol. incond.": np.multiply(
            np.sqrt([ng.uncond_var() for ng in ng_vec]), np.sqrt(days_in_year)
        ),
        "Vol. cond. @ t+1": np.multiply(
            np.sqrt([ng.P_predict_h() for ng in ng_vec]), np.sqrt(days_in_year)
        ),
        "Corr. cond. @ t": [ng.corr_ret_var() for ng in ng_vec],
        "Vol. cond. @ t de h_t+2": [
            "{:.4e}".format(np.sqrt(ng.cond_var())) for ng in ng_vec
        ],
    }

    df = pd.DataFrame(data=data, index=["1996-12-31", "2020-02-01"])

    return df


def check_constraints_and_stationarity(df: pd.DataFrame, bounds: list) -> bool:
    """
    Verifies if constraints and stationarity condition are satisfied.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the NGARCH model parameters.
    bounds : list
        The constraints on the NGARCH model parameters.

    Returns
    -------
    bool
        True if all constraints and stationarity condition are satisfied, False otherwise.
    """

    # Extract the parameters from the DataFrame
    lambda_values = [float(x) for x in df["λ"]]
    omega_values = [float(x) for x in df["ω"]]
    alpha_values = [float(x) for x in df["α"]]
    beta_values = [float(x) for x in df["β"]]
    gamma_values = [float(x) for x in df["γ"]]

    # Check constraints for each parameter
    for i, (lmbda, omega, alpha, beta, gamma) in enumerate(
        zip(lambda_values, omega_values, alpha_values, beta_values, gamma_values)
    ):
        if not (bounds[0][0] <= lmbda <= bounds[0][1]):
            print(f"Constraint not satisfied for λ at index {i}")
            return False

        if not (bounds[1][0] <= omega <= bounds[1][1]):
            print(f"Constraint not satisfied for ω at index {i}")
            return False

        if not (bounds[2][0] <= alpha <= bounds[2][1]):
            print(f"Constraint not satisfied for α at index {i}")
            return False

        if not (bounds[3][0] <= beta <= bounds[3][1]):
            print(f"Constraint not satisfied for β at index {i}")
            return False

        if not (bounds[4][0] <= gamma <= bounds[4][1]):
            print(f"Constraint not satisfied for γ at index {i}")
            return False

        # Check the stationarity condition
        if not (alpha * (1 + gamma**2) + beta < 1):
            print(f"Stationarity condition not satisfied at index {i}")
            return False

    return True


def compute_model_Q1(
    log_xreturns: np.ndarray,
    days_in_year: int,
    horizon: int,
    f_ngarch,
    measure,
    f_out_format_Q1,
    misspecified: bool = False,
) -> Tuple[Any, Any, Any, Any, Any, Union[ngarch, Any], Union[ngarch, Any]]:
    """
    Computes the implications on the parameters and plots for a model, which can be specified (optimized) or misspecified (non-optimized).

    Parameters
    ----------
    log_xreturns : np.ndarray
        The log returns data.
    days_in_year : int
        The number of days in a year.
    horizon : int
        The horizon for the simulations.
    f_ngarch : function
        The function to perform NGARCH estimation.
    Measure : class
        The Measure class.
    f_out_format_Q1 : function
        The function to format the output table for Q1.
    misspecified : bool, optional (default=False)
        Whether the model is misspecified or not.

    Returns
    -------
    Tuple[pd.DataFrame, object]
        The output table, P_1996, P_2020, Q_1996, Q_2020, ng1996, and ng2020.
    """

    # Inputs to initialize the MLE
    time_t = np.datetime64("1996-12-31")
    ng1996 = ngarch.initialize_at(time_t, log_xreturns, days_in_year)

    time_t = np.datetime64("2020-02-01")
    ng2020 = ngarch.initialize_at(time_t, log_xreturns, days_in_year)

    if not misspecified:
        # MLE + new NGARCH parameters
        ng1996 = f_ngarch(ng1996)
        ng2020 = f_ngarch(ng2020)

    # Output table
    out_Q1 = f_out_format_Q1([ng1996, ng2020])

    # Inputs for the simulations
    n_days = horizon * days_in_year
    n_paths = 10000

    # Simulation under P
    P_1996 = measure(*ng1996.simulateP(100, n_days, n_paths, ng1996.P_predict_h()))
    P_2020 = measure(
        *ng2020.simulateP(100, n_days, n_paths, ng2020.P_predict_h(), P_1996.z)
    )

    # Simulation under Q
    Q_1996 = measure(
        *ng1996.simulateQ(100, n_days, n_paths, ng1996.Q_predict_h(), P_1996.z)
    )
    Q_2020 = measure(
        *ng2020.simulateQ(100, n_days, n_paths, ng2020.Q_predict_h(), P_1996.z)
    )

    return out_Q1, P_1996, P_2020, Q_1996, Q_2020, ng1996, ng2020



def f_add_DTM(option_info: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days to maturity column from exdate and date columns.

    Args:
        option_info: A dataframe containing option information.

    Returns:
        The same dataframe with a new column called DTM.

    """
    option_info = option_info.copy()
    option_info["DTM"] = np.array(option_info["exdate"], dtype="datetime64") - np.array(
        option_info["date"], dtype="datetime64"
    )
    option_info["DTM"] = option_info["DTM"].dt.days
    return option_info


def f_add_moneyness(option_info: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column called K/S to option_info, which is moneyness.

    Args:
        option_info: A dataframe containing option information.
        spx: A dataframe containing SPX prices.

    Returns:
        The same dataframe with a new column called K/S.

    """
    option_info["K/S"] = np.zeros(len(option_info.index))

    date = option_info["date"].unique()

    prices = [spx[spx["date"] == d]["close"] for d in date]
    option_info.loc[:, ["strike_price"]] = option_info["strike_price"].div(1000)
    for i in range(len(prices)):
        option_info.loc[option_info["date"] == date[i], "K/S"] = option_info[
            option_info["date"] == date[i]
        ]["strike_price"].div(float(prices[i]))

    return option_info


def f_describe_table(option_info: pd.DataFrame, spx: pd.Series) -> pd.DataFrame:
    """
    Creates a descriptive table for the given option information.

    Parameters
    ----------
    option_info : pd.DataFrame
        The option information DataFrame.
    spx : pd.Series
        The S&P 500 index price data.

    Returns
    -------
    pd.DataFrame
        The descriptive table.
    """

    # Drop rows with missing implied volatility
    option_info.dropna(subset=["impl_volatility"], how="any", inplace=True)

    # Add days-to-maturity and moneyness columns
    option_info = f_add_DTM(option_info)
    option_info = f_add_moneyness(option_info, spx)

    # Reorganize data and show descriptive table
    warnings.simplefilter("ignore")
    table = option_info.groupby(["date", "cp_flag"])[
        ["strike_price", "impl_volatility", "delta", "DTM", "K/S"]
    ].describe()
    # Select specific statistics for each column
    table["strike_price"] = table["strike_price"][["count", "min", "max"]]
    table["impl_volatility"] = table["impl_volatility"][["50%", "min", "max"]]
    table["delta"] = table["delta"][["50%", "min", "max"]]
    table["DTM"] = table["DTM"][["50%", "min", "max"]]
    table["K/S"] = table["K/S"][["50%", "min", "max"]]
    warnings.resetwarnings()

    return table


def f_clean_table(
    option_info: pd.DataFrame, spx: pd.Series, keep_moneyness: bool = False
) -> pd.DataFrame:
    """
    Cleans and filters the option information DataFrame.

    Parameters
    ----------
    option_info : pd.DataFrame
        The option information DataFrame.
    spx : pd.Series
        The S&P 500 index price data.
    keep_moneyness : bool, optional
        If True, keep the moneyness column in the resulting DataFrame, by default False.

    Returns
    -------
    pd.DataFrame
        The cleaned and filtered option information DataFrame.
    """

    # Drop rows with missing implied volatility
    option_info.dropna(subset=["impl_volatility"], how="any", inplace=True)
    # Add days-to-maturity column and filter by DTM
    option_info = f_add_DTM(option_info)
    option_info = option_info[option_info["DTM"] > 7]
    option_info = option_info[option_info["DTM"] < 550]

    # List of columns to keep
    keep_col = [
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "volume",
        "best_bid",
        "mean_bidask",
        "open_interest",
        "impl_volatility",
        "DTM",
    ]

    if keep_moneyness:
        option_info = f_add_moneyness(option_info, spx)
        keep_col.append("K/S")

    # Convert date and exdate columns to datetime format
    option_info["date"] = pd.to_datetime(option_info["date"])
    option_info["exdate"] = pd.to_datetime(option_info["exdate"])
    # Compute mean bid-ask price
    option_info["mean_bidask"] = (
        option_info["best_bid"] + option_info["best_offer"]
    ) / 2
    # Remove duplicate rows
    option_info = option_info.drop_duplicates(
        subset=["date", "exdate", "cp_flag", "strike_price"]
    )

    return option_info[keep_col]


def f_add_Q3_info(option_info: pd.DataFrame, days_in_year: int = 252) -> pd.DataFrame:
    """
    Adds additional information (prices, dividend rates, risk-free rates, and moneyness)
    to the option information DataFrame.
        Parameters
    ----------
    option_info : pd.DataFrame
        The option information DataFrame.
    days_in_year : int, optional
        The number of days in a year, used for calculations, by default 252.

    Returns
    -------
    pd.DataFrame
        The updated option information DataFrame with additional information.
    """

    # Add stock prices and dividend rates to the option_info DataFrame
    option_info["S_t"] = get_price(option_info.date)
    option_info["y_t"] = get_dividend_rate(option_info.date)

    # Add risk-free rates to the option_info DataFrame
    date_unique = np.unique(option_info.date)
    rf = [
        get_risk_free_rate(date, option_info[option_info.date == date].DTM)
        for date in date_unique
    ]
    option_info["r_f"] = np.concatenate(([arr for arr in rf]))

    # Add moneyness (K/F) to the option_info DataFrame
    K = option_info.strike_price / 1000
    F = option_info.S_t * np.exp(
        (option_info.r_f - option_info.y_t) * option_info.DTM / days_in_year
    )
    option_info["F"] = F
    option_info["K/F"] = K / F

    return option_info


def f_F_CBOE(option_info: pd.DataFrame, days_in_year: int = 252) -> pd.DataFrame:
    """
    Calculates the CBOE forward price and adds it to the option information DataFrame.

    Parameters
    ----------
    option_info : pd.DataFrame
        The option information DataFrame.
    days_in_year : int
        The number of days in a year, used for calculations.

    Returns
    -------
    pd.DataFrame
        The updated option information DataFrame with the CBOE forward price added.
    """

    # Add a new column for the CBOE forward price, initially filled with NAs
    option_info["F_CBOE"] = pd.NA

    # Get unique DTM and date values
    DTM_unique = np.unique(option_info.DTM)
    date_unique = np.unique(option_info.date)

    # Loop over unique DTM values
    for dtm in DTM_unique:
        option_DTM_i = option_info[option_info.DTM == dtm]

        # Loop over unique dates
        for date in date_unique:
            # Keep options for the correct date and with both put and call for a strike
            option_DTM_ti = option_DTM_i[option_DTM_i.date == date].sort_values(
                ["strike_price", "cp_flag"]
            )

            # Filter out options with zero best_bid
            df = option_DTM_ti[option_DTM_ti.best_bid != 0]

            # Keep only options with duplicate strike_price values (i.e., both put and call)
            df = option_DTM_ti[
                option_DTM_ti.duplicated(subset=["strike_price"], keep=False)
            ]

            if not option_DTM_ti.empty:
                # Calculate the difference between the mean_bidask for each pair of put and call options
                diff = df.groupby("strike_price")["mean_bidask"].apply(
                    lambda x: x[x.index[0]] - x[x.index[1]]
                )

                if not diff.empty:
                    # Find the strike_price with the minimum absolute difference
                    K_diff_min = diff.abs().idxmin()
                    diff_min = diff[K_diff_min]

                    # Calculate the CBOE forward price
                    forward = (K_diff_min / 1000) + diff_min * np.exp(
                        option_DTM_ti.r_f * dtm / days_in_year
                    )

                    # Update the F_CBOE column with the calculated forward price
                    option_info.loc[option_DTM_ti.index, "F_CBOE"] = forward

    return option_info


def create_comparison_df(option_info):
    """
    Crée un DataFrame comparant les valeurs EX1 et EX2 pour une date donnée.

    Parameters
    ----------
    option_info : pd.DataFrame
        DataFrame contenant les informations sur les options, y compris les dates, DTM, exdiv_1 et exdiv_2.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les colonnes 'DTM_2020', 'EX1_2020', 'EX2_2020', 'DIFF' et 'DIFF_BOOL'.
    """
    date_unique = np.unique(option_info["date"])
    dtm_unique = np.unique(option_info.loc[option_info.date == date_unique[1]].DTM)
    ex1_unique = np.unique(option_info.loc[option_info.date == date_unique[1]].exdiv_1)
    ex2_unique = np.unique(option_info.loc[option_info.date == date_unique[1]].exdiv_2)

    comparison_df = pd.DataFrame(
        {
            "DTM_2020": dtm_unique,
            "EX1_2020": ex1_unique,
            "EX2_2020": ex2_unique,
            "DIFF": ex1_unique - ex2_unique,
            "DIFF_BOOL": (ex1_unique - ex2_unique) == 0,
        }
    )

    return comparison_df


def f_plot_Q3_comparison(option_info: pd.DataFrame) -> np.ndarray:
    """
    Trace un graphique comparant les prix ex-dividendes pour deux méthodes différentes
    pour deux dates différentes.

    Paramètres :
    -----------
    option_info : pd.DataFrame
        DataFrame contenant les données des options.

    Renvoie :
    ---------
    axes : np.ndarray
        Tableau contenant les axes des graphiques.
    """
    # Titres des graphiques
    title = ["1996-12-03", "2020-02-03"]

    # Supprimer les doublons pour chaque date
    df = option_info.drop_duplicates(subset=["date", "DTM", "exdiv_1", "exdiv_2"])
    # Garder uniquement les colonnes nécessaires et trier par date et DTM
    df = (
        df[["date", "DTM", "exdiv_1", "exdiv_2"]]
        .sort_values(["date", "DTM"])
        .reset_index(drop=True)
    )

    df = df.dropna()

    # Initialiser les indices des graphiques
    i = 1
    # Créer un tableau de graphiques de taille 2x1
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for date in np.unique(option_info.date):
        # Obtenir les données pour une date donnée
        data = df[df["date"] == date]

        # Créer un nouveau graphique
        ax = plt.subplot(2, 1, i)
        # Tracer les points pour les deux méthodes
        ax.scatter(
            data["DTM"],
            data["exdiv_1"],
            label="Methode 1",
            s=15,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.scatter(
            data["DTM"],
            data["exdiv_2"],
            label="Methode 2",
            s=15,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )

        ax.set_title(title[i - 1])
        ax.set_ylabel("Prix ex-dividende")
        ax.legend()
        ax.set_xscale("log")
        # Changer les marques de l'axe x
        ax.set_xticks([20, 50, 150, 400])
        # Utiliser un formatteur scalaire pour l'axe x
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        i += 1

    ax.set_xlabel("Jours à maturité")

    return axes


def f_plot_Q4_smiles(option_info: pd.DataFrame, date: str, date_str: str) -> None:
    """
    Trace un graphique de sourire de volatilité pour une date donnée.

    Paramètres :
    -----------
    option_info : pd.DataFrame
        DataFrame contenant les données des options.
    date : str
        Date pour laquelle tracer le graphique.
    date_str : str
        Chaîne de caractères pour le titre du graphique.

    Renvoie :
    ---------
    None
    """
    option_info_t = option_info[option_info.date == date]
    DTM_unique = np.unique(option_info_t.DTM)

    y_sub = math.ceil(len(DTM_unique) / 3)
    fig, axes = plt.subplots(y_sub, 3, figsize=(42, 30))

    i = 1
    for dtm in DTM_unique:
        # Garder seulement les options OTM
        data_call = option_info_t[
            (option_info_t.DTM == dtm) & (option_info_t.cp_flag == "C")
        ]
        data_put = option_info_t[
            (option_info_t.DTM == dtm) & (option_info_t.cp_flag == "P")
        ]

        moneyness_call = data_call.strike_price / (1000 * data_call.S_t)
        moneyness_put = data_put.strike_price / (1000 * data_put.S_t)

        warnings.simplefilter("ignore")
        data_call["K/S"] = moneyness_call
        data_put["K/S"] = moneyness_put
        warnings.resetwarnings()

        data_call_OTM = data_call[data_call["K/S"] >= 1].sort_values(["K/S"])
        data_put_OTM = data_put[data_put["K/S"] <= 1].sort_values(["K/S"])

        data_plot = pd.concat([data_put_OTM, data_call_OTM])

        ax = plt.subplot(y_sub, 3, i)

        ax.plot(
            data_plot["K/S"],
            data_plot["IV_method1"],
            "g--",
            linewidth=2,
            label="Methode 1",
        )
        ax.plot(
            data_plot["K/S"],
            data_plot["IV_method2"],
            "r--",
            linewidth=2,
            label="Methode 2",
        )
        ax.plot(
            data_plot["K/S"],
            data_plot["impl_volatility"],
            "b--",
            linewidth=2,
            label="OptionMetrics",
        )
        x_pos = (max(data_plot["K/S"]) + min(data_plot["K/S"])) * 0.5
        y_pos = (
            (max(data_plot["IV_method1"]) + min(data_plot["IV_method1"])) * 0.5 * 1.25
        )
        ax.text(x_pos, y_pos, "DTM = " + str(dtm) + " days", fontsize=25)

        i += 1

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes][0]
    fig.legend(lines_labels[0], lines_labels[1], loc="upper left", fontsize=45)
    fig.suptitle(
        "Volatility smiles : " + date_str + "\n Graphique de IV vs K/S", fontsize=75
    )
    plt.show()

    return None


def simulate_returns(
    option_info: pd.DataFrame, ng1996: ngarch, ng2020: ngarch, n_paths: int = 100_000
) -> pd.DataFrame:
    """
    Simule les rendements pour chaque option dans option_info à partir des modèles ngarch pour les années 1996 et 2020.

    Paramètres :
    ------------
    option_info : pd.DataFrame
        DataFrame contenant les données des options.
    ng1996 : ngarch
        Modèle ngarch pour l'année 1996.
    ng2020 : ngarch
        Modèle ngarch pour l'année 2020.
    n_paths : int, optionnel (défaut=100_000)
        Le nombre de chemins de simulations pour chaque modèle.

    Renvoie :
    ---------
    option_info : pd.DataFrame
        DataFrame avec une nouvelle colonne 'R_j' contenant les rendements simulés.
    """
    # Obtenir les dates et les DTM uniques dans le DataFrame option_info
    date_unique = np.unique(option_info.date)
    DTM_unique = np.unique(option_info.DTM)

    # Nombre de jours de négociation dans les années 1996 et 2020
    n_days96 = max(option_info[option_info.date == date_unique[0]].DTM)
    n_days20 = max(option_info[option_info.date == date_unique[1]].DTM)

    # Simuler les rendements pour les années 1996 et 2020
    ex_r_96 = measure(
        *ng1996.simulateQ(1, n_days96, n_paths, ng1996.Q_predict_h())
    ).ex_r
    ex_r_20 = measure(
        *ng2020.simulateQ(1, n_days20, n_paths, ng2020.Q_predict_h())
    ).ex_r

    # Ajouter une nouvelle colonne 'R_j' au DataFrame option_info
    option_info["R_j"] = pd.NA
    warnings.simplefilter("ignore")

    # Itérer sur les DTM et les dates uniques
    for dtm in DTM_unique:
        option_DTM_i = option_info[option_info.DTM == dtm]
        for date in date_unique:
            option_DTM_ti = option_DTM_i[option_DTM_i.date == date]
            if not option_DTM_ti.empty:
                if pd.DatetimeIndex([date]).year == 1996:
                    ex_r = ex_r_96
                else:
                    ex_r = ex_r_20

                # Simuler les rendements pour le DTM actuel
                R_j = np.exp(np.apply_along_axis(sum, 0, ex_r[:dtm]))
                # Ajouter les rendements simulés au DataFrame option_info
                for i in option_DTM_ti.index:
                    option_info.R_j[i] = R_j

    warnings.resetwarnings()

    return option_info


def calculate_option_prices(
    option_info: pd.DataFrame, ng1996: ngarch, ng2020: ngarch
) -> pd.DataFrame:
    """Calculate the option prices based on the ngarch models.

    Args:
        option_info: DataFrame containing the option information.
        ng1996: ngarch model fitted to data from 1996-12-03.
        ng2020: ngarch model fitted to data from 2020-02-03.

    Returns:
        DataFrame with the option prices.

    """
    # Get unique dates from the option_info DataFrame
    date_unique = np.unique(option_info.date)

    # Initialize the Option_price column with pd.NA values
    option_info["Option_price"] = pd.NA

    # Split the option_info DataFrame based on the unique dates
    option_info_96 = option_info[option_info.date == date_unique[0]]
    option_info_20 = option_info[option_info.date == date_unique[1]]

    # Calculate the option prices using the ngarch models and the option_info DataFrames split by dates
    option_price_96 = [
        ng1996.option_price(
            info.R_j,
            info.F_CBOE,
            info.strike_price / 1000,
            info.r_f,
            info.DTM,
            info.cp_flag == "C",
        )
        for _, info in option_info_96.iterrows()
    ]
    option_info.loc[option_info_96.index, "Option_price"] = option_price_96

    option_price_20 = [
        ng2020.option_price(
            info.R_j,
            info.F_CBOE,
            info.strike_price / 1000,
            info.r_f,
            info.DTM,
            info.cp_flag == "C",
        )
        for _, info in option_info_20.iterrows()
    ]
    option_info.loc[option_info_20.index, "Option_price"] = option_price_20

    # Return the option_info DataFrame with the calculated option prices
    return option_info
