"""The Black Merton Scholes economy

Local implementation of some useful functions for the BMS model. 
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

# Local packages
from jupyter_notebook import *


def d1(S, K, r, y, T, sigma):
    """Calculate d1 from the Black, Merton and Scholes formula"""
    return (np.log(S / K) + (r - y + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, r, y, T, sigma):
    """Calculate d2 from the Black, Merton and Scholes formula"""
    return (np.log(S / K) + (r - y - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def delta(S, K, r, y, T, sigma, is_call):
    """Return Black, Merton, Scholes delta of the European (call, put)"""
    _d1 = d1(S, K, r, y, T, sigma)
    d_sign = np.where(is_call, 1, -1)
    return d_sign * norm.cdf(d_sign * _d1)


def gamma(S, K, r, y, T, sigma, is_call=None):
    """Return Black, Merton, Scholes gamma of the European call or put

    Accepts is_call argument for consistency in the functions' signatures, but it is neglected
    """
    _d1 = d1(S, K, r, y, T, sigma)
    return np.exp(-y * T) * norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def theta(S, K, r, y, T, sigma, is_call):
    _d1 = d1(S, K, r, y, T, sigma)
    _d2 = _d1 - sigma * np.sqrt(T)
    d_sign = np.where(is_call, 1, -1)
    return (
        -np.exp(-y * T) * S * norm.pdf(_d1) * sigma / (2 * np.sqrt(T))
        - d_sign * r * K * np.exp(-r * T) * norm.cdf(d_sign * _d2)
        + d_sign * y * S * np.exp(-y * T) * norm.cdf(d_sign * _d1)
    )


def vega(S, K, r, y, T, sigma, is_call=None):
    """Return Black, Merton, Scholes vega of the European call or put

    Accepts is_call argument for consistency in the functions' signatures, but it is neglected
    """
    _d1 = d1(S, K, r, y, T, sigma)
    return S * np.exp(-y * T) * norm.pdf(_d1) * np.sqrt(T)


def option_price(S, K, r, y, T, sigma, is_call, ret_delta=False):
    """Return Black, Merton, Scholes price of the European option"""
    _d1 = d1(S, K, r, y, T, sigma)
    _d2 = _d1 - sigma * np.sqrt(T)

    # d_sign: Sign of the the option's delta
    d_sign = np.where(is_call, 1, -1)
    delta = d_sign * norm.cdf(d_sign * _d1)
    premium = np.exp(-y * T) * S * delta - d_sign * np.exp(-r * T) * K * norm.cdf(
        d_sign * _d2
    )
    if ret_delta:
        return premium, delta
    return premium


def _implied_volatility(opt_price, S, K, r, y, T, is_call, init_vol=0.6):
    """Inverse the BMS formula numerically to find the implied volatility"""

    def pricing_error(sig):
        sig = abs(sig)
        return option_price(S, K, r, y, T, sig, is_call) - opt_price

    return fsolve(pricing_error, init_vol)


implied_volatility = np.vectorize(_implied_volatility)


def simulate_underlying(S0, r, y, sigma, dt, shocks):
    """Simulate the GMB based on the user-provided standard normal shocks

    Parameters:
        S, r, y, sigma, : as usual
        dt     : the time step length in the simulation
        shocks : A (n_steps x n_sim) matrix of standard Normal shocks for a
                 simulation with n_steps time steps and n_sim paths

    Returns:
        S : A (n_steps+1 x n_sim) matrix with n_sim paths of the underlying simulated over n_steps time steps,
            starting at time 0
    """
    n_steps, n_sim = shocks.shape
    S = np.empty((n_steps + 1, n_sim))
    S[0, :] = S0
    for tn in range(n_steps):
        S[tn + 1, :] = S[tn, :] * np.exp(
            (r - y - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks[tn, :]
        )
    return S


def delta_hedge(premium, S_t, K, r, y, T, sigma, dt, is_call):
    """Delta hedges a **short** position in the option across simulated paths S_t

    S_t must start with a row including the current value of the stock
    """
    raise RuntimeError("Implement for Assignment 1")


class _moneyness:
    class __moneyness(struct):
        def __init__(self, mny_str, **kwargs):
            super().__init__(**kwargs)
            self.mny_str = mny_str

        def range(self, S, lb, ub, step):
            """Returns an np.arange(lb,ub,step) of moneyness levels and corresponding strikes"""
            mny = np.arange(lb, ub, step)
            K = self.get_strike(mny, S)
            return mny, K

        def delta_to_strike(self, delta, S, is_call):
            sigma = self.sigma
            T = self.T
            ln_S_K = (
                norm.ppf(delta) * sigma * np.sqrt(T)
                - (self.r - self.y + 0.5 * sigma**2) * T
            )
            return S * np.exp(-ln_S_K)

    class K_over_S(__moneyness):
        def __call__(self, S, K, r=None, y=None, T=None, sigma=None):
            return K / S

        def get_strike(self, mny, S):
            return mny * S

    class K_over_F(__moneyness):
        def forward_price(self, S, r=None, y=None, T=None, *args, **kwargs):
            """Allow for neglected arguments; for necessary ones, use instance's values when None"""
            if r is None:
                r = self.r
            if y is None:
                y = self.y
            if T is None:
                T = self.T
            return S * np.exp((r - y) * T)

        def __call__(self, S, K, *args, **kwargs):
            return K / self.forward_price(S, *args, **kwargs)

        def get_strike(self, mny, S):
            return mny * self.forward_price(S)

    # Class variable will be shared by all instances of _moneyness
    instance = None

    def __init__(self, mny, **kwargs):
        """Must provide the sufficient variables such that moneyness(S,K) returns the corresponding moneyness."""
        if mny == "K/S":
            _moneyness.instance = _moneyness.K_over_S(mny, **kwargs)
        elif mny == "K/F":
            _moneyness.instance = _moneyness.K_over_F(mny, **kwargs)
        else:
            raise ValueError(mny)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)

    def __repr__(self):
        return repr(self.instance)

    def __str__(self):
        return str(self.instance)


# Given the voodoo above, moneyness is a handle to the sole _moneyness.instance of __moneyness
moneyness = _moneyness("K/S")  # Default moneyness requires no additional info


def define_moneyness(mny, **kwargs):
    """Set the MNY definition to be used across figures & tables

    Currently supported MNY: 'K/S' or 'K/F'.

    **kwargs ust provide the sufficient variables such that moneyness(S,K) returns the
    corresponding moneyness.
      - K/S: nothing needed
      - K/F: r=risk_free_rate, y=dividend_yield, T=maturity,
    """
    return _moneyness(mny, **kwargs)
