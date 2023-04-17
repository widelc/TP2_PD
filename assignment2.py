import datetime as dt
import os
import sys
import math
import inspect
import warnings
import pandas as pd

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from pprint import pprint
from math import pi


if os.getcwd().startswith('/Users/christian/'):
    sys.path.append("../..") 
from jupyter_notebook import * 
import black_merton_scholes as bms
from monte_carlo import antithetic_normal

def load_zero_curve():
    """Load rates. Makes sure that within a date, rates are sorted by maturity."""
    rf = pd.read_csv('zerocd.csv', index_col=0).reset_index(drop=True)
    rf['date'] = pd.to_datetime(rf['date'])
    return rf.sort_values(by=['date','days'])

def load_dividend():
    y = pd.read_csv('distrd_108105.csv', index_col=0).reset_index(drop=True)
    y['date'] = pd.to_datetime(y['date'])
    return y.sort_values(by=['date'])[['date', 'rate']]

def load_price():
    p = pd.read_csv('secprd_108105.csv', index_col=0).reset_index(drop=True)
    p['date'] = pd.to_datetime(p['date'])
    return p.sort_values(by=['date'])[['date', 'close']]

def get_dividend_rate(time_t):
    y = load_dividend()

    date_ref = min([min(y.date),min(time_t)])
    interp   = interp1d((y.date - date_ref).apply(lambda x : x.days), 
                       y.rate/100, 
                       kind='linear')
    
    return interp((time_t - date_ref).apply(lambda x : x.days))

def get_price(time_t):
    p = load_price()
    date_ref = min([min(p.date),min(time_t)])
    interp   = interp1d((p.date - date_ref).apply(lambda x : x.days), 
                       p.close, 
                       kind='linear')
    
    return interp((time_t - date_ref).apply(lambda x : x.days))

def get_risk_free_rate(time_t, dtm):
    rf = load_zero_curve()
    rf = rf[rf.date == time_t]

    # Interpolate between the 2 closest maturities
    interp = interp1d(rf.days, rf.rate/100, kind='linear')
    return interp(dtm)

def get_log_excess_returns(days_in_year):
    # First, read the data on the underlying (108105 is the secid of the SP500 in OptionMetrics)
    spx = pd.read_csv('secprd_108105.csv', index_col=0).reset_index(drop=True)
    spx['date'] = pd.to_datetime(spx['date']) # convert str to actual dates
    assert len(spx.cfadj.unique())==1 # Make sure there are no splits unaccounted for
    spx['log_ret'] = np.log(spx['close'] / spx['close'].shift(1)) # compute log-returns
    
    # Then, load rates. The function makes sure that within a date, rates are sorted by maturity
    rf = load_zero_curve()
    
    # Then, select the shortest-maturity rate on each day
    #  ffill(): when a date in the resampling is not in rf, use the last observed value
    short_term = rf.resample('D',on='date').agg(['first']).ffill()
    short_term.columns = short_term.columns.get_level_values(0) # get rid of 'first' in column names
    short_term.date = short_term.index # Get rid of the ffill'ed value on date
    short_term = short_term.reset_index(drop=True) # avoids conflicts in the merge below
    
    # And re-express it in daily log-returns
    #  Formally, we could account for 1/365 day of interest rate between 2 weekdays,
    #  and 3/365 between the Friday close and the Monday close... We'll keep things simple here.
    short_term['rf'] = short_term['rate']/100 / days_in_year
    
    # Finally, get *excess* log-returns (log_retx)
    spx = spx.merge(short_term, left_on='date', right_on='date', how='inner')
    spx['log_xret'] = spx['log_ret'] - spx['rf'].shift(1) # rf is known at the beginning of the (1-day) period
    assert np.abs(spx.log_ret.iloc[1] - spx.rf.iloc[0] - spx.log_xret.iloc[1]) < 1e-16
    spx = spx.set_index('date')
    return spx['log_xret']


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
        
        ng = cls() # calls the constructor of the class (i.e. ngarch(), but robust to inheritance...)
        ng.lmbda = 0.01049 # lambda is a reserved word in Python   ##\simeq np.log(1.06) / (ng.days_in_year*np.sqrt(ng.uncond_var()))
        ng.omega = np.nan
        ng.alpha = 6.2530e-2
        ng.beta  = 0.90825
        ng.gamma = 0.5972
        
        ng.days_in_year = days_in_year
        Dt = 1 / days_in_year
        
        ng.omega = ng.variance_targeting( log_xret.var() )
        # print(ng)
        # print('Persistence:', ng.persistence())
        # print('Unconditional volatility:', np.sqrt(ng.uncond_var()/Dt))
        
        ng.log_xret = log_xret
        return ng

    
    def variance_targeting(self, var_target):
        omega = (1 - self.persistence())*var_target
        return omega
        
    def persistence(self):
        return self.alpha*(1 + self.gamma**2) + self.beta

    def uncond_var(self):
        return self.omega/(1 - self.persistence())
    
    def cond_var(self):
        h2 = self.P_predict_h() ** 2
        return 2 * (self.alpha ** 2) * h2 *(1 + 2 * (self.gamma ** 2)) 
    
    def corr_ret_var(self):
        ht = self.P_predict_h()
        return -2*self.gamma / np.sqrt(2 + 4*self.gamma**2)
    
    def P_predict_h(self):
        theta     = [self.lmbda, self.alpha, self.beta, self.gamma]
        h_t, eps  = f_ht_NGARCH(theta, self) 
        return self.omega + self.alpha * h_t[-1] * ((eps[-1] - self.gamma) ** 2) + self.beta * h_t[-1]
    
    def Q_predict_h(self):
        theta     = [self.lmbda, self.alpha, self.beta, self.gamma]
        h_t, eps  = f_ht_NGARCH(theta, self)
        return self.omega + self.alpha * h_t[-1] * ((eps[-1] - self.gamma - self.lmbda) ** 2) + self.beta * h_t[-1]
    
    def simulateP(self, S_t0, n_days, n_paths, h_tp1, z=None):
        '''Simulate excess returns and their variance under the P measure
        
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
        '''
        ex_r   = np.full((n_days, n_paths), np.nan)
        h      = np.full((n_days+1, n_paths), np.nan)
        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)

        # Going from 1 day to n_days ahead
        #  t0+1      is tn==0 because indices start at 0 in Python
        #  t0+n_days is consequently tn==n_days-1
        for tn in range(0,n_days):            
            # Simulate returns            
            sig        = np.sqrt(h[tn,:])
            ex_r[tn,:] = self.lmbda*sig - 0.5*h[tn,:] + sig*z[tn,:]
            
            # Update the variance paths
            h[tn+1,:] = self.omega + self.alpha*h[tn,:]*(z[tn,:] - self.gamma)**2 + self.beta*h[tn,:]

        return ex_r, h, z

    def simulateQ(self, S_t0, n_days, n_paths, h_tp1, z=None):
        
        ex_r   = np.full((n_days, n_paths), np.nan)
        h      = np.full((n_days+1, n_paths), np.nan)
        h[0,:] = h_tp1 

        if z is None:
            z = antithetic_normal(n_days, n_paths)

        for tn in range(0,n_days):            
            # Simulate returns            
            sig        = np.sqrt(h[tn,:])
            ex_r[tn,:] = -0.5*h[tn,:] + sig*z[tn,:]
            
            # Update the variance paths
            h[tn+1,:] = self.omega + self.alpha*h[tn,:]*(z[tn,:] - self.gamma - self.lmbda)**2 + self.beta*h[tn,:]

        return ex_r, h, z
    
    def option_price(self, cum_ex_r, F_t0_T, K, rf, dtm, is_call):
        
        cp     = [1 if cond else -1 for cond in [is_call]][0]
        disc   = np.exp(-rf * dtm / self.days_in_year)
        payoff = np.maximum(0,((F_t0_T * cum_ex_r) - K) * cp)

        option_price = np.mean(disc * payoff)

        return option_price

    
def plot_excess_return_forecasts(horizon, P, Q, annualized=False):
    ann = [1.0]
    y_prefix = ''
    if annualized:
        ann = horizon
        y_prefix = 'Annualized '

    fig,axes = plt.subplots(2,1, figsize=(14,10))

    ax = plt.subplot(2,1, 1)
    ax.plot(horizon, P.expected_ex_r/ann[0], label='P ex. returns forecasts (sim)')
    ax.plot(horizon, Q.expected_ex_r/ann[0], label='Q ex. returns forecasts (sim)')
    ax.legend(loc='upper right')   
    ax.set_ylabel(y_prefix+'Excess Returns')

    ax = plt.subplot(2,1, 2)
    # ERP is accumulated between t0 and horizon h (think about a buy and hold position)
    erp = np.cumsum(P.expected_ex_r - Q.expected_ex_r)/ann # Theoretically, the "- Q.expected_ex_r" part is useless here
    ax.plot(horizon, erp)    
    ax.set_ylabel(y_prefix+'Equity Risk Premium')
    ax.set_xlabel('Years to Maturity')

    return axes

def plot_var_forecasts(horizon, P, Q, annualized=False):
    """Plot variance forecasts and variance risk premium.

    Args:
        horizon: Maturities to compute forecasts.
        P: List of ngarch models of P series.
        Q: List of ngarch models of Q series.
        annualized: Whether to annualize the variance.

    Returns:
        Axes of the plot

    """

    title = ['1996-12-03', '2020-02-03']
    ann = [1.0]
    y_prefix = ''
    if annualized:
        ann = horizon
        y_prefix = 'Annualized '

    fig,axes = plt.subplots(2,len(P), figsize=(14,10))
    j = 0
    for i in range(len(P)):
        Pi = P[i]
        Qi = Q[i]

        #ax = plt.subplot(2,2, j)
        axes[j,i].plot(horizon, np.mean(Pi.h,axis=1)/ann[0], label='P variance forecasts (sim)')
        axes[j,i].plot(horizon, np.mean(Qi.h,axis=1)/ann[0], label='Q variance forecasts (sim)')
        axes[j,i].legend(loc='best')   
        axes[j,i].set_ylabel(y_prefix+'Variance')
        axes[j,i].set_title(title[i])

        #ax = plt.subplot(2,2, j)
        j += 1
        vrp = np.cumsum(np.mean(Pi.h,axis=1) - np.mean(Qi.h,axis=1))/ann 
        axes[j,i].plot(horizon, vrp)    
        axes[j,i].set_ylabel(y_prefix+'Variance Risk Premium')
        axes[j,i].set_xlabel('Years to Maturity')
        j -= 1

    return axes
    

def f_ht_NGARCH(theta, ng):
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
    h_t : conditional variances.
    eps : innovations.

    """
    # Extract data from ng
    lmbda = theta[0]    
    omega = ng.omega
    alpha = theta[1]
    beta  = theta[2]
    gamma = theta[3]

    # Get rid of the NA at the beggining
    log_xreturns = ng.log_xret[1:]
    
    # Initialize the conditional variances
    T   = len(log_xreturns)
    h_t = np.full(T, np.nan)
    eps = np.full(T, np.nan)
  
    # Start with unconditional variances
    h_t_ini = omega / (1 - alpha * (1 + (gamma ** 2)) - beta)
    eps_ini = (log_xreturns[0] - lmbda * np.sqrt(h_t_ini) + 0.5 * h_t_ini) / np.sqrt(h_t_ini)

    h_t[0] = h_t_ini
    eps[0] = eps_ini
    
    # Compute conditional variance and innovations at each step
    for t in range(T-1):
        h_t[t+1] = omega + alpha * h_t[t] * ((eps[t] - gamma)**2) + beta * h_t[t]
        eps[t+1] = (log_xreturns[t+1] - lmbda * np.sqrt(h_t[t+1]) + 0.5 * h_t[t+1]) / np.sqrt(h_t[t+1])
    
    return h_t, eps

def f_nll_NGARCH(theta, ng):
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
    h, eps = f_ht_NGARCH(theta, ng)
    nll    = -0.5 * np.sum(np.log(2 * pi * h) + eps ** 2)
    
    return -nll

def f_NGARCH(ng):
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
    warnings.simplefilter('ignore')
    bounds = [(0, None), (0, 1), (0.5, 1), (0, None)]
    ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([1 - x[1] * (1 + (x[3] ** 2)) - x[2]])}
    opt    = minimize(fun     = f_nll_NGARCH, 
                      x0      = theta_0, 
                      args    = ng,
                      method  = 'SLSQP',  
                      bounds  = bounds,
                      constraints = ineq_cons,
                      options = {'maxiter': 5000})
    warnings.resetwarnings()
 
    # Update parameters value in ng
    param = opt.x
    ng.lmbda = param[0]
    ng.alpha = param[1]
    ng.beta  = param[2]
    ng.gamma = param[3]

    return ng

def f_out_format_Q1(ng_vec):

    # Output table
    days_in_year = ng_vec[0].days_in_year
    data = {'λ'      : ["{:.4e}".format(ng.lmbda) for ng in ng_vec], 
            'ω'      : ["{:.4e}".format(ng.omega) for ng in ng_vec],
            'α'      : ["{:.4e}".format(ng.alpha) for ng in ng_vec],
            'β'      : ["{:.4e}".format(ng.beta)  for ng in ng_vec],
            'γ'      : [ng.gamma for ng in ng_vec],
            'Vol. incond.'     : np.multiply(np.sqrt([ng.uncond_var() for ng in ng_vec]), np.sqrt(days_in_year)),
            'Vol. cond. @ t+1' : np.multiply(np.sqrt([ng.P_predict_h() for ng in ng_vec]) , np.sqrt(days_in_year)),
            'Corr. cond. @ t'    : [ng.corr_ret_var() for ng in ng_vec],
            'Vol. cond. @ t de h_t+2' : ["{:.4e}".format(np.sqrt(ng.cond_var())) for ng in ng_vec]}
    
    df   = pd.DataFrame(data=data, index=['1996-12-31', '2020-02-01'])

    return df

def f_add_DTM(option_info):
    """
    Compute days to maturity column from exdate and date columns.

    Args:
        option_info: A dataframe containing option information.

    Returns:
        The same dataframe with a new column called DTM.

    """
    option_info['DTM'] = np.array(option_info['exdate'], dtype='datetime64') - np.array(option_info['date'], dtype='datetime64')
    option_info['DTM'] = option_info['DTM'].dt.days
    return option_info


def f_describe_table(option_info):
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
    # Drop rows with missing IV
    option_info.dropna(subset=['impl_volatility'], how='any', inplace=True)

    # Create DTM column and moneyness
    option_info = f_add_DTM(option_info)
                
    # Reorganize data and show descriptive table
    warnings.simplefilter('ignore')
    table = option_info.groupby(['date','cp_flag'])[['strike_price','impl_volatility','delta','DTM']].describe()
    table['strike_price']    = table['strike_price'][['count','min','max']]
    table['impl_volatility'] = table['impl_volatility'][['min','max']]
    table['delta'] = table['delta'][['min','max']]
    table['DTM']   = table['DTM'][['min','max']]
    warnings.resetwarnings()

    return table

def f_clean_table(option_info):
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
    keep_col    = ['date','exdate','cp_flag', 
                   'strike_price','volume','best_bid', 
                   'mean_bidask', 'open_interest',
                   'impl_volatility','DTM']
    option_info['date']   = pd.to_datetime(option_info['date'])
    option_info['exdate'] = pd.to_datetime(option_info['exdate'])
    option_info['mean_bidask'] = (option_info['best_bid'] + option_info['best_offer']) / 2

    option_info = option_info.drop_duplicates(subset=['date', 'exdate', 'cp_flag', 'strike_price'])

    return option_info[keep_col]

def f_add_Q3_info(option_info):
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
    # Ajouter les prix et les dividende à option_info
    option_info['S_t'] = get_price(option_info.date)
    option_info['y_t'] = get_dividend_rate(option_info.date)

    # Ajouter les taux sans risque à option_info
    date_unique = np.unique(option_info.date)
    rf = [get_risk_free_rate(date, option_info[option_info.date == date].DTM) for date in date_unique]
    option_info['r_f'] = np.concatenate(([arr for arr in rf]))

    return option_info

def f_F_CBOE(option_info, days_in_year):
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
    option_info['F_CBOE'] = pd.NA
    DTM_unique  = np.unique(option_info.DTM)
    date_unique = np.unique(option_info.date)

    for dtm in DTM_unique :
        option_DTM_i  = option_info[option_info.DTM == dtm]
        for date in date_unique:

            # Garder les options pour la bonne date et avec lesquelles on a un put et un call pour un strike
            option_DTM_ti = option_DTM_i[option_DTM_i.date == date].sort_values(['strike_price','cp_flag'])

            df = option_DTM_ti[option_DTM_ti.best_bid != 0]
            df = option_DTM_ti[option_DTM_ti.duplicated(subset = ['strike_price'], keep=False)]

            if not option_DTM_ti.empty:

                diff = df.groupby('strike_price')['mean_bidask'].apply(lambda x: x[x.index[0]] - x[x.index[1]])

                if not diff.empty:
                    
                    K_diff_min = diff.abs().idxmin()
                    diff_min   = diff[K_diff_min]
                    forward    = (K_diff_min / 1000) + diff_min * np.exp(option_DTM_ti.r_f * dtm / days_in_year)

                    option_info.loc[option_DTM_ti.index, 'F_CBOE'] = forward
    
    return option_info

def f_plot_Q3_comparison(option_info):
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
    title = ['1996-12-03', '2020-02-03']

    df = option_info.drop_duplicates(subset=['date', 'DTM','exdiv_1','exdiv_2'])
    df = df[['date', 'DTM','exdiv_1','exdiv_2']].sort_values(['date','DTM']).reset_index(drop = True)
    df = df.dropna()

    i = 1
    fig, axes = plt.subplots(2,1, figsize=(12,8))
    for date in np.unique(option_info.date):

        data = df[df['date'] == date]

        ax = plt.subplot(2,1, i)
        ax.scatter(data['DTM'], data['exdiv_1'], label='Methode 1', s= 15, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.scatter(data['DTM'], data['exdiv_2'], label='Methode 2', s= 15, edgecolor='black', linewidth=0.5, alpha=0.8)

        ax.set_title(title[i-1])
        ax.set_ylabel('Prix ex-dividende')
        ax.legend()
        ax.set_xscale('log')
        ax.set_xticks([20, 50, 150, 400])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        i += 1

    ax.set_xlabel('Jours à maturité')
    
    return axes

def f_impl_vol_Q4(option_info, days_in_year) :
    """
    Calcul les volatiltiés implicites en Question 4
    
    Paramètres :
    -----------
    option_info : pd.DataFrame
        DataFrame contenant les données des options.
    days_in_year: float
        Nombre de jours considéré dans l'année

    Renvoie :
    ---------
    impl_vol_mkt : np.ndarray
        Tableau de IV associé aux option dans option_info
        pour les prix ex-div de la question 3
    """
    warnings.simplefilter('ignore')
    impl_vol_mkt = bms.implied_volatility(opt_price = option_info.mean_bidask, 
                                      S         = [option_info.exdiv_1,option_info.exdiv_2], 
                                      K         = option_info.strike_price / 1000, 
                                      r         = option_info.r_f, 
                                      y         = 0, 
                                      T         = option_info.DTM / days_in_year, 
                                      is_call   = (option_info.cp_flag == 'C'))
    warnings.resetwarnings()
    
    return impl_vol_mkt


def f_plot_Q4_smiles(option_info, date, date_str) : 
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
    DTM_unique    = np.unique(option_info_t.DTM)

    y_sub    = math.ceil(len(DTM_unique)/3)
    fig,axes = plt.subplots(y_sub,3, figsize=(42,30))

    i = 1
    for dtm in DTM_unique:
        
        # Garder seulement les options OTM
        data_call = option_info_t[(option_info_t.DTM == dtm) & (option_info_t.cp_flag == 'C')]
        data_put  = option_info_t[(option_info_t.DTM == dtm) & (option_info_t.cp_flag == 'P')]

        moneyness_call = data_call.strike_price / (1000 * data_call.S_t)
        moneyness_put  = data_put.strike_price / (1000 * data_put.S_t)

        warnings.simplefilter('ignore')
        data_call['K/S'] = moneyness_call
        data_put['K/S']  = moneyness_put
        warnings.resetwarnings()

        data_call_OTM = data_call[data_call['K/S'] >= 1].sort_values(['K/S'])
        data_put_OTM  = data_put[data_put['K/S'] <= 1].sort_values(['K/S'])

        data_plot = pd.concat([data_put_OTM, data_call_OTM])

        ax = plt.subplot(y_sub,3, i)

        ax.plot(data_plot['K/S'], data_plot['IV_method1'],'g--', linewidth=2, label = 'Methode 1')
        ax.plot(data_plot['K/S'], data_plot['IV_method2'],'r--', linewidth=2, label = 'Methode 2')
        ax.plot(data_plot['K/S'], data_plot['impl_volatility'],'b--', linewidth=2, label = 'OptionMetrics')
        x_pos = (max(data_plot['K/S']) + min(data_plot['K/S'])) * 0.5 
        y_pos = (max(data_plot['IV_method1']) + min(data_plot['IV_method1'])) * 0.5 * 1.25
        ax.text(x_pos, y_pos, 'DTM = ' + str(dtm) + ' days', fontsize=25)

        i += 1 

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes][0]
    fig.legend(lines_labels[0], lines_labels[1], loc='upper left', fontsize = 45)
    fig.suptitle('Volatility smiles : ' + date_str + '\n Graphique de IV vs K/S', fontsize = 75)
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

def f_impl_vol_Q5(option_info, days_in_year) :
    """
    Calcul les volatiltiés implicites en Question 5 (NGARCH)
    
    Paramètres :
    -----------
    option_info : pd.DataFrame
        DataFrame contenant les données des options.
    days_in_year: float
        Nombre de jours considéré dans l'année

    Renvoie :
    ---------
    impl_vol_mkt : np.ndarray
        Tableau de IV associé aux option dans option_info
        pour les prix provenant du NGARCH
    """
    warnings.simplefilter('ignore')
    impl_vol_ng = bms.implied_volatility(opt_price = option_info.Option_price, 
                                        S         = option_info.exdiv_2, 
                                        K         = option_info.strike_price / 1000, 
                                        r         = option_info.r_f, 
                                        y         = 0, 
                                        T         = option_info.DTM / days_in_year, 
                                        is_call   = (option_info.cp_flag == 'C'))
    warnings.resetwarnings()
    
    return impl_vol_ng

def f_erreur_tarification(option_info):
    """
    Crée le tableau d'aggrégation demandé pour la question 5
    
    Paramètres :
    -----------
    option_info : pd.DataFrame
        DataFrame contenant les données des options.

    Renvoie :
    ---------
    summary : list
        Liste de tableau d'aggrégation d'erreur de IV entre
        le modèle NGARCH et les modèles des Q3 et Q4. (un tableau
        par date d'évaluation dans le dataset)
    """   
    option_info['Moneyness'] = option_info.strike_price / (1000 * option_info.S_t)
    date_unique = np.unique(option_info.date)
    summary = []
    for date in date_unique:

        option_info_d = option_info[option_info.date == date]

        # Calculer les erreurs de tarification pour chaque option
        df_errors = pd.DataFrame()
        df_errors['Maturity'] = option_info_d['DTM']
        df_errors['Moneyness'] = option_info_d['Moneyness']
        df_errors['Erreurs moy. vs Methode 1'] = option_info_d['IV_method1'] - option_info_d['IV_NGARCH']
        df_errors['Erreurs moy. vs Methode 2'] = option_info_d['IV_method2'] - option_info_d['IV_NGARCH']

        # Diviser les données en intervalles de maturité et de moneyness
        maturity_bins = np.linspace(option_info_d['DTM'].min(), option_info_d['DTM'].max(), num=5)
        moneyness_bins = np.linspace(option_info_d['Moneyness'].min(), option_info_d['Moneyness'].max(), num=5)
        df_errors['Maturity Bin'] = pd.cut(df_errors['Maturity'], bins=maturity_bins)
        df_errors['Moneyness Bin'] = pd.cut(df_errors['Moneyness'], bins=moneyness_bins)

        # Calculer la moyenne et l'écart-type des erreurs pour chaque intervalle
        grouped = df_errors.groupby(['Maturity Bin', 'Moneyness Bin'])
        summ    = grouped[['Erreurs moy. vs Methode 1', 'Erreurs moy. vs Methode 2']].agg([np.mean])
        summ.reset_index(inplace=True)
        summ.columns = summ.columns.droplevel(level=1)
        
        summary.append(summ.dropna())
        
    return summary