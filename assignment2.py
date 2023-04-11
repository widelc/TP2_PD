import datetime as dt
import os
import sys
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

    date_ref = min([min(y.date), min(time_t)])
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
        #print(ng)
        #print('Persistence:', ng.persistence())
        #print('Unconditional volatility:', np.sqrt(ng.uncond_var()/Dt))
        
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
        theta     = [self.lmbda, self.omega, self.alpha, self.beta, self.gamma]
        h_t, eps  = f_ht_NGARCH(theta, self.log_xret) 
        return self.omega + self.alpha * h_t[-1] * ((eps[-1] - self.gamma) ** 2) + self.beta * h_t[-1]
    
    def Q_predict_h(self):
        theta     = [self.lmbda, self.omega, self.alpha, self.beta, self.gamma]
        h_t, eps  = f_ht_NGARCH(theta, self.log_xret)
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
        
        cp = [1 if cond else -1 for cond in [is_call]][0]

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

    P.expected_h = np.mean(P.h,axis=1)
    Q.expected_h = np.mean(Q.h,axis=1)

    y_prefix = ''
    if annualized:
        days_in_year = 1 / horizon[0]
        P.expected_h = P.expected_h * days_in_year
        Q.expected_h = Q.expected_h * days_in_year
        y_prefix     = 'Annualized '

    fig, ax = plt.subplots()
    ax.plot(horizon, P.expected_h, label='P Var forecasts (sim)')
    ax.plot(horizon, Q.expected_h, label='Q Var forecasts (sim)')
    ax.legend(loc='upper right')   
    ax.set_xlabel('Years to Maturity')
    ax.set_ylabel(y_prefix + 'Variance')
    ax.set_title('Simulated Conditionnal Variance Forecast')
    
    return ax


def f_ht_NGARCH(theta, log_xreturns):

    # Extract data from ng
    lmbda = theta[0]    
    omega = theta[1]
    alpha = theta[2]
    beta  = theta[3]
    gamma = theta[4]

    # Get rid of the NA at the beggining
    log_xreturns = log_xreturns[1:]
    
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

def f_nll_NGARCH(theta, log_xreturns):

    h, eps = f_ht_NGARCH(theta, log_xreturns)
    nll    = -0.5 * np.sum(np.log(2 * pi * h) + eps ** 2)

    return -nll

def f_NGARCH(ng):

    # Initial guess
    theta_0 = [ng.lmbda, ng.omega, ng.alpha, ng.beta, ng.gamma]

    # Optimization (and ignore warnings)
    warnings.simplefilter('ignore')
    bounds = [(None, None), (1e-8, None), (1e-8, None), (1e-8, None), (1e-8, None)]
    opt    = minimize(fun    = f_nll_NGARCH, 
                      x0     = theta_0, 
                      args   = ng.log_xret,
                      method = 'Nelder-Mead',  
                      bounds = bounds,
                      options= {'maxiter': 5000})
    warnings.resetwarnings()
 
    # Update parameters value in ng
    param = opt.x
    ng.lmbda = param[0]
    ng.omega = param[1]
    ng.alpha = param[2]
    ng.beta  = param[3]
    ng.gamma = param[4]

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
    option_info['DTM'] = np.array(option_info['exdate'], dtype='datetime64') - np.array(option_info['date'], dtype='datetime64')
    option_info['DTM'] = option_info['DTM'].dt.days
    return option_info


def f_describe_table(option_info):

    # Drop rows with missing IV
    option_info.dropna(subset=['impl_volatility'], how='any', inplace=True)

    # Create DTM column
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

    keep_col    = ['date','exdate','cp_flag', 'strike_price','mean_bidask','impl_volatility','delta','gamma','vega','theta','DTM']
    option_info['date']   =   pd.to_datetime(option_info['date'])
    option_info['exdate'] = pd.to_datetime(option_info['exdate'])
    option_info['mean_bidask'] = (option_info['best_bid'] + option_info['best_offer']) / 2
    option_info = option_info.drop_duplicates(subset=['date', 'exdate', 'cp_flag', 'strike_price'])

    return option_info[keep_col]

def f_add_Q3_info(option_info, days_in_year):

    # Ajouter les prix et les dividende à option_info
    option_info['S_t'] = get_price(option_info.date)
    option_info['y_t'] = get_dividend_rate(option_info.date)

    # Ajouter les taux sans risque à option_info
    date_unique = np.unique(option_info.date)
    rf = [get_risk_free_rate(date, option_info[option_info.date == date].DTM) for date in date_unique]
    option_info['r_f'] = np.concatenate(([arr for arr in rf]))

    # Ajouer le moneyness K/F à option_info
    K = option_info.strike_price / 1000
    F = option_info.S_t * np.exp((option_info.r_f - option_info.y_t) * option_info.DTM / days_in_year)
    option_info['F'] = F
    option_info['K/F'] = K / F

    return option_info

def f_F_CBOE(option_info):

    option_info['F_CBOE'] = pd.NA
    DTM_unique  = np.unique(option_info.DTM)
    date_unique = np.unique(option_info.date)

    for dtm in DTM_unique :

        for date in date_unique:
            
            #date = np.datetime64(date)
            option_DTM_i  = option_info[option_info.DTM == dtm]
            option_DTM_ti = option_DTM_i[option_DTM_i.date == date].sort_values(['cp_flag', 'K/F'])

            if not option_DTM_ti.empty:

                call_OTM   = option_DTM_ti[option_DTM_ti.cp_flag == 'C'].loc[option_DTM_ti['K/F'] > 1]
                put_OTM    = option_DTM_ti[option_DTM_ti.cp_flag == 'P'].loc[option_DTM_ti['K/F'] < 1]
                option_OTM = pd.concat([put_OTM , call_OTM])

                forward_price = interp1d(option_OTM['K/F'], option_OTM["F"], kind='linear')
                forward    = forward_price(1) 

                

                option_info.loc[option_DTM_ti.index, 'F_CBOE'] = forward
    
    return option_info