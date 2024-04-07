# Imports
from enum import Enum
from abc import ABC, abstractclassmethod

class OPTION_TYPE(Enum):
    CALL_OPTION = 'Call Option'
    PUT_OPTION = 'Put Option'

class OptionPricingModel(ABC):
    """Abstract class defining interface for option pricing models."""

    def calculate_option_price(self, option_type):
        """Calculates call/put option price according to the specified parameter."""
        if option_type == OPTION_TYPE.CALL_OPTION.value:
            return self._calculate_call_option_price()
        elif option_type == OPTION_TYPE.PUT_OPTION.value:
            return self._calculate_put_option_price()
        else:
            return -1

    @abstractclassmethod
    def _calculate_call_option_price(self):
        """Calculates option price for call option."""
        pass

    @abstractclassmethod
    def _calculate_put_option_price(self):
        """Calculates option price for put option."""
        pass
# Imports
import numpy as np
from scipy.stats import norm 


class BlackScholesModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using Black-Scholes Formula.

    Call/Put option price is calculated with following assumptions:
    - European option can be exercised only on maturity date.
    - Underlying stock does not pay divident during option's lifetime.  
    - The risk free rate and volatility are constant.
    - Efficient Market Hypothesis - market movements cannot be predicted.
    - Lognormal distribution of underlying returns.
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma):
        """
        Initializes variables used in Black-Scholes formula .

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        """
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma

    def _calculate_call_option_price(self): 
        """
        Calculates price for call option according to the formula.        
        Formula: S*N(d1) - PresentValue(K)*N(d2)
        """
        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be exercised)     
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of the option)
        # d2 = (d1 - (sigma * sqrt(T)))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        return (self.S * norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0))
    

    def _calculate_put_option_price(self): 
        """
        Calculates price for put option according to the formula.        
        Formula: PresentValue(K)*N(-d2) - S*N(-d1)
        """  
        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be exercised)    
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of the option)
        # d2 = (d1 - (sigma * sqrt(T)))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2, 0.0, 1.0) - self.S * norm.cdf(-d1, 0.0, 1.0))
# Third party imports
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt



class MonteCarloPricing(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using Monte Carlo Simulation.
    We simulate underlying asset price on expiry date using random stochastic process - Brownian motion.
    For the simulation generated prices at maturity, we calculate and sum up their payoffs, average them and discount the final value.
    That value represents option price
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations):
        """
        Initializes variables used in Black-Scholes formula .

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        number_of_simulations: number of potential random underlying price movements 
        """
        # Parameters for Brownian process
        self.S_0 = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma 

        # Parameters for simulation
        self.N = number_of_simulations
        self.num_of_steps = days_to_maturity
        self.dt = self.T / self.num_of_steps

    def simulate_prices(self):
        """
        Simulating price movement of underlying prices using Brownian random process.
        Saving random results.
        """
        np.random.seed(20)
        self.simulation_results = None

        # Initializing price movements for simulation: rows as time index and columns as different random price movements.
        S = np.zeros((self.num_of_steps, self.N))        
        # Starting value for all price movements is the current spot price
        S[0] = self.S_0

        for t in range(1, self.num_of_steps):
            # Random values to simulate Brownian motion (Gaussian distibution)
            Z = np.random.standard_normal(self.N)
            # Updating prices for next point in time 
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + (self.sigma * np.sqrt(self.dt) * Z))

        self.simulation_results_S = S

    def _calculate_call_option_price(self): 
        """
        Call option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Call option payoff (it's exercised only if the price at expiry date is higher than a strike price): max(S_t - K, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.simulation_results_S[-1] - self.K, 0))
    

    def _calculate_put_option_price(self): 
        """
        Put option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Put option payoff (it's exercised only if the price at expiry date is lower than a strike price): max(K - S_t, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.K - self.simulation_results_S[-1], 0))
       

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        plt.figure(figsize=(12,8))
        plt.plot(self.simulation_results_S[:,0:num_of_movements])
        plt.axhline(self.K, c='k', xmin=0, xmax=self.num_of_steps, label='Strike Price')
        plt.xlim([0, self.num_of_steps])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Days in future')
        plt.title(f'First {num_of_movements}/{self.N} Random Price Movements')
        plt.legend(loc='best')
        plt.show()
# Imports
import numpy as np
from scipy.stats import norm 



class BinomialTreeModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using BOPM (Binomial Option Pricing Model).
    It caclulates option prices in discrete time (lattice based), in specified number of time points between date of valuation and exercise date.
    This pricing model has three steps:
    - Price tree generation
    - Calculation of option value at each final node 
    - Sequential calculation of the option value at each preceding node
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps):
        """
        Initializes variables used in Black-Scholes formula .

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        number_of_time_steps: number of time periods between the valuation date and exercise date
        """
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.number_of_time_steps = number_of_time_steps

    def _calculate_call_option_price(self): 
        """Calculates price for call option according to the Binomial formula."""
        # Delta t, up and down factors
        dT = self.T / self.number_of_time_steps                             
        u = np.exp(self.sigma * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(self.number_of_time_steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(self.S * u**j * d**(self.number_of_time_steps - j)) for j in range(self.number_of_time_steps + 1)])

        a = np.exp(self.r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   
        V[:] = np.maximum(S_T - self.K, 0.0)
    
        # Overriding option price 
        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1])
        return V[0]

    def _calculate_put_option_price(self): 
        """Calculates price for put option according to the Binomial formula."""  
        # Delta t, up and down factors
        dT = self.T / self.number_of_time_steps                             
        u = np.exp(self.sigma * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(self.number_of_time_steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(self.S * u**j * d**(self.number_of_time_steps - j)) for j in range(self.number_of_time_steps + 1)])

        a = np.exp(self.r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   
        V[:] = np.maximum(self.K - S_T, 0.0)
    
        # Overriding option price 
        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1])
        return V[0]
# Library imports
import datetime
import requests_cache
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
import yfinance as yf

class Ticker:
    """Class for fetcing data from yahoo finance."""
    
    @staticmethod
    def get_historical_data(ticker, start_date = None, end_date = None, cache_data = True, cache_days = 1):
        """
        Fetches stock data from yahoo finance. Request is by default cashed in sqlite db for 1 day.
        
        Params:
        ticker: ticker symbol
        start_date: start date for getting historical data
        end_date: end date for getting historical data
        cache_date: flag for caching fetched data into sqlite db
        cache_days: number of days data will stay in cache 
        """
        try:
            # initializing sqlite for caching yahoo finance requests
            expire_after = datetime.timedelta(days = 1)
            session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = expire_after)

            # Adding headers to session
            session.headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0', 
                               'Accept': 'application/json;charset=utf-8'
                              }
            
            if(start_date is not None and end_date is not None):
                data = yf.download(ticker, start = start_date, end = end_date)
            else:
                data = yf.download(ticker)

            if data is None:
                return None
            return data
        
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def get_columns(data):
        """
        Gets dataframe columns from previously fetched stock data.
        
        Params:
        data: dataframe representing fetched data
        """
        if data is None:
            return None
        return [column for column in data.columns]

    @staticmethod
    def get_last_price(data, column_name):
        """
        Returns last available price for specified column from already fetched data.
        
        Params:
        data: dataframe representing fetched data
        column_name: name of the column in dataframe
        """
        if data is None or column_name is None:
            return None
        if column_name not in Ticker.get_columns(data):
            return None
        return data[column_name].iloc[len(data) - 1]


    @staticmethod
    def plot_data(data, ticker, column_name):
        """
        Plots specified column values from dataframe.
        
        Params:
        data: dataframe representing fetched data
        column_name: name of the column in dataframe
        """
        try:
            if data is None:
                return
            data[column_name].plot()
            plt.ylabel(f'{column_name}')
            plt.xlabel('Date')
            plt.title(f'Historical data for {ticker} - {column_name}')
            plt.legend(loc = 'best')
            plt.show()

        except Exception as e:
            print(e)
            return
"""
Script testing functionalities of option_pricing package:
- Testing stock data fetching from Yahoo Finance using pandas-datareader
- Testing Black-Scholes option pricing model   
- Testing Binomial option pricing model   
- Testing Monte Carlo Simulation for option pricing   
"""


# Fetching the prices from yahoo finance
data = Ticker.get_historical_data('TSLA')
print(Ticker.get_columns(data))
print(Ticker.get_last_price(data, 'Adj Close'))
Ticker.plot_data(data, 'TSLA', 'Adj Close')

# Black-Scholes model testing
BSM = BlackScholesModel(100, 100, 365, 0.1, 0.2)
print(BSM.calculate_option_price('Call Option'))
print(BSM.calculate_option_price('Put Option'))

# Binomial model testing
BOPM = BinomialTreeModel(100, 100, 365, 0.1, 0.2, 15000)
print(BOPM.calculate_option_price('Call Option'))
print(BOPM.calculate_option_price('Put Option'))

# Monte Carlo simulation testing
MC = MonteCarloPricing(100, 100, 365, 0.1, 0.2, 10000)
MC.simulate_prices()
print(MC.calculate_option_price('Call Option'))
print(MC.calculate_option_price('Put Option'))
MC.plot_simulation_results(20)
# Standard python imports
from enum import Enum
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Third party imports
import plotly.graph_objs as go


app = dash.Dash(__name__)

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'

# Assuming get_historical_data and other relevant functions are defined in your local package
# and modified to work with Dash/Plotly where necessary

app.layout = html.Div([
    html.H1('Option Pricing'),
    dcc.RadioItems(
        id='pricing-method',
        options=[{'label': model.value, 'value': model.name} for model in OPTION_PRICING_MODEL],
        value='BLACK_SCHOLES'
    ),
    html.Div(id='model-params'),
    html.Button('Calculate option price', id='calculate-button'),
    html.Div(id='option-price-display')
])

@app.callback(
    Output('model-params', 'children'),
    Input('pricing-method', 'value')
)
def update_model_params(pricing_method):
    return [
        dcc.Input(id='ticker-symbol', type='text', value='AAPL', placeholder='Ticker symbol'),
        dcc.Input(id='strike-price', type='number', value=0, placeholder='Strike price'),
        dcc.Slider(id='risk-free-rate', min=0, max=100, step=1, value=10, marks={i: str(i) for i in range(0, 101, 10)}),
        dcc.Slider(id='sigma', min=0, max=100, step=1, value=20, marks={i: str(i) for i in range(0, 101, 10)}),
        dcc.DatePickerSingle(
            id='exercise-date',
            min_date_allowed=datetime.today() + timedelta(days=1),
            date=datetime.today() + timedelta(days=365)
        ),
        # Add any additional parameters specific to the pricing model here
    ]

@app.callback(
    Output('option-price-display', 'children'),
    [Input('calculate-button', 'n_clicks')],
    [
        Input('pricing-method', 'value'),
        Input('ticker-symbol', 'value'),
        Input('strike-price', 'value'),
        Input('risk-free-rate', 'value'),
        Input('sigma', 'value'),
        Input('exercise-date', 'date'),
        # Add any additional parameters if needed
    ]
)
def calculate_option_price(n_clicks, pricing_method, ticker, strike_price, risk_free_rate, sigma, exercise_date):
    if n_clicks is None:
        return "Please enter the parameters and click the button to calculate the option price."
    else:
        # Here you would include your logic to use the selected option pricing model
        # and return the calculated option price(s).
        # This is just a placeholder return statement.
        return f"Calculated option prices for {ticker} will be displayed here."

if __name__ == '__main__':
    app.run_server(debug=True)
