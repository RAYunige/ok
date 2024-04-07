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
from .base import OptionPricingModel


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
