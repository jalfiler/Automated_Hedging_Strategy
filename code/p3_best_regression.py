"""Seattle University, OMSBA 5062, P3: Best Regression FSS, Jomaica Lei

Classes:
Basket - sets up the give portfolio assigning weights, name, and portfolio of values.

Functions:
ols - Perform Ordinary Least Square regression.
regress_one_column - Perform simple linear regression using a single column of features.
regression - returns a residual sum of squares (RSS) and a set of weights to use in the surrogate basket.
best_regression_n - returns a portfolio of components and their weights that is the best (lowest RSS) 
                    proxy basket of a subset of size n of the company's full set of risks in the Basket class. 
best_regression_backtest - performs back testing of a proxy portfolio chosen by the best_regression_n method.
"""

import os
import csv
from datetime import datetime
from datetime import timedelta
import numpy as np
import math
import itertools
from p3_TimeSeries import TimeSeries, Fred, trim, USDCommodity, USDForex

DATA = '/Users/jomaicaalfiler/Desktop/MSBA/Python - 5062/Week 9/P3/'


def ols(X, y):
    "Perform Ordinary Least Square regression."
    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)
    Xty = np.dot(Xt, y)
    beta = np.linalg.solve(XtX, Xty)
    y_hat = np.dot(X, beta)
    residuals = y - y_hat
    rss = np.sum(residuals ** 2)
    return beta, sum(y), [rss]

def regress_one_column(features, response, column):
    """
    Perform simple linear regression using a single column of features.
    :param features:   feature vectors (n observations x m features)
    :param response:   response vector (n observations)
    :param column:     column number to pick from the feature vectors
    :return:           slope, intercept, RSS
    """
    feature_column = [feature[column] for feature in features]
    feature_column_with_intercept = [[1.0] * len(features), feature_column]
    feature_column_with_intercept = np.transpose(feature_column_with_intercept)
    slope, intercept, rss = ols(feature_column_with_intercept, response)
    return slope, intercept, rss

class Basket(object):
    "Sets up the give portfolio assigning weights, name, and portfolio of values."   
    def __init__(self, portfolio, weights, first=None, last=None):
        self.portfolio = portfolio
        self.weights = weights
        self.y_vec = []
        dates = portfolio[0].get_dates()
        for i in portfolio:
            dates = [n for n in i.get_dates() if n in dates]
        self.dates = dates

        y_vec = {}
        if first is None:
            first = min(self.dates)

        if last is None:
            last = max(self.dates)

        for i in self.dates:
            if i >= first and i <= last:
                num = 0
                for j in range(len(portfolio)):
                    num += portfolio[j].data[i] * weights[j]
                y_vec[i] = num
        self.y_vec1 = y_vec

    def regression(self, risk_list, first=None, last=None): 
        """Returns a residual sum of squares (RSS) and a set of weights to use in the surrogate basket.
        Also takes an optional start and end date for the data sampling. Returns the results as a
        dictionary
        
        >>> b = Basket([wti(), copper(), aluminum(), chy(), inr(), krw(), mxn(), myr()], [10485,      172,       1307,  30e6,  57e6, 1.3e6,  94e6, 1.4e9])
        >>> b.regression(['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr'])
        {'wti': 0.0004301969068451659, 'copper': 10484.999999492244, 'aluminum': 171.99999999316157, 'chy': 1307.0000000796922, 'inr': 29999999.999673966, 'krw': 56999999.98982785, 'mxn': 1299999.7358122845, 'myr': 94000000.00067061, 'intercept': 0.0004301969068451659, 'rss': 3.4330978605986246e+17, 'start': '1993-11-08', 'end': '2019-12-31'}
        >>> b.regression(['wti'])
        {'wti': 434678111.2261936, 'intercept': 434678111.2261936, 'rss': 1.1547810998748842e+39, 'start': '1993-11-08', 'end': '2019-12-31'}
        >>> b.regression(['aluminum'], first=datetime(2001,1,1), last=datetime(2001,12,31))
        {'aluminum': 384801223.4418913, 'intercept': 384801223.4418913, 'rss': 1.4677991717219846e+26, 'start': '2001-01-01', 'end': '2001-12-31'}
        >>> b.regression(['wti', 'copper', 'krw', 'chy'], last=datetime(2001,1,23))
        {'wti': 42598309.51623708, 'copper': -1895033.496455932, 'krw': 27372.393418383097, 'chy': 363875831384.3091, 'intercept': 42598309.51623708, 'rss': 1.645826311407339e+36, 'start': '1993-11-08', 'end': '2001-01-23'}
        """
        simple_dates = []
        self.risk_list = risk_list
        if first is None:
            first = min(self.dates)
        if last is None:
            last = max(self.dates)

        for i in self.dates:
            if i >= first and i <= last:
                simple_dates.append(i)

        self.y_vec = [self.y_vec1[k] for k in self.y_vec1 if k in simple_dates]
        regr1 = []
        b0 = 1 / math.sqrt(len(risk_list))
        for i in range(len(simple_dates)):
            regr1.append([b0] + [0] * len(risk_list))

        for i in range(len(risk_list)):
            for j in range(len(self.portfolio)):
                if risk_list[i] == self.portfolio[j].name:
                    vals = self.portfolio[j].get_values(simple_dates)
                    for k in range(len(simple_dates)):
                        regr1[k][i + 1] = vals[k] 

        results = ols(regr1, self.y_vec)

        beta = results[0]
        beta = [beta] if isinstance(beta, float) else beta

        rss = sum([(y - y_hat) ** 2 for y, y_hat in zip(self.y_vec, results[2])])

        dict1 = {}
        for i in range(len(risk_list)):
            dict1[risk_list[i]] = beta[i] if i < len(beta) else None

        dict1['intercept'] = beta[0]
        dict1['rss'] = rss
        dict1['start'] = first.strftime('%Y-%m-%d')
        dict1['end'] = last.strftime('%Y-%m-%d')
        return dict1
    
    def best_regression_n(self, n, first=None, last=None): 
        """Returns a portfolio of components and their weights that is the best (lowest RSS) 
        proxy basket of a subset of size n of the company's full set of risks in the Basket class. 
        Also returns the RSS for the returned proxy portfolio.
            
        >>> b.best_regression_n(4, last=datetime(2004,7,1))
       {'inr': 10585749.828259017, 'krw': 77087677.03360182, 'mxn': 1281874354.767722, 'myr': 91618011.19643155, 'intercept': 10585749.828259017, 'rss': 4.113253899331552e+28, 'start': '1993-11-08', 'end': '2004-07-01'}
        """
        if first == None:
            first = min(self.dates)
        if last == None:
            last = max(self.dates)
        option_list = self.portfolio
        best_rss = None
        big_list = [i.name for i in self.portfolio]
        for i in itertools.combinations(big_list, n):
            a = self.regression(i, first, last)
            if best_rss == None or abs(a['rss']) < best_rss:
                weighted_portfolio = a
        return weighted_portfolio

    def best_regression_backtest(self, n, split_date): 
        """Performs back testing of a proxy portfolio chosen by the best_regression_n method.
        (Training set is up to split_date, hold-out set is after split_date.) Returns the standard
        deviation of the value of the complete Basket along with the suggested hedges from
        the training set held during the hold-out period.
        
        >>> b.best_regression_backtest(4, datetime(2001, 1, 1)) 
        (38008892.176508605,
        {'inr': 9716382.415181508,
        'krw': 93022619.87910804,
        'mxn': 1078451721.797786,
        'myr': 91602010.30532515,
        'intercept': 9716382.415181508,
        'rss': 2.1184185369471096e+28,
        'start': '1993-11-08',
        'end': '2001-01-01',
        'wti': 0,
        'copper': 0,
        'aluminum': 0,
        'chy': 0})
        """
        listera = ['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr']
        train = b.best_regression_n(n, last=split_date)
        for i in listera:
            if i in train:
                pass
            if i not in train:
                train[i] = 0  
        c = Basket([wti(), copper(), aluminum(), chy(), inr(), krw(), mxn(), myr()],
                   [(10485 - train['wti']), (172 - train['copper']), (1307 - train['aluminum']),
                    (30e6 - train['chy']), (57e6 - train['inr']), (1.3e6 - train['krw']),
                    (94e6 - train['mxn']),
                    (1.4e9 - train['myr'])])
        hold_out = c.regression(['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr'],
                                first=split_date)  
        mean = (sum(c.y_vec) / len(c.y_vec))  
        var = sum([((x - mean) ** 2) for x in c.y_vec]) / len(c.y_vec)
        SD = var ** 0.5
        return SD, train  


class myr(USDForex):
    def __init__(self):
        super().__init__('DEXMAUS')  


class mxn(USDForex):
    def __init__(self):
        super().__init__('DEXMXUS')


class krw(USDForex):
    def __init__(self):
        super().__init__('DEXKOUS')


class inr(USDForex):
    def __init__(self):
        super().__init__('DEXINUS')


class chy(USDForex):
    def __init__(self):
        super().__init__('DEXCHUS')


class wti(USDCommodity):
    def __init__(self):
        super().__init__('wti', 'OIL Commodity', 'USD', 'DCOILWTICO')  


class copper(USDCommodity):
    def __init__(self):
        super().__init__('copper', 'COPPER Commodity', 'USD', 'PCOPPUSDM')  


class aluminum(USDCommodity):
    def __init__(self):
        super().__init__('aluminum', 'Aluminum Commodity', 'USD', 'PALUMUSDM') 


if __name__ == '__main__':
    b = Basket([wti(), copper(), aluminum(), chy(), inr(), krw(), mxn(), myr()],
               [10485, 172, 1307, 30e6, 57e6, 1.3e6, 94e6, 1.4e9])
    print(b.regression(['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr']))  
    print(b.regression(['wti']))
    print(b.regression(['aluminum'], first=datetime(2001, 1, 1), last=datetime(2001, 12, 31)))
    print(b.regression(['wti', 'copper', 'krw', 'chy'], last=datetime(2001, 1, 23)))
    print(b.best_regression_n(4, last=datetime(2004, 7, 1))) 
    print(b.best_regression_backtest(4, datetime(2001, 1, 1)))  
