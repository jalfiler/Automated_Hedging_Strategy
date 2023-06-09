"""Seattle University, OMSBA 5062, P3: Time Series, Jomaica Lei

Classes:
TimeSeries - class to hold a date/value series of data
Difference - a time series that is the difference between two other time series
Fred - a time series that is based on a csv file downloaded from fred.stlouis.org
dgs3mo - the 3-month treasury series from FRED
dgs10 - the 10-year treasury series from FRED
Bundesbank - reads in gold price data
gold_spot - creates and assigns gold values to a dicitionary
lag - creates a dictionary of profit on gold according a given holding period.
returns - calculates the ratio of profit for sold gold given a time period.
trim - helps split and format dates appropriately.
USDForex - helps convert foriegn currencies to proper dollar ratio for comparison purposes and
evaluation
USDCommodity - selects the proper values for the dates given and insert them into all dates of
year, referencing last value when missing value is found in data set.
"""

import os
import csv
from datetime import datetime
from scipy.stats import pearsonr
from datetime import timedelta
import numpy as np
import math
import itertools

DATA = '/Users/jomaicaalfiler/Desktop/MSBA/Python - 5062/Week 9/P3/'


class TimeSeries(object):
    """Holds a date/value series of data"""
    def __init__(self, name, title=None, unit=None, data=None):
        self.name = name
        self.title = title if title is not None else name
        self.unit = unit
        self.data = data if data is not None else {}
        self.first_date = None if len(self.data) == 0 else min(self.data)
        self.last_date = None if len(self.data) == 0 else max(self.data)

    def get_dates(self, candidates=None, start=None, end=None):
        """Get the dates where this series has values
        ts.get_dates() - gets all dates where ts has values
        ts.get_dates(start=d1,end=d2) - get all valid dates, d, 
                                        where d1<=d<=d2
        ts.get_dates(candidates=dates) - get all valid dates, d, 
                                         for d in dates
        :param candidates: if set, start and end are ignored, and 
                           returns the subset of dates within 
                           candidates that for which this series has data
        :param start:      minimum starting date of returned dates, 
                           defaults to beginning of this series
        :param end:        max ending date of returned dates, 
                           defaults to end of this series
        :return:           a list of dates in order for which
                           this series has values and that satisfy
                           the parameters' conditions
        """
        if candidates is not None:
            return [date for date in candidates if date in self.data]  
        if start is None:
            start = self.first_date
        if end is None:
            end = self.last_date
        return [date for date in sorted(self.data) if start <= date <= end]  
    
    def get_values(self, dates):
        """Get the values for the specified dates.
        :param dates:      list of dates to get values for
        :return:           the values for the given dates in the same order
        :raises:           KeyError  if any requested date has no value
        """
        ret = []
        for d in dates:
            try:
                ret.append(self.data[d])
            except KeyError:
                pass
        return ret
       
    def __sub__(self, other):
        """create a difference time series"""
        return Difference(self, other)
    
    def correlation(self, other):
        """Calculate the Pearson correlation coefficient between this series
        and another on all days when they both have values.
        Uses scipy.stats.pearsonr to calculate it.
        """
        corr_list1 = []
        corr_list2 = []
        for k in self.data:
            if k in other.data:
                corr_list1.append(self.data[k])
                corr_list2.append(other.data[k])  
        return pearsonr(corr_list1, corr_list2)


class trim(TimeSeries):
    def __init__(self, underlier, start=None, end=None):
        name = 'trim(' + underlier.name + ', ' + \
               str(start) + ', ' + str(end) + ')'
        dates = underlier.get_dates(start=start, end=end)
        data = {d: v for d, v in zip(dates, underlier.get_values(dates))}
        super().__init__(name, unit=underlier.unit, data=data)

    
class Fred(TimeSeries):
    """A time series that is based on a csv file downloaded from
    fred.stlouis.org
    """

    def __init__(self, name, title=None, unit=None, data_column=None):
        """Opens and reads the csv file in DATA/name.csv"""
        super().__init__(name.lower(), title, unit)
        filename = os.path.join(DATA, name + '.csv')
        if data_column is None:
            data_column = name
        with open(filename) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    value = float(row[data_column])
                except ValueError:
                    continue
                date = datetime.strptime(row['DATE'], "%Y-%m-%d")
                self.data[date] = value
        self.first_date = min(self.data)
        self.last_date = max(self.data)

        
class dgs3mo(Fred):
    """The 3-month treasury series from FRED"""
    
    def __init__(self):
        super().__init__('DGS3MO', '3-Month Treasury', 'percent')
        

class dgs10(Fred):
    """The 10-year treasury series from FRED"""
    
    def __init__(self):
        super().__init__('DGS10', '10-Year Treasury', 'percent')


class Difference(TimeSeries):
    """a time series that is the difference between two other time series"""

    def __init__(self, a, b):
        super().__init__(a.name + '-' + b.name, unit=a.unit)
        self.data = {d: (a.data[d] - b.data[d]) for d in a.data if d in b.data}
        self.first_date = min(self.data)
        self.last_date = max(self.data)


class Bundesbank(TimeSeries):
    """A time series that is based on a csv file downloaded from
    Deutsche Bundesbank, www.bundesbank.de which reads in gold price data
    """

    def __init__(self, name=None, filename=None):
        self.data = {}
        self.title = None
        self.unit = None
        filename = DATA + filename + '.csv'
        with open(filename) as csv_file:
            reader = csv.reader(csv_file)
            for row_number, row in enumerate(reader):
                if row_number == 1:
                    title = row[1]
                elif row_number == 2:
                    unit = row[1]
                else:
                    try:
                        datetime.strptime(row[0], '%Y-%m-%d')
                        break
                    except:
                        pass
        super().__init__(name, title, unit)
        with open(filename) as csv_file:  
            reader = csv.reader(csv_file)
            for skip in range(row_number):  
                next(reader)
            for row in reader:
                try:
                    self.data[datetime.strptime(row[0], '%Y-%m-%d')] = float(row[1])
                except ValueError:
                    pass
            self.first_date = min(self.data)
            self.last_date = max(self.data)


class gold_spot(Bundesbank):
    """Spot gold prices from London morning fix
    >>> gold = gold_spot()
    >>> gold.first_date
    datetime.datetime(1968, 4, 1, 0, 0)
    >>> gold.title, gold.unit
    ('Price of gold in London / morning fixing / 1 ounce of fine gold = USD ...', 'USD')
    """

    def __init__(self):
        """Bundesbank's constructor wants time series name and filename in
           the DATA directory (automatically appending '.csv' to it).
           Bundesbank will automatically pull title and unit from within
           the file.
        """
        super().__init__(self.__class__.__name__, 'BBEX3.D.XAU.USD.EA.AC.C04')


class USDForex(Fred):
    def __init__(self, data_column):
        super().__init__('fx_' + self.__class__.__name__, unit='USD', data_column=data_column)
        self.data = {d: 1 / self.data[d] for d in self.data}
        self.name = self.__class__.__name__


class USDCommodity(Fred):
    def __init__(self, name, title=None, unit=None, data_column=None):
        super().__init__('cmdty_' + name, unit='USD', data_column=data_column)
        data = {}
        d = self.first_date
        last_value = self.data[d]
        while d <= self.last_date:
            if d in self.data:
                last_value = self.data[d]  
            if d.weekday() < 5:
                try:
                    last_value = self.data[d]  
                    data[d] = last_value  
                except KeyError:
                    data[d] = last_value
            d += timedelta(days=1)
        self.data = data
        self.name = name


class lag(TimeSeries):
    """Time series that is a right-shifted copy of another.
    Shifting is done across a given number of data points, ignoring actual
    time intervals (e.g., a lag of one on a weekday-only series goes from
    Thursday to Friday and Friday to Monday, ignoring the weekend where
    there are no points)
    >>> lagger = lag(TimeSeries('a', data={datetime(2018,1,i):i for i in range(10,20,3)}), 2)
    >>> lagger.get_dates()
    [datetime.datetime(2018, 1, 16, 0, 0), datetime.datetime(2018, 1, 19, 0, 0)]
    >>> lagger.get_values(lagger.get_dates())
    [10, 13]
    """  

    def __init__(self, underlier, total_hold):
        super().__init__(underlier.name, data=underlier.data)
        self.first_date = min(self.data)
        self.last_date = max(self.data)
        self.data = {}
        current_dates = underlier.get_dates()
        values_lag = underlier.get_values(current_dates)
        dates_lag = current_dates[total_hold:]
        for i in range(len(dates_lag)):
            self.data[dates_lag[i]] = values_lag[i]


class returns(TimeSeries):
    """Time series that is profit ratio of buying the asset at time t[0]
    and selling it at time t[n]. Value is (t[n]-t[0])/t[0].
    >>> inv = returns(TimeSeries('a', data={datetime(2018,1,i):i for i in range(10,20,3)}), 1)
    >>> inv.get_dates()
    [datetime.datetime(2018, 1, 13, 0, 0), datetime.datetime(2018, 1, 16, 0, 0), datetime.datetime(2018, 1, 19, 0, 0)]
    >>> inv.get_values(inv.get_dates()) == [3/10, 3/13, 3/16]
    True
    """  

    def __init__(self, gold, total_hold):
        super().__init__(gold.name, data=gold.data)
        self.first_date = min(self.data)
        self.last_date = max(self.data)
        lagged = lag(gold, total_hold)
        self.data = {i: ((self.data[i] - lagged.data[i]) / lagged.data[i]) for i in lagged.get_dates()}  

