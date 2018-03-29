
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
# import random
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# http://www.learndatasci.com/python-finance-part-2-intro-quantitative-trading-strategies/

# In[2]:



# np.random.seed(1)
# ser = pd.Series(np.random.randn(100))
# ser = pd.Series([1,1,3,np.nan], index = ['a','b','c','d'])
# ser3 = pd.Series([3,4,2,5-g], index = ['b','d','c','a'])
# ser2 = pd.Series({'a':1, 'b':2})
# ser2
# len(ser)
# ser.value_counts()
# ser + ser3
# pd.DataFrame(np.array([[10,11], [5,6]]))
# df1 = pd.DataFrame([ pd.Series(np.arange(10,15)),
#                              pd.Series(np.arange(15,20))])
# df1
# df2 = pd.DataFrame( np.array([[10,11], [21,22]]), columns = ['a','b'], index = ['A', 'B'])
# df2

# s1 = pd.Series(np.arange(1,6,1))
# s2 = pd.Series(np.arange(6,11,1))
# s3 = pd.Series(np.arange(22,24), index = [3,4])
# df3 = pd.DataFrame({'C1': s1, 'C2': s2, 'C3':s3})
# df3


# This is to download data from google 
# Reference for exercise http://www.learndatasci.com/python-finance-part-yahoo-finance-api-pandas-matplotlib/

# In[5]:


tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
data_source = 'google'
start_date = '2010-01-04'
end_date = '2017-12-31'
panel_data = data.DataReader(tickers, data_source, start_date, end_date)


# In[6]:


import pickle
with open('panel_data.pickle', 'wb', -1) as handle:
    pickle.dump(panel_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('panel_data.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)


# In[7]:


print(panel_data.keys())
close = panel_data.loc['Close']
volume = panel_data.loc['Volume']
all_weekdays = pd.date_range(start = start_date, end= end_date, freq='B')
close = close.reindex(all_weekdays)
volume = volume.reindex(all_weekdays)
close.head(10)
volume.head(10)
all_weekdays
close.isnull().any()
close = close.dropna()
volume = volume.dropna()
close.isnull().any()


# In[8]:


aapl = close.loc[:, 'AAPL']
aapl_vol = volume.loc[:, 'AAPL']
short_rolling_aapl = aapl.rolling(window=20).mean()
long_rolling_aapl = aapl.rolling(window=100).mean()
vlong_rolling_aapl = aapl.rolling(window=200).mean()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.grid(color = 'red',linewidth=0.5)
ax.plot(aapl.index, aapl, label='AAPL')
ax.plot(aapl_vol.index, aapl_vol/1000000, label='Volume in million shares')
ax.plot(short_rolling_aapl.index, short_rolling_aapl, label='20 days rolling')
ax.plot(long_rolling_aapl.index, long_rolling_aapl, label='100 days rolling')
ax.plot(vlong_rolling_aapl.index, vlong_rolling_aapl, label='200 days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Closing price ($)')
ax.legend()
fig.savefig('AAPL1.png')
plt.show()


# As a general guideline, if the price is above a moving average the trend is up. If the price is below a moving average the trend is down. Moving averages can have different lengths though, so one may indicate an uptrend while another indicates a downtrend. They act as resistance if above the trading price and as floor if below the trading price.

# In[17]:


data = pd.read_pickle('./panel_data.pickle')
close = data.loc['Close']
print(close.head(10))
short_rolling = close.rolling(window=20).mean()
short_rolling.head(30) # obv first 20 will be NaN
long_rolling = close.rolling(window=100).mean()
long_rolling.tail()


# In[11]:


returns = close.pct_change(1)
returns.head()


# In[12]:


log_returns = np.log(close).diff()
log_returns.head()


# In[14]:


fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(2,1,1)
ax.grid()
for c in log_returns:
    ax.plot(log_returns.index, log_returns[c].cumsum(), label = str(c))
ax.set_ylabel('Cumulative log returns')
ax.legend(loc='best')

ax = fig.add_subplot(2,1,2)
for c in log_returns:
    ax.plot(log_returns.index, 100*(np.exp(log_returns[c].cumsum()) - 1), label=str(c))
ax.set_ylabel('Total relative returns (%)')
ax.legend(loc='best')
ax.grid()
fig.savefig('Returns.png')
plt.show()


# In[9]:


aapl = close.loc[:, 'AAPL']
aapl_vol = volume.loc[:, 'AAPL']
short_rolling_aapl = aapl.rolling(window=20).mean()
long_rolling_aapl = aapl.rolling(window=100).mean()
vlong_rolling_aapl = aapl.rolling(window=200).mean()
short_std = aapl.rolling(window=20).std()
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1,1,1)
ax.grid(color = 'red',linewidth=0.5)
ax.plot(aapl.index, aapl, label='AAPL')
ax.plot(aapl_vol.index, aapl_vol/1000000, label='Volume in million shares')
ax.plot(short_rolling_aapl.index, short_rolling_aapl, label='20 days rolling')
ax.plot(long_rolling_aapl.index, long_rolling_aapl, label='100 days rolling')
ax.plot(vlong_rolling_aapl.index, vlong_rolling_aapl, label='200 days rolling')
plt.fill_between(short_rolling_aapl.index, short_rolling_aapl-2*short_std,short_rolling_aapl+2*short_std, color='b', alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Closing price ($)')
ax.legend()
fig.savefig('AAPL.png', format='png', dpi=300)
plt.show()


# In[11]:


sp500 = pd.read_csv('sp500.csv')
sp500.describe()

