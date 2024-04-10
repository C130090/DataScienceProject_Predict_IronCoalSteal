import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

mfg = pd.read_csv("C:/Users/tanyixin/Desktop/Commodity Price Forecasting/mfgChina - Weekly.csv").fillna(' ')

## Convert all values from str to float ##
for col in range(mfg.shape[1]-1):
    for row in range(mfg.shape[0]):
        mfg[mfg.columns[col+1]][row] = np.float64(mfg[mfg.columns[col+1]][row])
            

## Data Feeding ##
mfg = mfg.rename(columns={mfg.columns[0]: 'Date'})


## Retrieve start of week (Mon) based on given date ##
mfg['start_week'] = pd.to_datetime(mfg['Date'])
mfg['start_week'] = mfg['start_week'].dt.to_period('W').apply(lambda r: r.start_time).dt.date


## Group data to weekly data (Use median for price data) ##
mfg_weekly = mfg.groupby(['start_week'], as_index=False).median()


## Unique start_week ##
mfg_weekly = mfg_weekly.drop_duplicates(['start_week']).reset_index(drop=True)


## Generate CSV file ##
mfg_weekly.to_csv('C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/mfg_weekly.csv', index=False)
