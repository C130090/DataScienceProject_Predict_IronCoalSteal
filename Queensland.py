import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

queensland = pd.read_csv("C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Clean version binary choice Queensland Vessel - Weekly.csv").fillna(' ')


## Data Feeding ##
queensland = queensland.rename(columns={queensland.columns[0]: 'Date'})


## Retrieve start of week (Mon) based on given date ##
queensland['start_week'] = pd.to_datetime(queensland['Date'])
queensland['start_week'] = queensland['start_week'].dt.to_period('W').apply(lambda r: r.start_time).dt.date


## Group daily to weekly data (Use median for price data) ##
queensland_weekly = queensland.groupby(['start_week'], as_index=False).median()


## Unique start_week ##
queensland_weekly = queensland_weekly.drop_duplicates(['start_week']).reset_index(drop=True)


## Generate CSV file ##
queensland_weekly.to_csv('C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/queensland_weekly.csv', index=False)
