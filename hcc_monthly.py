import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta
from calendar import monthrange

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

hcc_monthly = pd.read_csv("C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Clean version binary choice HCC - Monthly.csv").fillna(' ')


## Data Feeding ##
hcc_monthly = hcc_monthly.rename(columns={hcc_monthly.columns[0]: 'Date'})
hcc_monthly_units = hcc_monthly.iloc[0]
hcc_monthly = hcc_monthly.drop([0]).reset_index(drop=True)


## Imputing Missing Data ##
def impute_missing_rows(col, row):
    #Impute for End of Row
    if ((row + 1) > hcc_monthly.index.max()):
        return hcc_monthly[hcc_monthly.columns[col]][row-1]

    #Compute number of consecutive missing rows
    start_value = float(hcc_monthly[hcc_monthly.columns[col]][row-1])
    end_value = 0
    divisor = 1

    while (hcc_monthly[hcc_monthly.columns[col]][row] ==  ' -   ') or \
          (hcc_monthly[hcc_monthly.columns[col]][row] == ' '):
         
        row_end = row + 1
        end_value = (hcc_monthly[hcc_monthly.columns[col]][row_end])
            
        while (end_value ==  ' -   ' or end_value ==  ' '):
            row_end += 1
            
            if row_end > hcc_monthly.index.max():
                return hcc_monthly[hcc_monthly.columns[col]][row-1]

            end_value = (hcc_monthly[hcc_monthly.columns[col]][row_end])
                
        divisor += 1
        row += 1
        
    return start_value + ((float(end_value) - float(start_value)) / divisor)


hcc_monthly = hcc_monthly[120:205]
hcc_monthly = hcc_monthly.reset_index(drop=True)

for col in range(hcc_monthly.shape[1]):
    for row in range(hcc_monthly.shape[0]):
        if (hcc_monthly[hcc_monthly.columns[col]][row]) ==  ' ':
            hcc_monthly[hcc_monthly.columns[col]][row] = impute_missing_rows(col, row)


## Split into input and output labels ##
X = hcc_monthly
hcc_monthly_units = hcc_monthly_units.drop(['Date'])


## Visualization ##
for i in range(len(X.columns)-1):
    df = pd.DataFrame({X.columns[i+1]: X[X.columns[i+1]].tolist()})
    df = df.astype(float)
    ax = df.plot(kind='line', figsize=(6, 6))
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(hcc_monthly_units[i])   
    n = 10
    ax.set_xticklabels(X[X.columns[0]].tolist()[::n])
    ax.legend(loc=2)
    plt.show()


## Generate all days per month ##
init_date = pd.to_datetime('2001-01-01')

dict_date = {'date': [init_date]}

for i in range(7000):
    i = i + 1
    dict_date['date'].append(pd.to_datetime(init_date + timedelta(days=i)))

pd_date = pd.DataFrame(data=dict_date)
pd_date['date'] = pd_date['date'].dt.date
pd_date = pd_date['date'][3652:6240]

## Count number of days in month ##
from datetime import date

days_in_months_list = []

diff_years = 2019 - 2011
year = 2011
month = 0

for year_diff in range(diff_years):
    for month_diff in range(12):
        month += 1
        next_month = month + 1
        
        if next_month == 13:
            next_year = year + 1
            next_month_1 = 1
            day_in_month = (date(next_year, next_month_1, 1) - date(year, month, 1)).days
            days_in_months_list.append(day_in_month)
            month = 0 #reset
        else:   
            day_in_month = (date(year, next_month, 1) - date(year, month, 1)).days
            days_in_months_list.append(day_in_month)
    year += 1   
    
    
#Pop off unused months in a year"
for i in range(11):
    days_in_months_list.pop()

        
## Create Daily Dataframe ##
pd_dict_daily = pd.DataFrame()

for i in range(len(hcc_monthly.columns)):
    pd_dict_daily[hcc_monthly.columns[i]] = ""

    
## Convert Monthly data to daily data
col_num = 0
for col_index in range(len(hcc_monthly.columns)-1):
    curr_list = []
    col_num += 1
    for index in range(len(X[X.columns[col_num]])):
        curr_month_value = float(X[X.columns[col_num]][index])
        curr_month_days = days_in_months_list[index]
        for i in range(curr_month_days):
            cal_daily_value = round(float(curr_month_value/curr_month_days), 3)
            curr_list.append(cal_daily_value)          
    pd_dict_daily[X.columns[col_num]] = curr_list      

pd_dict_daily['Date'] = pd_date.values


## Retrieve start of week (Mon) based on given date ##
pd_dict_daily['start_week'] = pd.to_datetime(pd_dict_daily['Date'])
pd_dict_daily['start_week'] = pd_dict_daily['start_week'].dt.to_period('W').apply(lambda r: r.start_time).dt.date


## Group daily to weekly data (Use sum for vol data) ##
X_weekly = pd_dict_daily.groupby(['start_week'], as_index=False).sum()


## Unique start_week ##
X_weekly = X_weekly.drop_duplicates(['start_week']).reset_index(drop=True)


## Generate CSV file ##
X_weekly.to_csv('C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/hcc_monthly.csv', index=False)

