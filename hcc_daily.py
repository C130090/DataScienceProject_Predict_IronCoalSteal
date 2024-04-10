import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

hcc_daily = pd.read_csv("C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Clean version binary choice HCC - Daily.csv").fillna(' ')


## Data Feeding ##
hcc_daily.columns = hcc_daily.iloc[0]
hcc_daily = hcc_daily.rename(columns={hcc_daily.columns[0]: 'Date'})
hcc_daily_units = hcc_daily.iloc[1]
hcc_daily = hcc_daily.drop([0, 1]).reset_index(drop=True)


## Remove duplicated rows ##
hcc_daily = hcc_daily.drop([1016]).reset_index(drop=True)


## Imputing Missing Data ##
def impute_missing_rows(col, row):
    #Impute for End of Row
    if ((row + 1) > hcc_daily.index.max()):
        return hcc_daily[hcc_daily.columns[col]][row-1]

    #Compute number of consecutive missing rows
    start_value = float(hcc_daily[hcc_daily.columns[col]][row-1])
    end_value = 0
    divisor = 1

    while (hcc_daily[hcc_daily.columns[col]][row] ==  ' -   ') or \
          (hcc_daily[hcc_daily.columns[col]][row] == ' '):
         
        row_end = row + 1
        end_value = (hcc_daily[hcc_daily.columns[col]][row_end])
            
        while (end_value ==  ' -   ' or end_value ==  ' '):
            row_end += 1
            
            if row_end > hcc_daily.index.max():
                return hcc_daily[hcc_daily.columns[col]][row-1]
            
            end_value = (hcc_daily[hcc_daily.columns[col]][row_end])
                
        divisor += 1
        row += 1
        
    return start_value + ((float(end_value) - float(start_value)) / divisor)

for col in range(hcc_daily.shape[1]):
    for row in range(hcc_daily.shape[0]):
        if (hcc_daily[hcc_daily.columns[col]][row]) ==  ' ':
            hcc_daily[hcc_daily.columns[col]][row] = impute_missing_rows(col, row)


## Split into input and output labels ##
X = hcc_daily.drop(['PLATTS PREMIUM LOW VOL FOB AUS'], axis=1)
Y = hcc_daily['PLATTS PREMIUM LOW VOL FOB AUS']
hcc_daily_units = hcc_daily_units.drop(['Date', 'PLATTS PREMIUM LOW VOL FOB AUS'])


## Visualization ##
for i in range(len(X.columns)-1):
    df = pd.DataFrame({X.columns[i+1]: X[X.columns[i+1]].tolist()})
    df = df.astype(float)
    ax = df.plot(kind='line', figsize=(6, 6))
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(hcc_daily_units[i])   
    n = 200
    ax.set_xticklabels(X[X.columns[0]].tolist()[::n])
    ax.legend(loc=2)
    plt.show()

## Data Preparation - Combine Input values with Target Values (For Weekly Alignment) ##
X['PLATTS PREMIUM LOW VOL FOB AUS'] = Y
    
    
## Retrieve start of week (Mon) based on given date ##
X['start_week'] = pd.to_datetime(X['Date'])
X['start_week'] = X['start_week'].dt.to_period('W').apply(lambda r: r.start_time).dt.date


## Convert all values from str to float ##
for col in range(X.shape[1]-2):
    for row in range(X.shape[0]):
            X[X.columns[col+1]][row] = np.float64(X[X.columns[col+1]][row])
      
         
## Convert dataframe to type(float) ##
for col_index in range(len(X.columns)-2):
    X[X.columns[col_index+1]] = X[X.columns[col_index+1]].astype(float)


## Group daily to weekly data (Use median for price data) ##
X_weekly = X.groupby(['start_week'], as_index=False).median()


## Unique start_week ##
X_weekly = X_weekly.drop_duplicates(['start_week']).reset_index(drop=True)


## Generate CSV file ##
X_weekly.to_csv('C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/hcc_daily.csv', index=False)

