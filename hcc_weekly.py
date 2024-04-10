import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

hcc_weekly = pd.read_csv("C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Clean version binary choice HCC - Weekly.csv").fillna(' ')


## Data Feeding ##
hcc_weekly.columns = hcc_weekly.iloc[0]
hcc_weekly = hcc_weekly.rename(columns={hcc_weekly.columns[0]: 'Date'})
hcc_weekly_units = hcc_weekly.iloc[1]
hcc_weekly = hcc_weekly.drop([0, 1]).reset_index(drop=True)
hcc_weekly_1 = hcc_weekly.iloc[:, 0:25]
hcc_weekly_2 = hcc_weekly.iloc[:, 25:33]


## Fill in missing units ##
hcc_weekly_units['Coke price'] = hcc_weekly_units['Coke Price']

## Swap rows - date ##
temp = hcc_weekly_2[hcc_weekly_2.index == 179]
hcc_weekly_2.iloc[hcc_weekly_2.index == 179] = hcc_weekly_2.iloc[hcc_weekly_2.index == 180].values
hcc_weekly_2.iloc[hcc_weekly_2.index == 180] = temp.values


## Remove duplicated rows ##
hcc_weekly_2 = hcc_weekly_2.drop([181, 182]).reset_index(drop=True)


## Remove unnecessary features ##
hcc_weekly_1 = hcc_weekly_1.drop(['Utilization rate at standalone cokery (230 samples)', 
                                  'Days of CC consumption at standalone cokery (230 samples)',
                                  'Coke Price'], axis=1)


## Imputing Missing Data ##
def impute_missing_rows(col, row):
    #Impute for End of Row
    if ((row + 1) > hcc_weekly_1.index.max()):
        return hcc_weekly_1[hcc_weekly_1.columns[col]][row-1]

    #Compute number of consecutive missing rows
    start_value = float(hcc_weekly_1[hcc_weekly_1.columns[col]][row-1])
    end_value = 0
    divisor = 1

    while (hcc_weekly_1[hcc_weekly_1.columns[col]][row] ==  ' -   ') or \
          (hcc_weekly_1[hcc_weekly_1.columns[col]][row] == ' '):
         
        row_end = row + 1
        end_value = (hcc_weekly_1[hcc_weekly_1.columns[col]][row_end])
            
        while (end_value ==  ' -   ' or end_value ==  ' '):
            row_end += 1
            
            if row_end > hcc_weekly_1.index.max():
                return hcc_weekly_1[hcc_weekly_1.columns[col]][row-1]
            
            end_value = (hcc_weekly_1[hcc_weekly_1.columns[col]][row_end])
                
        divisor += 1
        row += 1
        
    return start_value + ((float(end_value) - float(start_value)) / divisor)

hcc_weekly_1 = hcc_weekly_1[328:hcc_weekly_1.shape[0]]
hcc_weekly_2 = hcc_weekly_2[0:446]
hcc_weekly_1 = hcc_weekly_1.reset_index(drop=True)
hcc_weekly_2 = hcc_weekly_2.reset_index(drop=True)


for col in range(hcc_weekly_1.shape[1]):
    for row in range(hcc_weekly_1.shape[0]):
        if (hcc_weekly_1[hcc_weekly_1.columns[col]][row]) ==  ' ':
            hcc_weekly_1[hcc_weekly_1.columns[col]][row] = impute_missing_rows(col, row)


## Split into input and output labels ##
hcc_weekly_1_units = hcc_weekly_units[0:25]
hcc_weekly_2_units = hcc_weekly_units[25:33]            
hcc_weekly_1_units = hcc_weekly_1_units.drop(['Date', 
                                              'Utilization rate at standalone cokery (230 samples)', 
                                              'Days of CC consumption at standalone cokery (230 samples)',
                                              'Coke Price'])
hcc_weekly_2_units = hcc_weekly_2_units.drop(['Date'])            
X1 = hcc_weekly_1
X2 = hcc_weekly_2         
            

## Visualization - X1 ##
for i in range(len(X1.columns)-1):
    df = pd.DataFrame({X1.columns[i+1]: X1[X1.columns[i+1]].tolist()})
    df = df.astype(float)
    ax = df.plot(kind='line', figsize=(6, 6))
    ax.set_xlabel(X1.columns[0])
    ax.set_ylabel(hcc_weekly_1_units[i])   
    n = 50
    ax.set_xticklabels(X1[X1.columns[0]].tolist()[::n], rotation=20)
    ax.legend(loc=2)
    plt.show()


## Visualization - X2 ##
for i in range(len(X2.columns)-1):
    df = pd.DataFrame({X2.columns[i+1]: X2[X2.columns[i+1]].tolist()})
    df = df.astype(float)
    ax = df.plot(kind='line')
    ax.set_xlabel(X2.columns[0])
    ax.set_ylabel(hcc_weekly_2_units[i])   
    n = 50
    ax.set_xticklabels(X2[X2.columns[0]].tolist()[::n], rotation=25)
    ax.legend(loc=2)
    
## Retrieve start of week (Mon) based on given date ##    
X1['start_week'] = pd.to_datetime(X1['Date'])
X1['start_week'] = X1['start_week'].dt.to_period('W').apply(lambda r: r.start_time).dt.date
    
X2['start_week'] = pd.to_datetime(X2['Date'])
X2['start_week'] = X2['start_week'].dt.to_period('W').apply(lambda r: r.start_time).dt.date     


## Replace 'Date' with 'start_week' && Drop 'start_week' ##
X1['Date'] = X1['start_week']
X1 = X1.drop(['start_week'], axis=1)

X2['Date'] = X2['start_week']
X2 = X2.drop(['start_week'], axis=1)


## Rename 'Date' to 'start_week' ##
X1 = X1.rename(columns={'Date': 'start_week'})
X2 = X2.rename(columns={'Date': 'start_week'})


## Unique start_week ##
X1 = X1.drop_duplicates(['start_week']).reset_index(drop=True)
X2 = X2.drop_duplicates(['start_week']).reset_index(drop=True)


## Convert all values from str to float ##
for col in range(X1.shape[1]-1):
    for row in range(X1.shape[0]):
            X1[X1.columns[col+1]][row] = np.float64(X1[X1.columns[col+1]][row])
            
for col in range(X2.shape[1]-1):
    for row in range(X2.shape[0]):
            X2[X2.columns[col+1]][row] = np.float64(X2[X2.columns[col+1]][row])
            
        
## Generate CSV file ##
X1.to_csv('C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/hcc_weekly_X1.csv', index=False)
X2.to_csv('C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/hcc_weekly_X2.csv', index=False)

