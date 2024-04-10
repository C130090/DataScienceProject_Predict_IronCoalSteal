import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

coal_sulphur = pd.read_csv("C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Clean version binary choice HCC - China Coal Sulphur Index.csv").fillna(' ')


## Data Feeding ##
coal_sulphur.columns = coal_sulphur.iloc[0]
coal_sulphur = coal_sulphur.rename(columns={coal_sulphur.columns[0]: 'Date'})
coal_sulphur_units = coal_sulphur.iloc[1]
coal_sulphur = coal_sulphur.drop([0, 1]).reset_index(drop=True)


## Imputing Missing Data ##
def impute_missing_rows(df, col, row):
    #Impute for End of Row
    if ((row + 1) > df.index.max()):
        return df[df.columns[col]][row-1]

    #Compute number of consecutive missing rows
    start_value = float(df[df.columns[col]][row-1])
    end_value = 0
    divisor = 1

    while (df[df.columns[col]][row] ==  ' -   ') or \
          (df[df.columns[col]][row] == ' '):
         
        row_end = row + 1
        end_value = (df[df.columns[col]][row_end])
            
        while (end_value ==  ' -   ' or end_value ==  ' '):
            row_end += 1
            
            if row_end > df.index.max():
                return df[df.columns[col]][row-1]
            
            end_value = (df[df.columns[col]][row_end])
                
        divisor += 1
        row += 1
        
    return start_value + ((float(end_value) - float(start_value)) / divisor)

coal_sulphur = coal_sulphur[36:554]
coal_sulphur = coal_sulphur.reset_index(drop=True)

for col in range(coal_sulphur.shape[1]):
    for row in range(coal_sulphur.shape[0]):
        if (coal_sulphur[coal_sulphur.columns[col]][row]) ==  ' ':
            coal_sulphur[coal_sulphur.columns[col]][row] = impute_missing_rows(coal_sulphur, col, row)
            

## Split into input and output labels ##
X = coal_sulphur
coal_sulphur_units = coal_sulphur_units.drop(['Date']) 
        

 ## Retrieve start of week (Mon) based on given date ##
X['start_week'] = pd.to_datetime(X['Date'])
X['start_week'] = X['start_week'].dt.to_period('W').apply(lambda r: r.start_time).dt.date


## Replace 'Date' with 'start_week' && Drop 'start_week' ##
X['Date'] = X['start_week']
X = X.drop(['start_week'], axis=1)


## Rename 'Date' to 'start_week' ##
X = X.rename(columns={'Date': 'start_week'})

## Unique start_week ##
X = X.drop_duplicates(['start_week']).reset_index(drop=True)


#Convert all values from str to float
for col in range(X.shape[1]-1):
    for row in range(X.shape[0]):
            X[X.columns[col+1]][row] = np.float64(X[X.columns[col+1]][row])
      
         
#Convert dataframe to type(float)
for col_index in range(len(X.columns)-1):
    X[X.columns[col_index+1]] = X[X.columns[col_index+1]].astype(float)          
            
            
## Group daily to weekly data (Use median for price data) ##
X_weekly = X.groupby(['start_week'], as_index=False).median()          


## Add Missing rows for week ##
def resetValues(df):
    for col in range(df.shape[1]-1):
        for row in range(df.shape[0]):
            df[df.columns[col+1]][row] = ' '
    return df

data_new = X_weekly[0:2]
newpd1 = X_weekly[0:2].reset_index(drop=True)
newpd1['start_week'][0] = '2015-09-28'
newpd1['start_week'][1] = '2015-10-05'
newpd1 = resetValues(newpd1)
data_new = data_new.append(newpd1)

data_new = data_new.append(X_weekly[2:18])
newpd1 = X_weekly[16:18].reset_index(drop=True)
newpd1['start_week'][0] = '2016-02-01'
newpd1['start_week'][1] = '2016-02-08'
newpd1 = resetValues(newpd1)
data_new = data_new.append(newpd1)

data_new = data_new.append(X_weekly[18:103])
newpd1 = X_weekly[102:103].reset_index(drop=True)
newpd1['start_week'][0] = '2017-10-02'
newpd1 = resetValues(newpd1)
data_new = data_new.append(newpd1)

data_new = data_new.append(X_weekly[103:120])
data_new = data_new.reset_index(drop=True)

for col in range(data_new.shape[1]):
    for row in range(data_new.shape[0]):
        if (data_new[data_new.columns[col]][row]) ==  ' ':
            data_new[data_new.columns[col]][row] = impute_missing_rows(data_new, col, row)
            

## Reassigning variable ##
X_weekly = data_new
        
                  
## Visualization ##
for i in range(len(X_weekly.columns)-1):
    df = pd.DataFrame({X_weekly.columns[i+1]: X_weekly[X_weekly.columns[i+1]].tolist()})
    df = df.astype(float)
    ax = df.plot(kind='line', figsize=(6, 6))
    ax.set_xlabel('Date')
    ax.set_ylabel(coal_sulphur_units[i])   
    n = 20
    ax.set_xticklabels(X_weekly[X_weekly.columns[0]].tolist()[::n], rotation=20)
    ax.legend(loc=2)  
    plt.show()          
            

## Generate CSV file ##
X_weekly.to_csv('C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/coal_sulphur.csv', index=False)            
            
            
            
            
            
            
            
            
            



