import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import math
import os.path

from datetime import timedelta, date
from calendar import monthrange
from sklearn.model_selection import train_test_split

start_time = time.time()

workingDir = "C:/Users/tanyixin/Desktop/Commodity Price Forecasting/Compilation/"  ##### TO UPDATE #####

coal_sulphur = pd.read_csv(workingDir + "coal_sulphur.csv").fillna('nan')
hcc_daily = pd.read_csv(workingDir + "hcc_daily.csv").fillna('nan')
hcc_weekly_X1 = pd.read_csv(workingDir + "hcc_weekly_X1.csv").fillna('nan')
hcc_weekly_X2 = pd.read_csv(workingDir + "hcc_weekly_X2.csv").fillna('nan')
hcc_monthly = pd.read_csv(workingDir + "hcc_monthly.csv").fillna('nan')
queensland = pd.read_csv(workingDir + "queensland_weekly.csv").fillna('nan')
mfgChina = pd.read_csv(workingDir + "mfg_weekly.csv").fillna('nan')
avgSteelMargin = pd.read_csv(workingDir + "avgSteelMargin.csv").fillna('nan')
averageSteelMargin_binary = pd.read_csv(workingDir + "AverageSteelMargin-Binary.csv").fillna('nan')
bfOperatingRate = pd.read_csv(workingDir + "BFOperatingRate.csv").fillna('nan')

## Generate weekly start date ##
init_date = pd.to_datetime('2001-01-01')

dict_date = {'start_week': [init_date]}

for i in range(1000):
    i = (i+1) * 7
    dict_date['start_week'].append(pd.to_datetime(init_date + timedelta(days=i)))

compile_weekly = pd.DataFrame(data=dict_date)
compile_weekly['start_week'] = compile_weekly['start_week'].dt.date

## Convert 'start_week' from String to Date ##
coal_sulphur['start_week'] = coal_sulphur['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
hcc_daily['start_week'] = hcc_daily['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
hcc_weekly_X1['start_week'] = hcc_weekly_X1['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
hcc_weekly_X2['start_week'] = hcc_weekly_X2['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
hcc_monthly['start_week'] = hcc_monthly['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
queensland['start_week'] = queensland['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
mfgChina['start_week'] = mfgChina['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
avgSteelMargin['start_week'] = avgSteelMargin['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
averageSteelMargin_binary['start_week'] = averageSteelMargin_binary['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date
bfOperatingRate['start_week'] = bfOperatingRate['start_week'].apply(lambda x: pd.to_datetime(x)).dt.date

## Merge all datasets ##
compile_weekly = pd.merge(compile_weekly, coal_sulphur, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, hcc_daily, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, hcc_weekly_X1, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, hcc_weekly_X2, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, hcc_monthly, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, queensland, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, mfgChina, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, avgSteelMargin, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, averageSteelMargin_binary, how='left', on=['start_week'])
compile_weekly = pd.merge(compile_weekly, bfOperatingRate, how='left', on=['start_week'])


# =============================================================================
# 
# 1) Data Cutting - Based on Target
#     
# =============================================================================
## Search for Start Index from Target ##
start_index = compile_weekly['PLATTS PREMIUM LOW VOL FOB AUS'].apply(lambda x: np.float64(x)).first_valid_index()
       
## Search for End Index from Target ##   
end_index = compile_weekly['PLATTS PREMIUM LOW VOL FOB AUS'].apply(lambda x: np.float64(x)).last_valid_index() + 1 
    

## Retrieve rows from all columns ## 
data_remaining = compile_weekly[start_index:end_index]
data_remaining = data_remaining.reset_index(drop=True)


# =============================================================================
# 
# Imputing Missing Data
# 
# =============================================================================
def impute_missing_rows(df, col, row):
    
    start_value = float(df[df.columns[col]][row-1])
    end_value = 0
    divisor = 2
    
    row_end = row + 1
    end_value = (df[df.columns[col]][row_end])
    
    while (math.isnan(end_value)):
        row_end += 1
        end_value = (df[df.columns[col]][row_end])
        divisor += 1
        
    return start_value + ((float(end_value) - float(start_value)) / divisor)


for col in range(data_remaining.shape[1]-1):
    col += 1
      
    ## Search for Start Key ##
    start_key = data_remaining[data_remaining.columns[col]].apply(lambda x: np.float64(x)).first_valid_index()
    
    ## Search for End Key ##
    end_key = data_remaining[data_remaining.columns[col]].apply(lambda x: np.float64(x)).last_valid_index()
            
    ## Calling Impute Function ##
    for row in range(data_remaining.shape[0] - (data_remaining.shape[0] - 1 - end_key) - start_key): ##Continue here
        row += start_key
        if math.isnan(data_remaining[data_remaining.columns[col]][row]):
            data_remaining[data_remaining.columns[col]][row] = impute_missing_rows(data_remaining, col, row)   
            
            
# =============================================================================
# 
# Backup File
# 
# =============================================================================
## Generate CSV file ##
data_remaining.to_csv('C:/Users/tanyi9/Desktop/Commodity Price Forecasting/Compilation/hcc_compilation.csv', index=False)

## Read CSV file ##
data_remaining = pd.read_csv("C:/Users/tanyi9/Desktop/Commodity Price Forecasting/Compilation/hcc_compilation.csv").fillna('nan')            
            
## Take Data from April 2014 onwards ##
data_remaining = data_remaining[11:].reset_index(drop=True)
    

# =============================================================================
#     
# FEATURE SELECTION
# 
# =============================================================================

data_remaining = data_remaining[[
                                   'start_week', #Default
                                   'PLATTS PREMIUM LOW VOL FOB AUS', #Default
                                   
                                   ## Selected Features ##
                                   'Coke inventory at steel mill (110 samples)',
                                   'Coke inventory at standalone cokery(100 samples)',
                                   'Queensland vessel queues',
                                   'Aus Met Exports',
                                   'US Met Exports',
                                   'India coking coal imports',
                                   'Coke price',
                                   'Rebar China exports',
                                   'Orginal Operating Rate',
                                   'ChinaPMI',
                                   'Average steel Margin',
                                   'Avg steel margin binary',
                                   'BF operating rate',
                                   'BF operating rate-Threshold',
                                   #'HRC China spot incl. VAT',
                                   #'Utilization rate at standalone cokery (100 samples)',
                                   #'Steel Margins',
                                   #'CN: Purchasing Managers\' Index: Mfg (China)', 

                                   ## Dummy Variables ##
                                   #'P1',
                                   #'P2',
                                   
                                   ## Temporary Added Variables ##
                                   #'Import met coal inventory at ports',
                                   'China coking coal imports',
                                   'CC inventory at steel mill affiliated cokery (110 samples)',
                                   'CC inventory at standalone cokery (100 samples)   ',
                                   'Stock at China domestic mines ',
                                   'Import met coal inventory at ports',
                                ]]




# =============================================================================
#     
# CREATE NEW FEATURES
# 
# =============================================================================
## 3) Create 'Inventory Downstream' Features ##
data_remaining['CC inventory downstream'] = data_remaining['CC inventory at steel mill affiliated cokery (110 samples)'] + data_remaining['CC inventory at standalone cokery (100 samples)   ']
data_remaining['Coke inventory at downstream'] = data_remaining['Coke inventory at steel mill (110 samples)'] + data_remaining['Coke inventory at standalone cokery(100 samples)']


## Drop Features ##
data_remaining = data_remaining.drop(['CC inventory at steel mill affiliated cokery (110 samples)',
                                      'CC inventory at standalone cokery (100 samples)   ',
                                      'Coke inventory at steel mill (110 samples)',
                                      'Coke inventory at standalone cokery(100 samples)',
                                      ], axis = 1)
    
        
# =============================================================================
# 
# Generate LAG cross-time frames
# 
# =============================================================================
lag_cross_time_frames = pd.DataFrame()
lag_cross_time_frames['start_week'] = data_remaining['start_week']
lag_cross_time_frames['PLATTS PREMIUM LOW VOL FOB AUS'] = data_remaining['PLATTS PREMIUM LOW VOL FOB AUS']

        
col_lags_0 = [
              'Average steel Margin',
              ]

num_lags = 1   #specify number of lags e.g. 27 meant lags 0-26 (27 diff kinds of lag)
for index in range(num_lags):
    index += 0
    for col in range(len(col_lags_0)):
        X_new = data_remaining[col_lags_0[col]].shift(index)
        lag_cross_time_frames[str(col_lags_0[col]) + str(" - Lag") + str(index)] = X_new  


col_lags_0 = [
              'ChinaPMI',
              ]

num_lags = 1   #specify number of lags e.g. 27 meant lags 0-26 (27 diff kinds of lag)
for index in range(num_lags):
    index += 8
    for col in range(len(col_lags_0)):
        X_new = data_remaining[col_lags_0[col]].shift(index)
        lag_cross_time_frames[str(col_lags_0[col]) + str(" - Lag") + str(index)] = X_new 
        

col_lags_0 = [
              'US Met Exports',
              ]

num_lags = 1   #specify number of lags e.g. 27 meant lags 0-26 (27 diff kinds of lag)
for index in range(num_lags):
    index += 13
    for col in range(len(col_lags_0)):
        X_new = data_remaining[col_lags_0[col]].shift(index)
        lag_cross_time_frames[str(col_lags_0[col]) + str(" - Lag") + str(index)] = X_new 

lag_cross_time_frames = lag_cross_time_frames.drop(['start_week', 
                                                    'PLATTS PREMIUM LOW VOL FOB AUS',                               
                                                    ], axis = 1)


# =============================================================================
# 
# 2) Data Cutting - Dynamic Data Cutting
#     
# =============================================================================
        
def dynamicDataCutting(data_remaining):    
    ## Search for Start Index from Target ##
    start_index = 0
    for col in range(data_remaining.shape[1]):
        row = data_remaining[data_remaining.columns[col]].apply(lambda x: np.float64(x)).first_valid_index()
        if row >= start_index:
            start_index = row
            
    ## Search for End Index from Target ##   
    end_index = 10000000
    for col in range(data_remaining.shape[1]):
        row = data_remaining[data_remaining.columns[col]].apply(lambda x: np.float64(x)).last_valid_index()
        if row <= end_index:
            end_index = row        
                          
    end_index += 1  
    
    ## Retrieve rows from all columns ## 
    data_remaining1 = data_remaining[start_index:end_index]
    data_remaining1 = data_remaining1
    return data_remaining1


# =============================================================================
# 
# COMBINATION GENERATOR
# 
# =============================================================================

import itertools

col_num_arr = []
for col_num in range(lag_cross_time_frames.shape[1]):
    col_num_arr.append(col_num)
    
result_pd = pd.DataFrame()
feat_selected = []
countSigPValue = []
train_result = []
test_result = []
bench_mark = []
train_size = []
test_size = []
resultList = []
result_summary = []
    
## Specify number of features applied ##
for iter_tries in range(1):
    iter_tries += 3 #Pick 3,4,5 ##### TO UPDATE #####

    storage_combine = list(itertools.combinations(col_num_arr, iter_tries))
    
    for combi in range(len(storage_combine)):
        final_frame = pd.DataFrame()
        for cols in range(len(storage_combine[combi])):
            final_frame[lag_cross_time_frames.columns[storage_combine[combi][cols]]] = lag_cross_time_frames[lag_cross_time_frames.columns[storage_combine[combi][cols]]]
            
        ## Call Dynamic Data Cutting Function ##
        data_remaining1 = dynamicDataCutting(final_frame)
        first_index = data_remaining1.first_valid_index()
        last_index = data_remaining1.last_valid_index()
        
        data_remaining1['start_week'] = data_remaining['start_week'][first_index:last_index+1]
        data_remaining1['PLATTS PREMIUM LOW VOL FOB AUS'] = data_remaining['PLATTS PREMIUM LOW VOL FOB AUS'][first_index:last_index+1]

        data_remaining1 = data_remaining1.reset_index(drop=True)

        # =============================================================================
        # 
        # FEATURE ENGINEERING
        # 
        # =============================================================================
        
        ## Data Preparation ##
        acc_test_RF = 0 #initial benchmark
        acc_test_logR = 0 #initial benchmark
        wanted_feat = []
        
        benchmark = [1.]
        total_periods = [13]  ##### TO UPDATE #####  #e.g. 1 - Forecast 1 weeks ahead
        
        for num_period in range(len(total_periods)):
            period = total_periods[num_period]
        
            X_train_val_test = data_remaining1.drop(['start_week', 'PLATTS PREMIUM LOW VOL FOB AUS'], axis = 1)[0:(data_remaining1.shape[0]-period)]
            target_list1 = []
            target_list_percent = []
            
            for i in range(data_remaining1.shape[0]-period):
                cal1 = data_remaining1['PLATTS PREMIUM LOW VOL FOB AUS'][i+period] - data_remaining1['PLATTS PREMIUM LOW VOL FOB AUS'][i]
                calpercent = (data_remaining1['PLATTS PREMIUM LOW VOL FOB AUS'][i+period] - data_remaining1['PLATTS PREMIUM LOW VOL FOB AUS'][i]) / data_remaining1['PLATTS PREMIUM LOW VOL FOB AUS'][i] *100 
                    
                if cal1 > 0.0:
                    cal1 = 1
                else:
                    cal1 = 0    
                    
                target_list1.append(cal1) 
                target_list_percent.append(calpercent)
            
            target_y1 = pd.DataFrame()
            target_y1['PLATTS PREMIUM LOW VOL FOB AUS_' + str(period) + '_Week_Ahead'] = target_list1
            target_y_percent = pd.DataFrame()
            target_y_percent['PLATTS PREMIUM LOW VOL FOB AUS_' + str(period) + '_Week_Ahead'] = target_list_percent
            
            ## Change dataframe to series ##
            target_y = target_y1['PLATTS PREMIUM LOW VOL FOB AUS_' + str(period) + '_Week_Ahead'] 
        
            train_x, test_x = train_test_split(X_train_val_test, test_size=0.3, shuffle=False) ##### TO UPDATE #####
            train_y, test_y = train_test_split(target_y, test_size=0.3, shuffle=False) ##### TO UPDATE #####

            logit_model = sm.Logit(train_y, train_x.astype(float))
            result = logit_model.fit()
        
            y_pred_train_logR = result.predict(train_x.astype(float))
            y_pred_test_logR = result.predict(test_x.astype(float))
        
            y_pred_train_logR = y_pred_train_logR.apply(lambda x: 1 if x > 0.5 else 0)
            y_pred_test_logR = y_pred_test_logR.apply(lambda x: 1 if x > 0.5 else 0)
            
            tn1, fp1, fn1, tp1 = confusion_matrix(train_y, y_pred_train_logR).ravel()
            tn3, fp3, fn3, tp3 = confusion_matrix(test_y, y_pred_test_logR).ravel()
              
            acc_train_logR = (tp1 + tn1) / (tn1 + fp1 + fn1 + tp1)
            acc_test_logR = (tp3 + tn3) / (tn3 + fp3 + fn3 + tp3)
            
            random_guess_Benchmark = test_y.mean() ## % Price Up ##
            random_guess_Benchmark1 = 1 - test_y.mean() ### % Price Down ##
            
            ## Higher of % Price Up/% Price Down ##
            if random_guess_Benchmark1 > random_guess_Benchmark:
                random_guess_Benchmark = random_guess_Benchmark1
                
            ## Significance P-Values / Total P-Values ##
            countSig = 0
            for i in range(len(result.pvalues.index.values)):
                if result.pvalues.values[i] <= 0.05:
                   countSig += 1 
                   
            numCountSig = countSig / len(result.pvalues.index.values)
            
            feat_selected.append(train_x.columns)
            countSigPValue.append(numCountSig)
            train_result.append(acc_train_logR)
            test_result.append(acc_test_logR)
            bench_mark.append(random_guess_Benchmark)
            train_size.append(train_x.shape[0])
            test_size.append(test_x.shape[0])
            resultList.append(result)
            result_summary.append(result.summary())
        
        
result_pd['Features'] = feat_selected
result_pd['Count P-Value Significant'] = countSigPValue
result_pd['train_result'] = train_result
result_pd['test_result'] = test_result 
result_pd['bench_mark'] = bench_mark 
result_pd['train_size'] = train_size 
result_pd['test_size'] = test_size 
result_pd['result'] = resultList
result_pd['result_summary'] = result_summary 
   
## Select top 50 Results ##
result_pd_sorted = result_pd.sort_values(by=['test_result', 'train_result'], ascending=[False, False]).reset_index(drop=True)
    
    
# =============================================================================
# 
# OUTPUT GENERATOR
# 
# =============================================================================
excelDf = pd.DataFrame()
modelID = []
terms = []
coef = []
pvalues =[]
numCountSigPValue = []
trainAcc =[]
testAcc = []
benchMark = []
trainSize = []
testSize = []


filename = "C:/Users/tanyixin/OneDrive/30 May/29Jun/hcc_results_Wk" + str(period) + ".txt" ##### TO UPDATE #####
filename1 = "C:/Users/tanyixin/OneDrive/30 May/29Jun/hcc_results_Wk" + str(period) + ".xlsx" ##### TO UPDATE #####

if (os.path.isfile(filename)):
    os.remove(filename)

with open(filename, "a") as myfile:

    for idx in range(result_pd_sorted.shape[0]):
        
        result = result_pd_sorted['result'][idx]
        
        for i in range(len(result.pvalues.index.values)):
            modelID.append(idx)
            terms.append(result.pvalues.index.values[i])
            coef.append(round(result.params.values[i], 4))
            pvalues.append(round(result.pvalues.values[i], 3))
            numCountSigPValue.append(result_pd_sorted.iloc[idx]['Count P-Value Significant'])
            trainAcc.append(result_pd_sorted.iloc[idx]['train_result'])
            testAcc.append(result_pd_sorted.iloc[idx]['test_result'])
            benchMark.append(result_pd_sorted.iloc[idx]['bench_mark'])
            trainSize.append(result_pd_sorted.iloc[idx]['train_size'])
            testSize.append(result_pd_sorted.iloc[idx]['test_size'])
        
        # =============================================================================
        #     
        #     WRITE TO NOTEPAD
        #     
        # =============================================================================
        print("\n", "-"*5, "Predict (PLATTS PREMIUM LOW VOL FOB AUS) - ", period, " Weeks Ahead", "-"*5, file=myfile)
        print("\nTrain ACC - LogR:", result_pd_sorted.iloc[idx]['train_result'], file=myfile)      
        print("\nTest ACC - LogR:", result_pd_sorted.iloc[idx]['test_result'], file=myfile)
        print("\nRandom Guess Benchmark:", result_pd_sorted.iloc[idx]['bench_mark'], file=myfile)
        print(result_pd_sorted.iloc[idx]['result_summary'], file=myfile)
        
    
excelDf['Model_ID'] = modelID
excelDf['Features'] = terms
excelDf['Coef'] = coef
excelDf['P-Values'] = pvalues
excelDf['Count P-Value Significant'] = numCountSigPValue
excelDf['Train Acc'] = trainAcc
excelDf['Test Acc'] = testAcc
excelDf['Bench Mark'] = benchMark
excelDf['Train Size'] = trainSize
excelDf['Test Size'] = testSize
    
    
myfile.close()
excelDf.to_excel(filename1, index=False)   
 
end_time = time.time()
print("\nElapsed Time: ", int(round((end_time - start_time)/60)), "min")


