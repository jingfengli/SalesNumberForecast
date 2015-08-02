# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:38:19 2015

@author: jingfengli

This script predicts the next month's sales number in Canada. 
It contains 0) loading the data, 1) feature engineering,  2) random forest regression,
            3) graphing        , 4) save data for matlab graphing
"""

#%% 0) LOAD THE DATA FROM LOCAL DIRECTORY
##########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv('/Volumes/mac/Work/JobHunt/Palantir/PE Exercise/SummaryData.csv') # downloaded from statcan.gc.ca
time0 = df.Ref_Date

#%% 1) Feature Engineering 
##########################################################
# monthly sales contain a trend. Thus, whitening all the data 
# by looking at the percent change in a given month’s data relative to the previous month. 
salesunadjusted = np.asarray(df.Unadjusted)
Percent_salesunadjusted = salesunadjusted[1:]  * 100.0 / salesunadjusted[:-1] -100
Percent_salesadjusted =  np.asarray(df.Seasonally_adjusted[1:]) *100.0/ np.asarray(df.Seasonally_adjusted[:-1]) -100
Percent_Stock = np.asarray(df.Stock_Exchange_value[1:]) *100.0/ np.asarray(df.Stock_Exchange_value[:-1]) -100
Percent_Dow_Jones = np.asarray(df.Dow_Jones_index[1:]) *100.0/ np.asarray(df.Dow_Jones_index[:-1]) -100
Percent_Unemployment = np.asarray(df.Unemployment[1:]) *100.0/ np.asarray(df.Unemployment[:-1]) -100

Percent_GDP = np.asarray(df.GDP[1:]) *100.0/ np.asarray(df.GDP[:-1]) - 100

# GDP data is missing before 1997
# Imputing missing values in Percent_GDP, using the median of the same month across years
for i in range(len(Percent_GDP)):
    if np.isnan(Percent_GDP[i]):
        Percent_GDP[i] = np.nanmedian(Percent_GDP[i::12])

#%% 2) Random Forest Regression
##########################################################
# Set up Features and Labels for random forest
# Define FEATURES:
#       the past 12 month of the unadjusted sales, 
#       Other indexes of past month

Nback = 12 # look at the Nback history

feat = np.zeros([len(Percent_salesunadjusted) - 12,17]) # 12 history points, and 4 other indicates

# put the past i month unadjusted data as features
for i in range(Nback):
    feat[...,i] = Percent_salesunadjusted[ i : i+len(Percent_salesunadjusted)-12 ]
    
# Other features from last Month

feat[...,12] = Percent_salesadjusted[Nback-1:-1]  #season_adjustedSales
feat[...,13] = Percent_Stock[Nback-1:-1]        #Stock_exchange
feat[...,14] = Percent_Dow_Jones[Nback-1:-1]    #Dow_Jones
feat[...,15] = Percent_Unemployment[Nback-1:-1] #unemployment
feat[...,16] = Percent_GDP[Nback-1:-1]          #GDP with imputing numbers

# Define LABELS:
#       the percent increase in sales
#       excluding the first Nback points, because they dont have features associated with them        
labl = Percent_salesunadjusted[Nback:]

#%% Run Random Forest
#  To predict the kth data point, I used data from 1 to (k-1) data points to train the model.
#  Set an arbitraray minimum for k
kst = 5*12-1                # Let k > 5*12, because for a smaller k, the amount of training data is very small

n = len(labl) - kst         # Total number of predictions [or tests] generated
test_feat = feat[kst:,:]    # features associated with the test data.
test_labl = labl[kst:]      # ground truth of the test data
test_pred = np.zeros([n])   # Initiate the vector storing the predictions
test_pred_err = np.zeros([n])

for i in range(n):
    # to generate model for k th data point, we used data points [ 1 to (k-1) ]
    # to build the model
    k = i + kst
    train_feat = feat[0:k,:]
    train_labl = labl[0:k]
     
    # Random forest regressor
    clf2 = RandomForestRegressor(n_estimators=1000,                     # number of trees in the forest
                            criterion='mse', max_depth=None,            # use 'mean squared error' as criterion
                            min_samples_split=2, min_samples_leaf=1,    
                            max_features='auto', max_leaf_nodes=None,   # max_features=n_features.
                            bootstrap=True, oob_score=True, n_jobs=1,   # use the out of bagging data to cross-validate the prediction, and estimate the error of the model
                            random_state=None, verbose=0)
    # use the model to predict the k th data point
    clf2.fit(train_feat, train_labl)
    test_pred_err[i] = np.std(clf2.oob_prediction_ - train_labl)
    test_pred[i] = clf2.predict(test_feat[i,:])

#%% Reverse engineering to get back the next month's sales number
# Use the predicted percent increase in unadjusted sale, related to this month
# to compute the predicted next month's sales

this_monthsale = salesunadjusted[kst+12:-1]                               # this month sales
recover_pred = (test_pred *  this_monthsale /100.0) + this_monthsale      # predicted_percent_increase * thismonth + thismonth
ground_truth = (test_labl * this_monthsale /100.0) + this_monthsale       # ground truth based on percent_increase 
recover_raw = salesunadjusted[kst+12+1:]                                  # ground truth based on raw data

#%% 3). Graphing 1 without CI
##########################################################
x = np.asarray(range(len(time0)))

fig2= plt.figure()
#plt.gca().set_yscale('log')
plt.plot(x,(salesunadjusted),'b',label="Actual")
plt.plot(x[kst+13:],(recover_pred),'r-',label="Predicted")
my_xticks = np.asarray(time0)     
plt.xticks(x[0::12], my_xticks[0::12])
plt.xlabel('Year / Month')
plt.ylabel('Unadjusted Canadian Retail Sales Number')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


#%% With CI, confidence interval

SD_percent = np.std(recover_pred - ground_truth)

Upper_pred = recover_pred + 2*SD_percent
Lower_pred = recover_pred - 2*SD_percent

fig2= plt.figure()
#plt.gca().set_yscale('log')
plt.plot(x,(salesunadjusted),'b',label="Actual")
plt.plot(x[kst+13:],(recover_pred),'r-',label="Predicted")
plt.plot(x[kst+13:],(Upper_pred),'g-',label = '95% Confidence Interval of the prediction')
plt.plot(x[kst+13:],(Lower_pred),'g-')
my_xticks = np.asarray(time0)     
plt.xticks(x[0::12], my_xticks[0::12])
plt.xlabel('Year / Month')
plt.ylabel('Unadjusted Canadian Retail Sales Number')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

print 'Explained variance in the data %.2f %%'   %(100 - 100*np.var(recover_pred - ground_truth) *1.0/ np.var(ground_truth) )

print '95 %% CI in units of mean sales number is %.2f %%' % (100*2*SD_percent/np.mean(ground_truth))

#%% 4) Save data for matlab
##########################################################
from scipy.io import loadmat, savemat
savemat('/Volumes/mac/Work/JobHunt/Palantir/PE Exercise/Canada_Prediction.mat',{'recover_pred':recover_pred,'ground_truth':ground_truth,
                                                                                'test_pred':test_pred,'test_labl':test_labl,'salesunadjusted':salesunadjusted,'kst':kst})

