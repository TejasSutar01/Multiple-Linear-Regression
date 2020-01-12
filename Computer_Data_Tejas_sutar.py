# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:25:03 2019

@author: tejas
"""

pwd
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=Computer_Datacsv

#Label Encoder for cd, multi,premium
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

select_column=['cd','multi','premium']

le.fit(df[select_column].values.flatten())

df[select_column]= df[select_column].apply(le.fit_transform)



#corelation matrix
df.corr() 
corelation_values=df.corr() 
 
import seaborn as sns
sns.pairplot(df)
# scatter plot
 
#here colinearity is high between hd & ram but scatter plot shows no any linearity.

#Split the data into train and test .

from sklearn.model_selection import train_test_split

train_data,test_data=train_test_split(df)
#drop the column no. 1

train_data=train_data.reset_index()
test_data=test_data.reset_index()

train_data1=df.drop(["Unnamed: 0"], axis=1)
test_data1=df.drop(["Unnamed: 0"], axis=1)

import statsmodels.formula.api as sf

df.head(2)

#Building the regression model w.r.t. t training data.

m1=sf.ols("price~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=train_data1).fit()

m1.summary()   #R-SQUARE=0.776
#Here r-square is less than 0.8 ,we need some transformations for r-square value more than 0.8.

#check with the vif value.
rsq=sf.ols("price~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=train_data1).fit().rsquared

rsq_final=1/(1-rsq)
#vif value is less than 10.i.e 4.53

import statsmodels.api as sm

sm.graphics.influence_plot(m1)

sm.graphics.plot_partregress_grid(m1)

#Transformations for better value of r-squared.
#Take the sqrt for all input variables.
trans_m1=sf.ols("price~np.sqrt(speed)+np.sqrt(hd)+np.sqrt(ram)+np.sqrt(screen)+cd+multi+premium+np.sqrt(ads)+trend",data=train_data1).fit()
trans_m1.summary()  #r-squared=0.793

#providing the sqrt to output variable w.r.t. to output 
trans_m2=sf.ols("np.sqrt(price)~np.sqrt(speed)+np.sqrt(hd)+np.sqrt(ram)+np.sqrt(screen)+cd+multi+premium+np.sqrt(ads)+trend",data=train_data1).fit()
trans_m2.summary()    #r-squared=0.807


#taking log 
trans_m3=trans_m1=sf.ols("np.log(price)~np.log(speed)+np.log(hd)+np.log(ram)+np.log(screen)+cd+multi+premium+np.log(ads)+trend",data=train_data1).fit()
trans_m3.summary()     #r-squared=0.791



#taking log for price & sqrt for speed and hd
trans_m4=sf.ols("np.log(price)~np.sqrt(speed)+np.sqrt(hd)+ram+screen+cd+multi+premium+ads+trend",data=train_data1).fit()
trans_m4.summary()    #0.805


#Taking log to price,sqrt to speed,hd & qudratic to speed & hd
trans_m5=trans_m1=sf.ols("np.log(price)~np.sqrt(speed)+(speed*speed)+np.sqrt(hd)+(hd*hd)+(ram)+(screen)+cd+multi+premium+ads+trend",data=train_data1).fit()
trans_m5.summary()        #0.814
             

#taking th model5 because r squared is high i.e 0.815 with rmse value of train=244.3 & test rmse =249.3
trans_m5=sf.ols("np.log(price)~np.sqrt(speed)+(speed*speed)+np.sqrt(hd)+(hd*hd)+(ram)+(screen)+cd+multi+premium+ads+trend",data=train_data1).fit()
trans_m5.summary()                   #0.814


#train data
final_train_pred=trans_m5.predict(train_data1)
final_train_pred1=np.exp(final_train_pred)

#train residual
train_res=train_data1['price']-final_train_pred1
train_res

#train rmse
train_rsme=np.sqrt(np.mean(train_res*train_res))
train_rsme                    #244.60

#test pred
final_test_pred=trans_m5.predict(test_data1)
final_test_pred1=np.exp(final_test_pred)

#test residuals
final_test_res=test_data1['price']-final_test_pred1

#test rmse
final_test_rmse=np.sqrt(np.mean(final_test_res*final_test_res))
   

#testing the finalised model with original dataset

df1=df.drop(["Unnamed: 0"],axis=1)
final=sf.ols("np.log(price)~np.sqrt(speed)+(speed*speed)+np.sqrt(hd)+(hd*hd)+(ram)+(screen)+cd+multi+premium+ads+trend",data=df1).fit()
final.summary()                     #0.814

#prediction model
final_pred_log=final.predict(df1)
final_pred=np.exp(final_pred_log)


#checking the linearity with plotting the scatter plot
plt.scatter(df1['price'],final_pred);plt.xlabel("Actual Values");plt.ylabel("fitted values")

plt.scatter(final_pred,final.resid_pearson, c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")
#somehow homoscadasticity is present

#check for normality
plt.hist(final.resid_pearson)
#Data is normally distributed

import pylab
import scipy.stats as sc

#checking the final model it shows the data points are within the points.
sc.probplot(final.resid_pearson, dist="norm", plot=pylab)