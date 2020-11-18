#!/usr/bin/env python
# coding: utf-8

# In[22]:


#A minimum of 5 variables in your dataset used during your analysis
#Read data set

import pandas as pd
boston_orig = pd.read_csv('boston_house_prices.csv')

boston_orig.columns


# In[23]:


boston_orig.shape


# In[24]:


# select columns to keepplot o

#CRIM per capita crime rate by town
#ZN proportion of residential land zoned for lots over 25,000 sq.ft. 
#INDUS proportion of non-retail business acres per town
#CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#NOX nitric oxides concentration (parts per 10 million) 
#RM average number of rooms per dwelling 
#AGE proportion of owner-occupied units built prior to 1940 
#DIS weighted distances to five Boston employment centres 
#RAD index of accessibility to radial highways 
#TAX full-value property-tax rate per $10,000 
#PTRATIO pupil-teacher ratio by town - B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
#LSTAT % lower status of the population 
#MEDV Median value of owner-occupied homes in $1000's


# In[25]:


cols = ['MEDV', 'CRIM', 'RM', 'AGE', 'TAX', 'RAD']
boston = boston_orig[cols]
boston.shape


# In[26]:


#column name headers
boston.head()


# In[27]:


#verifying the data types of each column
boston.dtypes


# In[28]:


#checking for missing data
pd.isna(boston).sum()


# In[29]:


#include a histogram of each of the 5 variables 

for col in cols:
    boston.hist(column=col)


# In[31]:


boston.mean()


# In[32]:


#standard deviaiton
boston.std()


# In[33]:


#PMF using pandas
import seaborn as sns
probabilities = boston['MEDV'].value_counts(normalize=True)    
sns.barplot(probabilities.index, probabilities.values)


# In[34]:


import seaborn as sns
probabilities = boston['RM'].value_counts(normalize=True)    
sns.barplot(probabilities.index, probabilities.values)


# In[35]:


#mean = boston.mean()
#var = spread
var = boston.var()
var


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


#Cumulative distrribution function (cdf for MEDV)
def EvalCdf(t, x):
    """ t is series, x is value to calc CDF on """
    count = 0.0
    for value in t:
        if value <= x:
            count += 1 
    prob = count / len(t)
    return prob


# In[37]:


# cacl CDF for MEDB 5 .. 50
cdf = []
x_values = []
for x in range(5,55,2):
    x_values.append(x)
    medv_cdf = EvalCdf(boston['MEDV'], x)
    cdf.append(medv_cdf)
print(cdf)


# In[38]:


# plot 


# In[ ]:





# In[39]:


#scatter plot for CDF MEDV 

import matplotlib.pyplot as plt
plt.scatter(x_values, cdf)


# In[40]:


#RM
cdf = []
x_values = []
for x in range(5,55,2):
    x_values.append(x)
    RM_cdf = EvalCdf(boston['RM'], x)
    cdf.append(RM_cdf)
print(cdf)


# In[41]:


plt.scatter(x_values, cdf)


# In[53]:


plt.scatter(boston['MEDV'], boston['RM'])


# In[55]:


plt.scatter(boston['MEDV'], boston['CRIM'])

from pandas.plotting import scatter_matrix
scatter_matrix(boston, figsize=(25,25))


# In[56]:


cols


# In[ ]:





# In[51]:


# Testing dff of means of MEDV and RM
abs( boston['MEDV'].mean() - boston['RM'].mean() )


# In[52]:


# linear regr
from sklearn import linear_model

X = boston[['RM', 'AGE']]
y = boston['MEDV']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)


# In[53]:


dir(regr)


# In[ ]:


regr


# In[55]:


regr.get_params()


# In[58]:


import statsmodels.api as sm
X = boston[["RM", 'AGE']]
y = boston["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[ ]:




