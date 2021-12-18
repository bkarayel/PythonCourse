#!/usr/bin/env python
# coding: utf-8

# ### Integrated, Plot ans Stats Berra Karayel 0054477 Assingment 2

# In[1]:


import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


def lin_regress(indep, dep):
                      
    coefficient=None
    Std_error=None
    T1=None
    T2 = None

    #list-wise deletion of missing values
    
    indep_clean = indep.replace("Na", "")     
    indep_clean.dropna(inplace=True)  #making them numpy arrays
    
    dep_clean = dep.replace("Na", "")
    dep_clean.dropna(inplace=True)

    # adding 1 column (for intercept) to the X matrix
    onecolumn = np.ones((indep_clean.shape[0], 1))
    indep = np.concatenate((onecolumn, indep_clean), axis=1)
    
    # calculating coefficients, solving linear matrix equation
    coefficient = np.linalg.solve(np.dot(indep.T, indep), np.dot(indep.T, dep))  
    
    #calculating estimated value of dependent variable in our linear regression
    n = indep.shape[1] #input array
    h = np.ones((indep.shape[0], 1))
    θ = coefficient.reshape(1, n)  
    for i in range(0, indep.shape[0]):
        h[i] = float(np.matmul(θ, indep[i]))
    est_value_dep = h.reshape(indep.shape[0])
    
    #calculating residuals
    residuals = np.subtract(dep, est_value_dep)
    sum_squares = np.dot(residuals.T, residuals)   #.T gives transpose
    how_many_sample = X.shape[0]
    denominator = how_many_sample - n
    
    #coefficient variance
    sigma_squared = np.true_divide(sum_squares, denominator)
    inverse = np.linalg.inv(np.dot(indep.T, indep))
    coefficient_variance = np.dot(sigma_squared, inverse)
    var_coef = np.diag(coefficient_variance)

    # Standard error of coefficients
    Std_error = np.sqrt(var_coef)
    
    # T-statistics, confidence interval
    t = stats.t.ppf(0.975, denominator)
    t_part = t * Std_error

    # calculating credible intervals, t2 upper, t1 lower
    upper = np.add(θ, t_part)
    T2 = upper.reshape(-1, )
    lower = np.subtract(θ, t_part)
    T1 = lower.reshape(-1, )
    
    return coefficient, Std_error, T1, T2, indep_clean, dep


# In[13]:


df=pd.read_csv('/Users/berrakarayel/Desktop/2015.csv')
print(df.columns)
df_updated=df[['Happiness Score','Health (Life Expectancy)']]
X = df[['Happiness Score']]
y = df['Health (Life Expectancy)']
coefficient, Std_error, T1, T2, indep_clean, dep_clean = lin_regress(X,y)
df_new = pd.DataFrame(list(zip(coefficient, Std_error, T1, T2)), columns =['B_hat', 'Standard Error', 'Lower 95%', 'Upper 95%']) 
plt.style.use("seaborn-whitegrid")
sns.jointplot(x='Happiness Score',y='Health (Life Expectancy)',data=df_updated,kind="reg")
plt.show()  
print(coefficient)
print(Std_error)
print(T1)
print(T2)

