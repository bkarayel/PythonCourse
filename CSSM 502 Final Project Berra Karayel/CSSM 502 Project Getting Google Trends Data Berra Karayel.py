#!/usr/bin/env python
# coding: utf-8

# ### CSSM 502 Final Project Getting Google Trends Data

# In[1]:


pip install pytrends


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date
import matplotlib.dates as mdates
import seaborn as sns
import pytrends
import os
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc 
from scipy.cluster import hierarchy


# In[3]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from pytrends.request import TrendReq 


# In[5]:


# initialize
pytrends = TrendReq(hl='tr-TR', tz=360, timeout=(10,25))


# define keyword set
Searches = ["Covid-19", "kovid-19", "corona", "korona", "covid-19 belirtileri", "corona belirtileri", "kovid-19 belirtileri", "covid belirtileri", "korona belirtileri", "corona belirtisi", "korona belirtisi", "corona semptomları", "covid-19 semptomları", "covid-19 virüsü", "covid virüsü", "kovid-19 virüsü", "corona virüsü", "koronavirüs", "pandemic", "pandemi", "karantina", "quarantine", "covid","kovid", "kovid virüsü", "Omicron", "delta", "deltamicron", "omikron", "omicron belirtileri", "omikron belirtileri", "delta belirtileri", "varyant", "omicron semptomları", "omikron semptomları", "omicron öldürür mü", "omicron virüsü", "delta virüsü", "BioNTech","Turkovac","Sinovac","Sputnik","biontech aşısı","sinovac aşısı","alman aşısı","çin aşısı","türk aşısı","rus aşısı","Covid aşısı","korona aşısı","corona aşısı","covid 19 vaccine","biontech yan etkileri","alman aşısı yan etkileri","çin aşısı yan etkileri","sinovac yan etkileri", "kol ağrısı","aşı yan etkileri","corona aşısı yan etkileri","korona aşısı yan etkileri", "covid-19 aşısı yan etki", "covid-19 aşısı yan etkileri", "biotech aşısı yan etkileri", "sinovac aşısı yan etkileri", "sputnik aşısı yan etkileri", "turkovac aşısı yan etkileri", "turkovac aşısı"]
groupkeywords = list(zip(*[iter(Searches)]*1))
groupkeywords = [list(x) for x in groupkeywords]
dicti = {}
i = 1


# In[6]:


groupkeywords


# In[7]:


for trending in groupkeywords:
    pytrends.build_payload(trending, timeframe = "2021-01-13 2021-06-22", geo = "TR")
    dicti[i] = pytrends.interest_over_time()
    i+=1

results = pd.concat(dicti, axis=1)
results.columns = results.columns.droplevel(0)
results = results.drop("isPartial", axis = 1)

#this code gives the keywords based on each other, the data is not normalized, mean should be 100 for each of the keyword


# In[8]:


results.head()


# In[9]:


results.describe()


# In[10]:


results.to_csv(r'/Users/berrakarayel/Desktop/CSSM502 project google trends keyword data.csv')


# In[14]:


plt.plot(results["Covid-19"])
plt.ylabel("search amounts")
plt.xlabel("date")
plt.title("Number of Google Searches for Covid-19")


# In[12]:


plt.savefig('/Users/berrakarayel/Desktop/covid-19 searches')


# In[15]:


results["corona belirtileri"].plot()
plt.ylabel("search amounts")
plt.xlabel("date")
plt.title("Number of Google Searches for Corona Belirtileri")


# In[16]:


plt.savefig('/Users/berrakarayel/Desktop/corona belirtileri')


# In[17]:


results["aşı yan etkileri"].plot()
plt.xlabel("date")
plt.title("Number of Google Searches for Aşı Yan Etkileri")


# In[18]:


plt.savefig('/Users/berrakarayel/Desktop/aşı yan etkileri')

