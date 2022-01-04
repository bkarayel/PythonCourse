#!/usr/bin/env python
# coding: utf-8

# ### Berra Karayel 0054477 CSSM502 Assingment 3

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import seaborn as sns
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings('ignore')


# In[5]:


# I have read the data set

data = pd.read_csv("/Users/berrakarayel/Desktop/cses4_cut.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

data


# In[12]:


## Classifiers without reduction and withotu pre-processing

##cross validation, I have informed the model and predicted the outcomes on new data

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)  #randomly split my data into train and test set


# Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy=cross_val_score(decision_tree, X, y, cv=cv).mean()

# Naive Bayes
naive_bayes = GaussianNB()
NB_accuracy=cross_val_score(naive_bayes, X, y, cv=cv).mean()

# Logistic Regression
LR = LogisticRegression()
LR_accuracy=cross_val_score(LR, X, y, cv=cv).mean()

# Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy=cross_val_score(LDA, X, y, cv=cv).mean()

# Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy=cross_val_score(random_forest, X, y, cv=cv).mean()

# Support Vector Machine
SVM = SVC(probability = True)
SVM_accuracy=cross_val_score(SVM, X, y, cv=cv).mean()

# K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy=cross_val_score(KNN, X, y, cv=cv).mean()

# Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy=cross_val_score(QDA, X, y, cv=cv).mean()



pd.options.display.float_format = '{:,.2f}%'.format
accuracies1 = pd.DataFrame({
    'Model'       : ["Decision Tree", "Naive Bayes", "Logistic Regression", "Linear Discriminant Analysis", "Random Forest Classifier", "Support Vector Machine", "K-Nearest Neighbors", "Quadratic Discriminant Analysis"],
    'Accuracy'    : [100*DT_accuracy, 100*NB_accuracy, 100*LR_accuracy, 100*LDA_accuracy, 100*RF_accuracy, 100*SVM_accuracy, 100*KNN_accuracy, 100*QDA_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies1.sort_values(by='Accuracy', ascending=False)


# In[11]:


## Feature Selection

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


#Selecting the features based on k highest scores

test = SelectKBest(score_func=chi2, k='all')
fit = test.fit(X, y)
kscores = fit.scores_
X_new = test.fit_transform(X, y)

# Features in descending order by score

dicts = {}
dicts=dict(zip(data.columns, kscores))
sort_dicts = sorted(dicts.items(), key=lambda x: x[1], reverse=True)

# I took 12 features with the highest score

sort_dicts[:12]


# In[13]:


# I took 12 features with the highest score

highest_12=data[['D2011','D2015','D2016','D2021','D2022','D2023','D2026','D2027','D2028','D2029','D2030','age']]

# new table after dimensionality-reduction
highest_12


# In[14]:


# data distribution of the highest 12 features

import matplotlib.pyplot as plt
plt.figure(figsize = (20, 15))
plotnumber = 1

for column in highest_12:
    if plotnumber <= 12:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(highest_12[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[15]:


from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer

# pre-processing and transforming the new data set with 12 highest features in Gaussian form
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
highest_12_trans = quantile_transformer.fit_transform(highest_12)


# In[16]:


#displaying the new data in Gaussian form after pre-processing

import matplotlib.pyplot as plt
plt.figure(figsize = (20, 15))
plotnumber = 1

for column in range(highest_12_trans.shape[1]):
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(highest_12_trans[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[19]:


## Classifiers with dimensionality-reduction and pre-processing


# Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy=cross_val_score(decision_tree, highest_12_trans, y, cv=cv).mean()

# Naive Bayes
naive_bayes = GaussianNB()
NB_accuracy=cross_val_score(naive_bayes, highest_12_trans, y, cv=cv).mean()

# Logistic Regression
LR = LogisticRegression()
LR_accuracy=cross_val_score(LR, highest_12_trans, y, cv=cv).mean()

# Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy=cross_val_score(LDA, highest_12_trans, y, cv=cv).mean()

# Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy=cross_val_score(random_forest, highest_12_trans, y, cv=cv).mean()

# Support Vector Machine
SVM = SVC(probability = True)
SVM_accuracy=cross_val_score(SVM, highest_12_trans, y, cv=cv).mean()

# K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy=cross_val_score(KNN, highest_12_trans, y, cv=cv).mean()

# Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy=cross_val_score(QDA, highest_12_trans, y, cv=cv).mean()



pd.options.display.float_format = '{:,.2f}%'.format
accuracies2 = pd.DataFrame({
    'Model'       : ["Decision Tree", "Naive Bayes", "Logistic Regression", "Linear Discriminant Analysis", "Random Forest Classifier", "Support Vector Machine", "K-Nearest Neighbors", "Quadratic Discriminant Analysis"],
    'Accuracy'    : [100*DT_accuracy, 100*NB_accuracy, 100*LR_accuracy, 100*LDA_accuracy, 100*RF_accuracy, 100*SVM_accuracy, 100*KNN_accuracy, 100*QDA_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies2.sort_values(by='Accuracy', ascending=False)


# In[21]:


## Optimizing the model and its hyperparameters

#I have chosen the top 5 classifier to find the best hyperparameter

### Random Forest Classifier

best_score=0
n_estimators= [100,200,500,1000]
criterion=['gini', 'entropy']
for i in n_estimators:
    for k in criterion:
        random_forest = RandomForestClassifier(n_estimators=i,criterion=k)
        RF_accuracy=cross_val_score(random_forest, highest_12_trans, y, cv=cv).mean()
        if RF_accuracy > best_score:
            best_score=RF_accuracy
            best_est=i
            best_cri=k
RF_accuracy=best_score
print("Best score is:",best_score,"with estimator:",best_est,"criterion:",best_cri)


### Support Vector Machine

best_score=0
clist=[0.1,1,2,5]
kernel=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed2']
for i in clist:
    for k in kernel:
        SVM = SVC(C=i,kernel=k)
        SVM_accuracy=cross_val_score(SVM, highest_12_trans, y, cv=cv).mean()
        if SVM_accuracy>best_score:
            best_score=SVM_accuracy
            best_c=i
            best_k=k
SVM_accuracy=best_score
print("Best score is:",best_score,"with c:",best_c,"kernel:",k)



### Linear Discriminant Analysis
        
best_score=0        
solver=['svd', 'lsqr', 'eigen']
for i in solver:    
    LDA = LinearDiscriminantAnalysis(solver=i)
    LDA_accuracy=cross_val_score(LDA, highest_12_trans, y, cv=cv).mean()
    if LDA_accuracy>best_score:
        best_score=LDA_accuracy
        best_solver=i
LDA_accuracy=best_score
print("Best score is:",best_score,"with solver:",best_solver)


        
### Logistic Regression

best_score=0     
penalty=['l1', 'l2', 'elasticnet', 'none']
for i in penalty:
    LR = LogisticRegression(penalty=i)
    LR_accuracy=cross_val_score(LR, highest_12_trans, y, cv=cv).mean()
    if LR_accuracy > best_score:
        best_score=LR_accuracy
        best_p=i
LR_accuracy=best_score
print("Best score is:",best_score,"with penalty",best_p)



### K-Nearest Neighbors
        
best_score=0
for i in range(2,10):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN_accuracy=cross_val_score(KNN, highest_12_trans, y, cv=cv).mean()
    if KNN_accuracy > best_score:
        best_score=KNN_accuracy
        best_n=i
KNN_accuracy=best_score
print("Best score is:",best_score,"with number of neighbors:",best_n)




pd.options.display.float_format = '{:,.2f}%'.format
accuracies3 = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors',],
    'Accuracy'    : [100*LR_accuracy, 100*SVM_accuracy, 100*LDA_accuracy, 100*RF_accuracy, 100*KNN_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies3.sort_values(by='Accuracy', ascending=False)


# In[22]:


### Final Results


print("Classifiers without reduction:")
display(accuracies1.sort_values(by='Accuracy', ascending=False))
print("Classifiers with dimensionality-reduction and pre-processing:")
display(accuracies2.sort_values(by='Accuracy', ascending=False))
print("After optimizing the model and its hyperparameters:")
display(accuracies3.sort_values(by='Accuracy', ascending=False))

