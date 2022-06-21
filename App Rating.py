#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('googleplaystore.csv')
df.head()


# # Check for null value in data

# In[37]:


df.isnull().sum(axis=0)


# Drop Records ith nulls in an of the columns

# In[38]:


print("Frame Size before: ",df.shape)
df.dropna(subset=['Rating','Type','Content Rating','Current Ver','Android Ver'],axis=0, inplace=True)
print("rame Size After: ",df.shape)
df.isnull().sum(axis=0)


# Extract the numerical value from the column ,multiply the value by 1000 if size is mentioned in MB

# In[39]:


j=df.columns.get_loc('Size')
for i in range(0,len(df)):
    if df.iloc[i,j].lower().endswith('k'):
        df.iloc[i,j]=float(df.iloc[i,j][0:-1])
    elif df.iloc[i,j].lower().endswith('m'):
        df.iloc[i,j]=float(df.iloc[i,j][0:-1])*1000


# In[40]:


df.Size = pd.to_numeric(df.Size,errors='coerce')
df.dropna(subset=['Size'],inplace=True)
df.shape


# Rewiews is a numeric field that is loaded as string field . Convert it into numeric (int/float)

# In[41]:


df.Reviews =df.Reviews.astype('float64')
df.Reviews.dtypes


# In[42]:


df.Installs=df.Installs.str.replace(',','').str.replace('+','').astype('int64')
df.Installs.dtype


# Price field is string andhas $ symbol. Remove $sign and convert it intpo  umeric

# In[43]:


df.Price=df.Price.str.replace('$','').astype('float64')


# In[44]:


df.Price.dtype


# # Sanity Check:

# Average rating shoiuld be between 1 and 5 as onl these values are allowed on the pla store. Drop the rows  that have a value outside this range

# In[45]:


filter=(df.Rating<1) | (df.Rating>5)
print("Data Frame size :",df.shape,"count of rows containing WRONG Rating:",filter.value_counts())


# Reviews should not be more than installs as only those who installed can review the
# app. If there are any such records, drop them.

# In[46]:


rows=df[df.Installs < df.Reviews].index
df.drop(rows,axis=0,inplace=True)
df.shape


# In[47]:


rows=df[(df.Type.str.lower()=='free' )& (df.Price > 0)].index
rows


# For free apps (tpes="free"), the price should not be >0. Drop Any such rows

# In[48]:


rows=df[(df.Type.str.lower()=='free')& (df.Price >0)].index
rows


# Boxplot for Price

# In[49]:


import seaborn as sns


# In[50]:


sns.boxplot(df.Price)


# Boxplot For reviews

# In[51]:


sns.boxplot(x='Reviews',data=df)


# Histogram For rating

# In[52]:


plt.hist(df.Rating)


# Histogram For size

# In[53]:


plt.hist(df.Size)


# # Outlier Treatment:
# Price: From the box plot , it seems like there are some apps with ver high price . A price of $200 for an application on the play store is very high and suspicious!

# Check out the records with very high price is 200 indeed a high price ? Drop these as most seem to be junk apps
# 

# In[54]:


df.drop(df[df.Price>250].index, axis=0, inplace=True)
df.shape


# # Review :Vry few have very high number of reviews .these are all star apps that dont help with the analysis ans , in fact will skew it. Drop records having more than   million reviews

# In[55]:


df.drop(df[df.Reviews>200000].index, axis=0,inplace =True)
df.shape


# #installs:There seems to be some outlier inthis field too . Apps having very high number of installs should be dropped from analysis

# In[56]:


#find out the different percentiles -10,25,50,70,90,95,99 decide a threshold as cutoff for outlier ad drop records having value more than that


# In[57]:


df.Installs.quantile([0.1,0.25,0.5,0.70,0.9,0.95,0.99])


# In[58]:


df=df[df.Installs<10000000]


# In[59]:


df.shape


# #bivariat analysis: Lets look at how the available predictors relate to the variable of interest . i.e. our target variable rating .Make Scatter Plots and box plot (for character feature)to assess the relation between rating and the other feature

# Make Scatter Plot/joinplo for rating Vs Price

# In[60]:


plt.scatter(x=df.Price,y=df.Rating)


# In[61]:


#Rating Doesnt increase with price


# In[62]:


plt.scatter(x=df.Size,y=df.Rating)


# In[ ]:


#It can be obsereve that heavier apps are having higher rating


# # Make Scatter plot/joinplot for rating Vs Reviews

# In[63]:


plt.scatter(x=df.Reviews,y=df.Rating)


# #scatter plot indicats higher rating for apps having Max Reviews .But this cannit be always it could be outlier

# # Make boxplot for rating vs Content Rating

# In[65]:


plt.figure(figsize=[12,6])
sns.boxplot(y='Rating',x='Content Rating',data=df)


# In[66]:


#Not much conclusioncould be drawn as the plot is almost same for Contet Ratings, Except adults and only 18+ & unrated


# 
# # Make Boxplot for Rating vs Catgory

# In[67]:


plt.figure(figsize=[24,6])
sns.boxplot(y='Rating',x='Category',data=df)
plt.xticks(rotation=90)


# # Data Preprocessing 

# In[70]:


inp1=df
inp1.reset_index(drop=True,inplace=True)
inp1.head()


# # Review and install have some value that are still relatavel ver high . Before building a linear egression model, ou need to reduce the scre . Apply log transformation to review and install

# In[72]:


inp1.Reviews=np.log1p(inp1.Reviews)
inp1.Installs=np.log1p(inp1.Reviews)
inp1.head()


# # Drop Columns app last update , current ver, androd ver.These variable are not useful for our task

# In[75]:


inp1.drop(['App','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)
inp1.head()


# In[76]:


inp1=pd.get_dummies(inp1,columns=['Category','Genres','Content Rating','Type'],drop_first=True)
inp2=inp1.copy()
inp2.columns


# In[83]:


from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(inp2,test_size=0.3,random_state=100)


# In[84]:


y_train=df_train.pop('Rating')
X_train=df_train

y_test=df_test.pop('Rating')
X_test=df_test


# In[90]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)

from sklearn.metrics import r2_score
y_train_predict=lm.predict(X_train)
r2_score(y_train,y_train_predict)


# In[91]:


X_test_prect=lm.predict(X_test)
r2_score(y_test,X_test_prect)


# In[ ]:





# In[ ]:




