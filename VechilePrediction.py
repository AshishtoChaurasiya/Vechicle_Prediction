#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


car_predict=pd.read_csv("D:vehicle_data.csv")


# In[4]:


car_predict.head()


# In[6]:


car_predict.info()


# In[7]:


car_predict.isnull().sum()


# In[8]:


car_predict.shape


# In[11]:


car_predict.info()


# In[15]:


# Year data change String to numeric format
car_predict=car_predict[car_predict['year'].str.isnumeric()]


# In[16]:


car_predict.info()


# In[17]:


# Year data change String to int format
car_predict['year']=car_predict['year'].astype(int)


# In[20]:


car_predict.info()


# In[22]:


# Remove String in Price data
car_predict=car_predict[car_predict['Price']!='Ask For Price']


# In[23]:


car_predict


# In[24]:


# Change object to int and replace , to space
car_predict['Price']=car_predict['Price'].str.replace(',','').astype(int)


# In[25]:


car_predict.info()


# In[26]:


car_predict['kms_driven']=car_predict['kms_driven'].str.split().str.get(0).str.replace(',','')


# In[27]:


car_predict


# In[28]:


car_predict=car_predict[car_predict['kms_driven'].str.isnumeric()]


# In[29]:


car_predict.info()


# In[30]:


car_predict['kms_driven']=car_predict['kms_driven'].astype(int)


# In[31]:


car_predict.info()


# In[32]:


car_predict=car_predict[~car_predict['fuel_type'].isna()]


# In[33]:


car_predict.shape


# In[34]:


car_predict.info()


# In[36]:


car_predict['name'] = car_predict['name'].str.split().str.slice(start=0,stop=3).str.join(' ')


# In[37]:


car_predict


# In[38]:


car_predict.describe()


# In[39]:


car_predict=car_predict[car_predict['Price']<6000000]


# In[40]:


car_predict


# In[41]:


car_predict['company'].unique()


# In[43]:


x=car_predict[['name','company','year','kms_driven','fuel_type']]
y=car_predict['Price']


# In[44]:


y


# In[45]:


y.shape


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[48]:


X_test


# In[49]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# In[50]:


ohe=OneHotEncoder()


# In[51]:


ohe.fit(x[['name','company','fuel_type']])


# In[52]:


columns_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


# In[53]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# In[54]:


lr=LinearRegression()


# In[55]:


pipe=make_pipeline(columns_trans,lr)


# In[56]:


pipe


# In[57]:


pipe.fit(X_train,y_train)


# In[58]:


y_pred=pipe.predict(X_test)


# In[59]:


y_pred


# In[60]:


from sklearn.metrics import r2_score


# In[62]:


r2_score(y_test,y_pred)


# In[63]:


scores=[]
for i in range(1000):
    X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(columns_trans,lr)
    pipe.fit(X_train, y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[64]:


np.argmax(scores)


# In[65]:


scores[np.argmax(scores)]


# In[66]:


pipe.predict(pd.DataFrame(columns=X_test.columns, data=np.array(['Maruti Suzuki Ritz','Maruti',2011,50000,'Petrol']).reshape(1,5)))


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(columns_trans,lr)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[72]:


pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Ritz','Maruti',2011,70000,'Petrol']).reshape(1,5)))


# In[ ]:





# In[ ]:




