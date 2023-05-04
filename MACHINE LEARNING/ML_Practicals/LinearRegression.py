#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[4]:


x = np.array([1,2,3,4,5])


# In[5]:


y = np.array([7,14,15,18,19])


# In[6]:


x_mean = np.mean(x)
y_mean = np.mean(y)


# In[7]:


x_mean


# In[8]:


y_mean


# In[9]:


np.sum(x*y)


# In[10]:


n = np.size(x)
Sxy = np.sum(x*y) - n*x_mean*y_mean
Sxx = np.sum(x*x) - n * x_mean*x_mean


# In[11]:


Q1 = Sxy/Sxx
Q0 = y_mean - Q1*x_mean


# In[12]:


Q1


# In[13]:


Q0


# In[14]:


plt.scatter(x,y)
plt.xlabel('Intependent variable X')
plt.ylabel('Dependent variable y')


# In[15]:


y_pred = Q0 + Q1*x


# In[16]:


y_pred


# In[17]:


y


# In[20]:


plt.scatter(x,y,color = 'red')
plt.plot(x,y_pred, color = 'green')
plt.xlabel('X')
plt.ylabel('y')


# In[21]:


error = y- y_pred


# In[22]:


error


# In[23]:


se = np.sum(error**2)


# In[24]:


se


# In[25]:


mse = se/n


# In[26]:


mse


# In[27]:


rmse = np.sqrt(mse)


# In[28]:


rmse


# In[29]:


reg = LinearRegression()


# In[31]:


x


# In[32]:


x = x.reshape(-1,1)


# In[33]:


x


# In[34]:


reg.fit(x,y)


# In[35]:


scik_pred = reg.predict(x)


# In[36]:


scik_pred


# In[37]:


y


# In[38]:


Scik_mse = mean_squared_error(y,scik_pred)


# In[39]:


Scik_mse


# In[40]:


mse


# In[41]:


from sklearn import datasets


# In[45]:


diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y=True)


# In[46]:


diabetes_X


# In[47]:


diabetes_Y


# In[48]:


reg.coef_    (Q1,Q2,Q3)


# In[49]:


reg.intercept_    Q0


# In[50]:


Q0


# In[51]:


Q1


# In[ ]:




