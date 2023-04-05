#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib')


# In[2]:


pip install pandas-datareader


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[4]:


import tensorflow as tf
from tensorflow import keras


# In[5]:


start='2010-01-01'
end='2022-06-30'

df=data.DataReader('SBIN','yahoo',start,end)
df.head()


# In[6]:


df.tail()


# In[7]:


df = df.reset_index()
df.head()


# In[8]:


df= df.drop(['Date','Adj Close'],axis=1)
df.head()


# In[9]:


plt.plot(df.Close)


# In[10]:


ma100=df.Close.rolling(100).mean()
ma100


# In[11]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')


# In[12]:


ma200=df.Close.rolling(200).mean()
ma200


# In[13]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')


# In[14]:


df.shape


# In[15]:


#slpitting data into training and testing

data_training =pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)


# In[16]:


data_training.head()


# In[17]:


data_testing.head()


# In[18]:


get_ipython().system('pip install scikit-learn')


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[20]:


data_training_array=scaler.fit_transform(data_training)
data_training_array


# In[21]:


data_training_array.shape


# In[22]:


x_train=[]
y_train=[] 

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    

x_train,y_train=np.array(x_train),np.array(y_train)


# In[23]:


x_train.shape


# In[24]:


# ML model


# In[25]:


from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential


# In[26]:


model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,
               input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu',return_sequences=True))
model.add(Dropout(0.5))

model.add(Dense(units=1))


# In[27]:


model.summary()


# In[28]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)


# In[29]:


model.save('keras_model.h5')


# In[30]:


#testing the data
data_testing.head()


# In[31]:


past_100_days = data_training.tail(100)


# In[32]:


final_df=past_100_days.append(data_testing,ignore_index=True)


# In[33]:


final_df.head()


# In[34]:


input_data=scaler.fit_transform(final_df)
input_data


# In[35]:


input_data.shape


# In[36]:


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


# In[37]:


x_test,y_test=np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[48]:


# MAKING PREDCTIONS

y_predicted=model.predict(x_test)


# In[49]:


y_predicted.shape


# In[40]:


y_test


# In[41]:


y_predicted


# In[42]:


scaler.scale_


# In[43]:


scale_factor=1/0.00682769
y_predicted= y_predicted * scale_factor
y_test=y_test * scale_factor


# In[44]:


y_pred=df.Close.rolling(40).mean()
y_pred


# In[45]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
# plt.plot(df.Close,label='Original price')
# plt.plot(y_pred,'r',label = 'Predicted Price')
plt.plot(y_predicted,'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:




