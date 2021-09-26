#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import preprocessing , model_selection, neighbors,utils


# In[4]:


import pandas as pd
data=pd.read_csv(r"C:\Users\user\Desktop\horse-colic.data", sep=' ')
data.columns = ['Surgery','Age','Hospital_Number','rectal_temperature','pulse','respiratory_rate','temperature_of_extremities','peripheral_pulse','mucous_membranes','capillary_refill_time','pain','peristalsis','abdominal_distension','nasogastric_tube','nasogastric_reflux','nasogastric_reflux_PH','rectal_examination','abdomen','packed_cell_volume','total_protein','abdominocentesis_appearance','abdomcentesis_total_protein','outcome','surgical_lesion','type_of_lesion1','type_of_lesion2','type_of_lesion3','cp_data']
data.reset_index(inplace=True)
data.set_axis(['Surgery','Age','Hospital_Number','rectal_temperature','pulse','respiratory_rate','temperature_of_extremities','peripheral_pulse','mucous_membranes','capillary_refill_time','pain','peristalsis','abdominal_distension','nasogastric_tube','nasogastric_reflux','nasogastric_reflux_PH','rectal_examination','abdomen','packed_cell_volume','total_protein','abdominocentesis_appearance','abdomcentesis_total_protein','outcome','surgical_lesion','type_of_lesion1','type_of_lesion2','type_of_lesion3','cp_data','unknown'], 
                    axis='columns', inplace=True)
data


# In[5]:


data.isnull().sum()


# In[6]:


data.drop('Hospital_Number',inplace=True,axis=1)
data.drop('pulse',inplace=True,axis=1)
data.drop('respiratory_rate',inplace=True,axis=1)
data.drop('temperature_of_extremities',inplace=True,axis=1)
data.drop('peripheral_pulse',inplace=True,axis=1)
data.drop('mucous_membranes',inplace=True,axis=1)
data.drop('capillary_refill_time',inplace=True,axis=1)
data.drop('pain',inplace=True,axis=1)
data.drop('peristalsis',inplace=True,axis=1)
data.drop('abdominal_distension',inplace=True,axis=1)
data.drop('nasogastric_tube',inplace=True,axis=1)
data.drop('nasogastric_reflux',inplace=True,axis=1)
data.drop('nasogastric_reflux_PH',inplace=True,axis=1)
data.drop('rectal_examination',inplace=True,axis=1)
data.drop('abdominocentesis_appearance',inplace=True,axis=1)
data.drop('abdomen',inplace=True,axis=1)
data.drop('packed_cell_volume',inplace=True,axis=1)
data.drop('total_protein',inplace=True,axis=1)
data.drop('outcome',inplace=True,axis=1)
data.drop('surgical_lesion',inplace=True,axis=1)
data.drop('type_of_lesion1',inplace=True,axis=1)
data.drop('type_of_lesion2',inplace=True,axis=1)
data.drop('type_of_lesion3',inplace=True,axis=1)
data.drop('cp_data',inplace=True,axis=1)


# In[7]:


data.drop('unknown',inplace=True,axis=1)


# In[8]:


data


# In[9]:


data['Surgery'].unique()


# In[10]:


for col in data:
    print(col)
    print(data[col].unique())


# In[15]:


data['Surgery'].replace({'?':-999},inplace=True)
data['Surgery'].unique()
data['rectal_temperature'].replace({'?':-999},inplace=True)
data['rectal_temperature'].unique()
data['abdomcentesis_total_protein'].replace({'?':-999},inplace=True)
data['abdomcentesis_total_protein'].unique()


# In[16]:


data['Surgery']=data['Surgery'].astype(int)
data['rectal_temperature']=data['rectal_temperature'].astype(float)
data['abdomcentesis_total_protein']=data['abdomcentesis_total_protein'].astype(float)
data.dtypes


# In[17]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X.shape)
print(y.shape)


# In[19]:


clf= neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train.astype('int'))


# In[20]:


prediction =clf.predict(X_test)
accuracy= clf.score(X_test,y_test.astype('int'))
print("accuracy",accuracy)

