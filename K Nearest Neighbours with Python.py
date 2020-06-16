#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df = pd.read_csv("C:\BACHELOR OF ENGINEERING(ISE) NOTES\Mini project\KNN\K-Nearest-Neighour-master\Classified Data",index_col=0)


# In[15]:


df.head()


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()


# In[18]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[19]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[20]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[36]:


import seaborn as sns
sns.pairplot(df,hue='TARGET CLASS')


# In[37]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# In[42]:


from sklearn.neighbors import KNeighborsClassifier


# In[43]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[44]:


knn.fit(X_train,y_train)


# In[45]:


pred = knn.predict(X_test)


# In[46]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


# In[47]:


print(confusion_matrix(y_test,pred))


# In[48]:


print(classification_report(y_test,pred))


# In[50]:


accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    accuracy_rate.append(score.mean())


# In[51]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    error_rate.append(1-score.mean())


# In[52]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[53]:


plt.figure(figsize=(10,6))
#plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
  #       markerfacecolor='red', markersize=10)
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[54]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[55]:


# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




