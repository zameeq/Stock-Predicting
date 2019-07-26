#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')



df=quandl.get('WIKI/GOOGL')
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/(df['Adj. Close']) * 100.0
df['PCT_CHANGE']=(df['Adj. Low']-df['Adj. Open'])/(df['Adj. Open']) * 100.0
df=df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]

forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out= int(math.ceil(0.01 * len(df)))

df['label']=df[forecast_col].shift(-forecast_out)


X=np.array(df.drop(['label'],1))
#X=X[:-forecast_out]
X_lately=X[-forecast_out:]


#X=preprocessing.scale(X)
X=X[:-forecast_out+0]
df.dropna(inplace=True)
y=np.array(df['label'])
y=np.array(df['label'])


print(len(X),len(y),len(df))


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.4)

clf=LinearRegression(n_jobs=1)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
prediction=clf.predict(X_lately)

print(prediction,accuracy,forecast_out)

print(df)

df['forecast']=np.nan

last_date=df.iloc[-1].name
last_unix=last_date.timestamp()  # converting the time into yymmddhhmmss
one_day=86400 # Convet the day into Seconds
next_unix=last_unix+one_day

for i in prediction:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date]=[np.nan for _i in range (len(df.columns)-1)] +[i]
    
    
df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=6)
plt.xlabel('Date')
plt.ylabel('price')
plt.show()
    




# In[ ]:




