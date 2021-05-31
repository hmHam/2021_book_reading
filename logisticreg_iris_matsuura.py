#!/usr/bin/env python
# coding: utf-8

# In[2]:


import logisticreg
import csv
import numpy as np
import random


n_test = 90
X = []
y = []
with open("iris.data") as fp:
    for row in csv.reader(fp):
        if row[4] == "Iris-setosa":
            y.append(0)
            X.append(row[:-1])
        elif row[4] == "Iris-versicolor":
            i = random.randint(0, len(y))
            y.insert(i,1)
            X.insert(i,row[:-1])
        else:
            break
        
y = np.array(y, dtype=np.float64)
X = np.array(X, dtype=np.float64)
y_train = y[:-n_test]
X_train = X[:-n_test]
y_test = y[-n_test:]
X_test = X[-n_test:]
model = logisticreg.LogisticRegression(tol=0.01)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
n_hits = (y_test == y_predict).sum()
print("Accuracy: {}/{} = {}".format(n_hits, n_test, n_hits/n_test))


# In[ ]:




