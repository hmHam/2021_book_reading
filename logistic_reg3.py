import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import sklearn
import requests,zipfile
import io

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
res = requests.get(url).content
iris = pd.read_csv(io.StringIO(res.decode("utf-8")),header = None)

iris[4] = iris[4].map(lambda x:0 if x =='Iris-setosa' else 2 if x =='Iris-virginica' else 1)
ris = iris.sample(frac=1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X = iris[[0,1,2,3]]
y = iris[4]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0)

model= LogisticRegression()
model.fit(X_train,y_train)

print('正解率(train):{:.3f}'.format(model.score(X_train,y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test,y_test)))