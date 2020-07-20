import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn import tree,svm
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

dataframe =pd.read_csv("forest3.csv")
a=corr_matrix=dataframe.corr()
print(a)
b=corr_matrix["area"].sort_values(ascending=False)
print(b)
data = np.array(dataframe)

x = data[0:, 4:-1]
y = data[0:, -1]
y = y.astype('float')
#x = x.astype('float')

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=5)


print("svm")

svm_model =LogisticRegression()
svm_model.fit(x_train,y_train)
a = svm_model.score(x_test,y_test)
print(a)







pickle.dump(svm_model, open("model.pkl", "wb"))
