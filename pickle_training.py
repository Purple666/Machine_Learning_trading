from sklearn import preprocessing, model_selection, svm
from sklearn import linear_model

import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import MetaTrader5 as mt5
import os
import time
import datetime
import IPython
import IPython.display
import seaborn as sns

start_time = time.time()

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

n_samples = 90000
    
COIN_rates = mt5.copy_rates_from_pos("USDJPY", mt5.TIMEFRAME_M1, 0, n_samples)
df = pd.DataFrame(COIN_rates)

df.set_index('time',drop=True,inplace=True)
df.drop(['spread','real_volume','open'],axis=1,inplace=True)

forecast_col = 'close'
forecast_out = 60
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#----------------------------- to write pickles -----------------------------#
# 1. ARDRegression
ARDRegression_clf = linear_model.ARDRegression()
ARDRegression_clf.fit(X_train, y_train)
with open('ARDRegression_clf.pickle','wb') as f:
    pickle.dump(ARDRegression_clf, f)
a = round(ARDRegression_clf.score(X_test, y_test),2)
print(f"ARDRegression {a}")

# 2. HuberRegressor
HuberRegressor_clf = linear_model.HuberRegressor()
HuberRegressor_clf.fit(X_train, y_train)
with open('HuberRegressor_clf.pickle','wb') as f:
    pickle.dump(HuberRegressor_clf, f)
b = round(HuberRegressor_clf.score(X_test, y_test),2)
print(f"HuberRegressor {b}")

# 3. LinearRegression
LinearRegression_clf = linear_model.LinearRegression()
LinearRegression_clf.fit(X_train, y_train)
with open('LinearRegression_clf.pickle','wb') as f:
    pickle.dump(LinearRegression_clf, f)
c = round(LinearRegression_clf.score(X_test, y_test),2)
print(f"LinearRegression {c}")

# 4. SupportVectorRegression
SupportVectorRegression_clf = svm.SVR()
SupportVectorRegression_clf.fit(X_train,y_train)
with open ('SupportVectorRegression_clf.pickle','wb') as f:
    pickle.dump(SupportVectorRegression_clf, f)
d = round(SupportVectorRegression_clf.score(X_test,y_test),2)
print(f"SupportVectorRegression {d}")

# 5. BayesianRidge
BayesianRidge_clf = linear_model.BayesianRidge()
BayesianRidge_clf.fit(X_train, y_train)
with open('BayesianRidge_clf.pickle','wb') as f:
    pickle.dump(BayesianRidge_clf, f)
e = round(BayesianRidge_clf.score(X_test, y_test),2)
print(f"BayesianRidge {e}")

# 6. PassiveAggressiveRegressor
PassiveAggressiveRegressor_clf = linear_model.PassiveAggressiveRegressor()
PassiveAggressiveRegressor_clf.fit(X_train, y_train)
with open('PassiveAggressiveRegressor_clf.pickle','wb') as f:
    pickle.dump(PassiveAggressiveRegressor_clf, f)
k = round(PassiveAggressiveRegressor_clf.score(X_test, y_test),2)
print(f"PassiveAggressiveRegressor {k}")

# 7. RANSACRegressor
RANSACRegressor_clf = linear_model.RANSACRegressor()
RANSACRegressor_clf.fit(X_train, y_train)
with open('RANSACRegressor_clf.pickle','wb') as f:
    pickle.dump(RANSACRegressor_clf, f)
g = round(RANSACRegressor_clf.score(X_test, y_test),2)
print(f"RANSACRegressor {g}")

# 8. RidgeCV
RidgeCV_clf = linear_model.RidgeCV()
RidgeCV_clf.fit(X_train, y_train)
with open('RidgeCV_clf.pickle','wb') as f:
    pickle.dump(RidgeCV_clf, f)
h = round(RidgeCV_clf.score(X_test, y_test),2)
print(f"RidgeCV {h}")

# 9. SGDRegressor
SGDRegressor_clf = linear_model.SGDRegressor()
SGDRegressor_clf.fit(X_train, y_train)
with open('SGDRegressor_clf.pickle','wb') as f:
    pickle.dump(SGDRegressor_clf, f)
i = round(SGDRegressor_clf.score(X_test, y_test),2)
print(f"SGDRegressor {i}")

# 10. TheilSenRegressor
TheilSenRegressor_clf = linear_model.TheilSenRegressor()
TheilSenRegressor_clf.fit(X_train, y_train)
with open('TheilSenRegressor_clf.pickle','wb') as f:
    pickle.dump(TheilSenRegressor_clf, f)
j = round(TheilSenRegressor_clf.score(X_test, y_test),2)
print(f"TheilSenRegressor {j}")

consensus = a+b+c+d+e+k+g+h+i+j
print(f"\nnumber of samples was: ",n_samples)
print("consensus is: ",round(consensus*10,2),"%")

