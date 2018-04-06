import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#%% FIRST MODEL FIT WITH THE WHOLE DATA SET
# IMPORT DATA

ys = np.genfromtxt('ys.csv', delimiter=',')
ts = np.genfromtxt('ts.csv', delimiter=',')

#converting numpy array to pandas DataFrame
ys = pd.DataFrame(ys)
ts = pd.DataFrame(ts)

#%% PLOT

plt.plot(ts,ys)


#%% LINEAR REGRESSION
lm = linear_model.LinearRegression()
model = lm.fit(ts, ys)

#%% PREDICTION
predictions = lm.predict(ts)

#%% COEFF
m = lm.coef_
q = lm.intercept_

#%% PLOT
plt.plot(ts, ys)
plt.plot(ts, predictions)

#%% ACTUAL PREDICTION, NO WEIGHTS
N = len(ys)
PH = 30 #prediction horizon

tp_tot = np.zeros(N)
yp_tot = np.zeros(N)


for i in range(1, N+1):
    ts_tmp = ts[0:i]
    ys_tmp = ys[0:i]
    ns = len(ys_tmp)

    # MODEL
    lm_tmp = linear_model.LinearRegression()    
    model_tmp = lm_tmp.fit(ts_tmp, ys_tmp)
    m_tmp = model_tmp.coef_
    q_tmp = model_tmp.intercept_

    # PREDICTION
    tp = ts[0][ns-1] + PH
    yp = m_tmp*tp + q_tmp
    
    
    tp_tot[i-1] = tp    
    yp_tot[i-1] = yp
    
plt.plot(tp_tot,yp_tot) 
plt.plot(ts,ys)    
    
    
#%% ACTUAL PREDICTION, WITH WEIGHTS
N = len(ys)
PH = 30 #prediction horizon
mu = 0.9

tp_tot = np.zeros(N)
yp_tot = np.zeros(N)


for i in range(1, N+1):
    ts_tmp = ts[0:i]
    ys_tmp = ys[0:i]
    ns = len(ys_tmp)
    
    weights = np.ones(ns)*mu
    for k in range(ns):
        weights[k] = weights[k]**k
    weights = np.flip(weights, 0)
        

    # MODEL
    lm_tmp = linear_model.LinearRegression()    
    model_tmp = lm_tmp.fit(ts_tmp, ys_tmp, sample_weight=weights)
    m_tmp = model_tmp.coef_
    q_tmp = model_tmp.intercept_

    # PREDICTION
    tp = ts[0][ns-1] + PH
    yp = m_tmp*tp + q_tmp
    
    
    tp_tot[i-1] = tp    
    yp_tot[i-1] = yp

#plt.Figure()    
plt.plot(tp_tot, yp_tot, '--', label='prediction') 
plt.plot(ts, ys, label='original') 
plt.legend()
#plt.show()

    

    