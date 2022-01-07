import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
from tabulate import tabulate
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from numpy import linalg as LA
from scipy.stats import chi2


#Loading in the data
df = pd.read_csv('google_stock.csv')

#Defining the dependent variable
date = df['Date']
stock = df['Close']

#Checking for missing data or NaN
print(df.isnull().values.any())
print(stock.isnull().values.any())

#Correlation Matrix
corr = df.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),square=True)
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

#Preliminary visualization of raw data
fig, ax = plt.subplots(figsize = (10,9))
plt.plot(date,stock)
plt.xticks(rotation=45)
ax.set_xticks(ax.get_xticks()[::200])
plt.xlabel("Time")
plt.ylabel("Dollars ($USD)")
plt.title("Google Stock Close Price")
plt.show()

#ADF-test
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
       print('\t%s: %.3f' % (key, value))


stock_acf = ADF_Cal(stock)
print(stock_acf)

#Calculating ACF
def calc_autocorrelation_coef(y,t_lag,title):
    mean = sum(y)/len(y)
    denominator = 0

    for value in y:
        denominator = denominator + (value - mean)**2

    numerator = 0
    numerator_list = []
    for x in range(0, t_lag+1):
        for i in range(x,len(y)):
            numerator = numerator + (y[i] - mean)*(y[i-x]-mean)
        numerator_list.append(numerator)
        numerator = 0

    r_list = [x/denominator for x in numerator_list]

    x = np.linspace(-t_lag, t_lag, t_lag*2+1)
    # Plotting ACF of Residuals for AR(1)
    fig = plt.figure(figsize=(16, 8))
    r_list_r_yy = r_list[::-1]
    r_list_Ry = r_list_r_yy[:-1] + r_list
    r_list_Ry_np = np.array(r_list_Ry)
    plt.stem(x, r_list_Ry_np, use_line_collection=True)
    plt.title(title)
    plt.ylabel('ACF')
    plt.xlabel('Lag')
    m_pred = 1.96 / math.sqrt(len(y))
    plt.axhspan(-m_pred, m_pred, alpha=.1, color='black')
    plt.show()

    return r_list

#Plotting ACF and PACF using statsmodel
acf = sm.tsa.stattools.acf(stock, nlags=20)
pacf = sm.tsa.stattools.pacf(stock, nlags=20)

fig = plt.figure(figsize=(6, 10))
fig.tight_layout(pad=4)
plt.subplot(2, 1, 1)
plot_acf(stock, ax=plt.gca(), lags=20, title="ACF of Raw Data")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.subplot(2, 1, 2)
plot_pacf(stock, ax=plt.gca(), lags=20, title="PACF of Raw Data")
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.show()

#Plotting ACF
raw_data_acf = calc_autocorrelation_coef(stock,20,'ACF of Raw Data')

#Calculating Rolling Mean and Variance of Raw Data
def cal_rolling_var_mean(df,dependent_variable):
    sales_mean = 0
    sales_variance = 0

    mean_sales_time = []
    var_sales_time = []

    for i in range(1, len(df)+1):
        rows = df.head(i)

        seq_sales = rows[dependent_variable].values[-1]
        seq_sales_var = rows[dependent_variable].values

        sales_mean = sales_mean + seq_sales
        mean_sales_time.append(sales_mean / i)

        if i == 1:
            var_sales_time.append(seq_sales_var[-1])
        elif i > 1:
            var_sales_time.append(statistics.variance(seq_sales_var))

    return (mean_sales_time,var_sales_time)

var_sales_time = cal_rolling_var_mean(df,'Close')[1]
mean_sales_time = cal_rolling_var_mean(df,'Close')[0]

#Close Price Rolling Variance
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(date,var_sales_time, label = "Rolling Variance")
plt.xticks(rotation=45)
ax.set_xticks(ax.get_xticks()[::200])
plt.xlabel("Year")
plt.ylabel("Dollars ($USD)")
plt.legend()
plt.title("Google Stock Close Price Rolling Variance")
plt.show()

#Close Price Rolling Mean
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(date,mean_sales_time, label = "Rolling Mean")
plt.xticks(rotation=45)
ax.set_xticks(ax.get_xticks()[::200])
plt.xlabel("Year")
plt.ylabel("Dollars ($USD)")
plt.legend()
plt.title("Google Stock Close Price Rolling Mean")
plt.show()

#Logarithmic Transformation and Differencing
stock_log1 = np.log(stock)
stock_log1 = stock_log1[1:]

print(ADF_Cal(stock_log1))

stock_log2 = np.log(stock_log1)
stock_log2 = stock_log2[1:]

print(ADF_Cal(stock_log2))

stock_log3 = np.log(stock_log2)
stock_log3 = stock_log3[1:]

print(ADF_Cal(stock_log3))

stock_diff1_after_log3 = stock_log3.diff()
stock_diff1_after_log3 = stock_diff1_after_log3[1:]
print(ADF_Cal(stock_diff1_after_log3))


#Calculating Rolling Mean and Variance AFter 3rd log + 1st diff
diff_air_pass_mean = 0
diff_air_pass_variance = 0

diff_mean_air_pass_time = []
diff_var_air_pass_time = []

for i in range(1, len(stock_diff1_after_log3)+1):
    rows = stock_diff1_after_log3.head(i)
    seq_air_mean = rows.values[-1]
    seq_air_var = rows.values

    diff_air_pass_mean = diff_air_pass_mean + seq_air_mean
    diff_mean_air_pass_time.append(diff_air_pass_mean / i)

    if i == 1:
        diff_var_air_pass_time.append(seq_air_var[-1])
    elif i > 1:
        diff_var_air_pass_time.append(statistics.variance(seq_air_var))


#Close Price Rolling Variance (3rd log + 1st diff)
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(date[4:],diff_var_air_pass_time, label = "Rolling Variance")
plt.xticks(rotation=45)
ax.set_xticks(ax.get_xticks()[::200])
plt.xlabel("Year")
plt.ylabel("Dollars ($USD)")
plt.legend()
plt.title("Google Stock Close Price Rolling Variance After 3rd log and 1st diff")
plt.show()

#Close Price Rolling Mean (3rd log + 1st diff)
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(date[4:],diff_mean_air_pass_time, label = "Rolling Mean")
plt.xticks(rotation=45)
ax.set_xticks(ax.get_xticks()[::200])
plt.xlabel("Year")
plt.ylabel("Dollars ($USD)")
plt.legend()
plt.title("Google Stock Close Price Rolling Mean after 3rd log and 1st diff")
plt.show()

#Defining New Data
stationary_data = stock_diff1_after_log3
new_data = stationary_data.tolist()

#Plotting Stationary Data
fig, ax = plt.subplots(figsize = (10,9))
#ax.plot(date,stock, label = "Original")
plt.plot(date[4:],stationary_data, label = " 3rd log + 1st diff")
plt.xticks(rotation=45)
ax.set_xticks(ax.get_xticks()[::200])
plt.xlabel("Year")
plt.ylabel("Dollars ($USD)")
plt.legend()
plt.title("Google Stock Close Price")
plt.show()

#Close Price Differencing
fig, ax = plt.subplots(figsize = (10,9))
ax.plot(date,stock, label = "Original")
plt.plot(date[1:], stock_log1, label="1st log", alpha = 0.5)
plt.plot(date[2:], stock_log2, label="2nd log", alpha = 0.5)
plt.plot(date[3:], stock_log3, label="3rd log", alpha = 0.5)
plt.plot(date[4:], stock_diff1_after_log3, label="3rd log + 1st diff", alpha = 0.5)
plt.xticks(rotation=45)
ax.set_xticks(ax.get_xticks()[::200])
plt.xlabel("Year")
plt.ylabel("Dollars ($USD)")
plt.legend()
plt.title("Google Stock Original vs. Differencing")
plt.show()

#Plotting ACF and PACF using statsmodel
acf = sm.tsa.stattools.acf(stock, nlags=20)
pacf = sm.tsa.stattools.pacf(stock, nlags=20)

fig = plt.figure(figsize=(6, 10))
fig.tight_layout(pad=4)
plt.subplot(2, 1, 1)
plot_acf(new_data, ax=plt.gca(), lags=20, title="ACF of Stationary Data")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.subplot(2, 1, 2)
plot_pacf(new_data, ax=plt.gca(), lags=20, title="PACF of Stationary Data")
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.show()


df_stationary = pd.DataFrame(new_data,columns={'stationary_stock'})

#Time Series Decomposition======================
"""decomp_data = pd.Series(np.array(df_stationary['stationary_stock']),
                 index = pd.date_range('2006-01-09',
                 periods=len(df_stationary['stationary_stock']),
                 freq = 'm',
                 name='google_stock'))"""
decomp_data = pd.Series(np.array(df['Close']),
                 index = pd.date_range('2006-01-09',
                 periods=len(df['Close']),
                 freq = 'd',
                 name='google_stock'))
STL = STL(decomp_data)
res = STL.fit()#optimization

T = res.trend
S = res.seasonal
R = res.resid

#Plotting decomposition
fig = res.plot()
plt.show()


adjusted_seasonal = decomp_data.values - S
adjusted_trend = decomp_data.values -T


fig, ax  = plt.subplots(figsize=(10,10))
plt.plot(date,decomp_data.values, label = 'Original Data')
plt.plot(date,adjusted_seasonal, label = 'Adjusted Seasonal')
plt.plot(date,adjusted_trend, label = 'Adjusted Trend')
ax.set_xticks(ax.get_xticks()[::350])
plt.xticks(rotation=45)
plt.title('Google Stock Time Series Decomposition')
plt.ylabel("Magnitude")
plt.xlabel("Time")
plt.legend()
plt.show()


#Strength and Seasonality
#Calculate Strength of Trend
R = np.array(R)
T = np.array(T)
S = np.array(S)

F_t = np.max([0, (1 - np.var(R)/np.var(T+R))])
print("The strength of the trend for this data set is: ",F_t)



#Calculate Strength of /Seasonality
F_s = max([0, (1 - np.var(R)/np.var(S+R))])
print("The strength of the seasonality for this data set is: ",F_s)



#================================================

#Holt-Winters Method:==========================
#Splitting the Data
#Calculate Q Function
def calc_q(autocorrelation_list, num_samples):
    acf_sq =[x**2 for x in autocorrelation_list[1:]]
    q = sum(acf_sq)*num_samples
    return q

X_train, X_test, y_train, y_test = train_test_split(date,stock, shuffle = False,test_size=0.20)

#Calculate Error
def calc_error(y,predicted_value):
    err = []
    for i,j in zip(y,predicted_value):
        err.append(i-j)
    return err[1:]

#Error Squared
def calc_error_squared(error):
    error_squared = [x**2 for x in error]
    return error_squared

#MSE
def calc_mse(error_squared):
    mse = sum(error_squared)/len(error_squared)
    return mse

#Chi-square test
def chi_square(Q,y_train_length,row_length,columns_length):
    DOF = y_train_length - row_length - columns_length
    alfa = 0.01
    chi_critical = chi2.ppf(1-alfa,DOF)

    if Q < chi_critical:
        print("Residual is white")
    else:
        print("The Residual is NOT white")


def holt_winter(y_train,y_test):
    aapl_holttw = ets.ExponentialSmoothing(y_train.values, trend='additive',damped=False,seasonal=None,
                                           seasonal_periods=4).fit()
    aapl_holtfw = aapl_holttw.forecast(steps=len(y_test))
    aapl_holtfw = pd.DataFrame(aapl_holtfw).set_index(y_test.index)
    aapl_forecast_error_holtw = calc_error(y_test,aapl_holtfw.values)
    forecast_error_squared = calc_error_squared(aapl_forecast_error_holtw)
    forecast_error_mse = calc_mse(forecast_error_squared)
    #aapl_holtw_forecast_error_mse = np.square(np.subtract(y_test, aapl_forecast_error_holtw)).mean()

    return aapl_holtfw,forecast_error_mse, aapl_forecast_error_holtw

stock_hw_forecast,stock_hw_forecast_error_mse, stock_hw_forecast_error = holt_winter(y_train,y_test)
print('Holt-Winters Forecast Error MSE:',stock_hw_forecast_error_mse)
hw_forecast_error_acf = calc_autocorrelation_coef(stock_hw_forecast_error,20,'Holt-Winter Forecast Error ACF')
hw_forecast_error_Q = calc_q(hw_forecast_error_acf,len(stock_hw_forecast_error))
print('Holt-Winters Forecast Error Q:',hw_forecast_error_Q)
hw_forecast_error_variance = np.var(stock_hw_forecast_error)
print('Holt-Winters Forecast Error Variance:',hw_forecast_error_variance)
hw_forecast_error_mean = np.mean(stock_hw_forecast_error)
print('Holt-Winters Forecast Error Mean:',hw_forecast_error_mean)


#Plotting Holt-Winter Forecast
fig, ax  = plt.subplots(figsize = (10,10))
plt.plot(X_train,y_train, label = 'Training dataset')
plt.plot(X_test,y_test, label = 'Testing dataset')
plt.plot(X_test,stock_hw_forecast, label = 'Holt-Winter Method h-step forecast')
ax.set_xticks(ax.get_xticks()[::450])
plt.xticks(rotation=45)
plt.xlabel("Time (t)")
plt.ylabel("Magnitude")
plt.legend(loc = 'upper left')
plt.title("Google Stock Holt-Winter Forecasting")
plt.show()



#===============================================


#Feature Selection==============

X = df[['Open','High','Low','Volume']]
Y = df['Close']

X_train1, X_test1, Y_train1, Y_test1  = train_test_split(X,Y,shuffle = False,test_size=0.20)

#Calculating Singular Values and Condition Number
X = X_train1.values
H = np.matmul(X.T,X)
_,d,_ = np.linalg.svd(H)
print('Singular Values of Original',d)
print('The condition number of original = ',LA.cond(X))

#Addding constant to matrix
X_train1['one'] = 1
X_train1 = X_train1[ ['one'] + [ column for column in X_train1.columns if column!= 'one']]
#print(X_train)

#Calculating Coefficients using LSE
H = np.matmul(X_train1.values.T,X_train1.values)
inv = np.linalg.inv(H)
beta = np.matmul(np.matmul(inv,X_train1.values.T),Y_train1.values)
print("Calculated Coefficients using LSE:",beta)

#Calculating Coefficients using OLS
model = sm.OLS(Y_train1,X_train1).fit()
print(model.summary())#Coefficients are identical

#Removing Predictors - one constant
X_train1.drop(['one'],axis=1,inplace=True)

model = sm.OLS(Y_train1,X_train1).fit()
X = X_train1.values
H = np.matmul(X.T,X)
_,d,_ = np.linalg.svd(H)
print('Singular Values of First Iteration',d)
print('The Condition Number of First Iteration = ',LA.cond(X))
print(model.summary())

#Removing Volume
X_train1.drop(['Volume'],axis=1,inplace=True)#pvalue > 0.05
X_test1.drop(['Volume'],axis=1,inplace=True)

model = sm.OLS(Y_train1,X_train1).fit()
X = X_train1.values
H = np.matmul(X.T,X)
_,d,_ = np.linalg.svd(H)
print('Singular Values of Final Iteration',d)
print('The Condition Number of Final Iteration = ',LA.cond(X))
print(model.summary())
#============================


#Multiple Linear Regression==================

forecast = model.predict(X_test1)
prediction = model.predict(X_train1)

#Plotting Multiple Linear Regression Forecast vs. Test Set
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(X_train,Y_train1, label = 'Training dataset')
ax.plot(X_train,prediction, label='Prediction')
ax.plot(X_test,Y_test1,alpha=0.5, label = 'Testing dataset')
ax.plot(X_test,forecast, alpha=0.5,label = 'OLS Method h-step forecast')
plt.xlabel("Time (t)")
plt.ylabel("Magnitude")
ax.set_xticks(ax.get_xticks()[::350])
plt.legend(loc = 'upper left')
plt.xticks(rotation=45)
plt.title("Google Stock OLS 1-step Prediction and h-step Forecast")
plt.show()

#Plotting Multiple Linear Regression Forecast vs. Test Set
fig, ax = plt.subplots(figsize = (8,9))
ax.plot(X_test,Y_test1,'b',alpha=0.5, label = 'Testing dataset')
ax.plot(X_test,forecast,'r', alpha=0.5,label = 'OLS Method h-step forecast')
plt.xlabel("Time (t)")
plt.ylabel("Magnitude")
ax.set_xticks(ax.get_xticks()[::100])
plt.legend(loc = 'upper left')
plt.xticks(rotation=45)
plt.title("Google Stock OLS h-step Forecast")
plt.show()

#Comparing Performance

#Calculating Prediction Error
def calc_error(y,predicted_value):
    err = []

    for i,j in zip(y,predicted_value):
        err.append(i-j)
    return err
mlr_prediction_error = calc_error(Y_train1,prediction)
mlr_forecast_error = calc_error(Y_test1,forecast)

#Calculating MSE of Forecast/Prediction Error Multiple Linear Regression
mlr_forecast_error_squared = calc_error_squared(mlr_forecast_error)
mlr_forecast_error_mse = calc_mse(mlr_forecast_error_squared)
print("MLR Forecast Error MSE:",mlr_forecast_error_mse)

#Plotting ACF of Forecast Error
forecast_error_acf_r_y = calc_autocorrelation_coef(mlr_forecast_error,20,"ACF of MLR Forecast Error")

#Calculating MLR Forecast Error Q
stock_mlr_q = calc_q(forecast_error_acf_r_y,len(Y))
print("MLR Forecast Error Q", stock_mlr_q)

#Calculating MLR Mean and Variance of Forecast Error
mlr_forecast_error_mean = np.mean(mlr_forecast_error)
mlr_forecast_error_variance = np.var(mlr_forecast_error)
print("MLR Forecast Error Variance:",mlr_forecast_error_variance)
print("MLR Forecast Error Mean:",mlr_forecast_error_mean)

#Forecast Estimated Variance
forecast_error_squared_sum = sum([x**2 for x in mlr_forecast_error])
fore_variance = math.sqrt(forecast_error_squared_sum * (1/(len(mlr_forecast_error) - 7 - 1)))
print('Multiple Linear Regression Forecast Error Estimated Variance:',fore_variance)

#Prediction Error Analysis
mlr_prediction_error_squared = calc_error_squared(mlr_prediction_error)
mlr_prediction_error_mse = calc_mse(mlr_prediction_error_squared)
print("MLR Prediction Error MSE:",mlr_prediction_error_mse)

#Plotting ACF of Prediction Error
prediction_error_acf_r_y = calc_autocorrelation_coef(mlr_prediction_error,20,"ACF of Prediction Error")

#Calculating MLR Prediction Error Q
mlr_prediction_error_Q = calc_q(prediction_error_acf_r_y,len(Y))
print("MLR Prediction Error Q", stock_mlr_q)

mlr_prediction_error_mean = np.mean(mlr_prediction_error)
mlr_prediction_error_variance = np.var(mlr_prediction_error)
print("MLR Prediction Error Variance:",mlr_forecast_error_variance)
print("MLR Prediction Error Mean:",mlr_forecast_error_mean)

#F-test and T-test analysis
print(model.summary())

#ARMA, ARIMA and SARIMA==================================

#GPAC FUNCTION
def cal_gpac(j1,k1,data):
    y = data
    ry = calc_autocorrelation_coef(y,50,"ACF")
    print(len(ry))
    numerator = []
    denominator = []

    phi_k1 = []
    phi_k2 = []
    phi_k3 = []
    phi_k4 = []
    phi_k5 = []
    phi_k6 = []
    phi_k7 = []
    phi_k8 = []
    phi_k9 = []
    phi_k10 = []
    phi_k11 = []
    phi_k12 = []
    phi_k13 = []
    phi_k14 = []
    phi_k15 = []


    for k in range(1,k1+1):
        for j in range(0,j1):
            if k==1:
                numerator.append(ry[j+k])
                denominator.append(ry[j])
                phi_k1.append(numerator[j]/denominator[j])

            elif k==2:
                numerator_array2 = np.zeros((k, k))
                denominator_array2 = np.zeros((k, k))
                numerator_array2[0] = (ry[j], ry[j + 1])
                numerator_array2[1] = (ry[j + 1], ry[j + 2])
                if j == 0:
                    denominator_array2[0] = (ry[j],ry[j+1])
                    denominator_array2[1] = (ry[j+1],ry[j])
                else:
                    denominator_array2[0] = (ry[j],ry[j-1])
                    denominator_array2[1] = (ry[j+1],ry[j])
                phi_k2.append(np.linalg.det(numerator_array2)/np.linalg.det(denominator_array2))

            elif k==3:
                numerator_array3 = np.zeros((k, k))
                denominator_array3 = np.zeros((k, k))

                if j == 0:
                    numerator_array3[0] = (ry[j], ry[j+1], ry[j + 1])
                    numerator_array3[1] = (ry[j + 1], ry[j], ry[j + 2])
                    numerator_array3[2] = (ry[j + 2], ry[j + 1], ry[j + 3])
                    denominator_array3[0] = (ry[j], ry[j + 1], ry[j+2])
                    denominator_array3[1] = (ry[j + 1], ry[j], ry[j+1])
                    denominator_array3[2] = (ry[j+2], ry[j+1], ry[j])
                    phi_k3.append(np.linalg.det(numerator_array3) / np.linalg.det(denominator_array3))

                elif j==1:
                    numerator_array3[0] = (ry[j], ry[j - 1], ry[j + 1])
                    numerator_array3[1] = (ry[j + 1], ry[j], ry[j + 2])
                    numerator_array3[2] = (ry[j + 2], ry[j + 1], ry[j + 3])

                    denominator_array3[0] = (ry[j], ry[j - 1], ry[j])
                    denominator_array3[1] = (ry[j + 1], ry[j], ry[j - 1])
                    denominator_array3[2] = (ry[j + 2], ry[j + 1], ry[j])
                    phi_k3.append(np.linalg.det(numerator_array3) / np.linalg.det(denominator_array3))

                elif j > 1:
                    numerator_array3[0] = (ry[j], ry[j - 1], ry[j + 1])
                    numerator_array3[1] = (ry[j + 1], ry[j], ry[j + 2])
                    numerator_array3[2] = (ry[j + 2], ry[j + 1], ry[j + 3])

                    denominator_array3[0] = (ry[j], ry[j - 1], ry[j-2])
                    denominator_array3[1] = (ry[j + 1], ry[j], ry[j-1])
                    denominator_array3[2] = (ry[j+2], ry[j+1], ry[j])
                    phi_k3.append(np.linalg.det(numerator_array3)/np.linalg.det(denominator_array3))

            elif k == 4:
                numerator_array4 = np.zeros((k, k))
                denominator_array4 = np.zeros((k, k))


                if j == 0:
                    numerator_array4[0] = (ry[j],     ry[j - 1 + 2], ry[j - 2 + 4], ry[j + 1])
                    numerator_array4[1] = (ry[j + 1], ry[j],     ry[j - 1 + 2], ry[j + 2])
                    numerator_array4[2] = (ry[j + 2], ry[j + 1], ry[j],     ry[j + 3])
                    numerator_array4[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j + 4])

                    denominator_array4[0] = (ry[j],     ry[j - 1 + 2], ry[j - 2 + 4], ry[j-k+1+6])
                    denominator_array4[1] = (ry[j + 1], ry[j],     ry[j - 1 + 2], ry[j-k+2+4])
                    denominator_array4[2] = (ry[j + 2], ry[j + 1], ry[j],     ry[j-k+3+2])
                    denominator_array4[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j-k+4])

                    phi_k4.append(np.linalg.det(numerator_array4) / np.linalg.det(denominator_array4))

                elif j == 1:
                    numerator_array4[0] = (ry[j], ry[j - 1], ry[j - 2 + 2], ry[j + 1])
                    numerator_array4[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j + 2])
                    numerator_array4[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j + 3])
                    numerator_array4[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j + 4])

                    denominator_array4[0] = (ry[j], ry[j - 1], ry[j - 2 + 2], ry[j - k + 1 + 4])
                    denominator_array4[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - k + 2 + 2])
                    denominator_array4[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - k + 3])
                    denominator_array4[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j - k + 4])
                    phi_k4.append(np.linalg.det(numerator_array4) / np.linalg.det(denominator_array4))
                elif j == 2:
                    numerator_array4[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j + 1])
                    numerator_array4[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j + 2])
                    numerator_array4[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j + 3])
                    numerator_array4[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j + 4])

                    denominator_array4[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - k + 1 + 2])
                    denominator_array4[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - k + 2])
                    denominator_array4[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - k + 3])
                    denominator_array4[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j - k + 4])
                    phi_k4.append(np.linalg.det(numerator_array4) / np.linalg.det(denominator_array4))
                elif j > 1:
                    numerator_array4[0] = (ry[j],   ry[j-1], ry[j-2], ry[j+1])
                    numerator_array4[1] = (ry[j+1], ry[j],   ry[j-1], ry[j+2])
                    numerator_array4[2] = (ry[j+2], ry[j+1], ry[j],   ry[j+3])
                    numerator_array4[3] = (ry[j+3], ry[j+2], ry[j+1], ry[j+4])

                    denominator_array4[0] = (ry[j],   ry[j-1], ry[j-2], ry[j-k+1])
                    denominator_array4[1] = (ry[j+1], ry[j],   ry[j-1], ry[j-k+2])
                    denominator_array4[2] = (ry[j+2], ry[j+1], ry[j],   ry[j-k+3])
                    denominator_array4[3] = (ry[j+3], ry[j+2], ry[j+1], ry[j-k+4])
                    phi_k4.append(np.linalg.det(numerator_array4) / np.linalg.det(denominator_array4))

            elif k == 5:
                numerator_array5 = np.zeros((k, k))
                denominator_array5 = np.zeros((k, k))

                if j == 0:
                    numerator_array5[0] = (ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j - 3 + 6], ry[j + 1])
                    numerator_array5[1] = (ry[j + 1], ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j + 2])
                    numerator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1 + 2], ry[j + 3])
                    numerator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4])
                    numerator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j + 5])

                    denominator_array5[0] = (ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j - 3 + 6], ry[j - k + 1 + 8])
                    denominator_array5[1] = (ry[j + 1], ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j - k + 2 + 6])
                    denominator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1 + 2], ry[j - k + 3 + 4])
                    denominator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - k + 4 + 2])
                    denominator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j - k + 5])

                    phi_k5.append(np.linalg.det(numerator_array5) / np.linalg.det(denominator_array5))

                elif j == 1:
                    numerator_array5[0] = (ry[j], ry[j - 1], ry[j - 2 + 2], ry[j - 3 + 4], ry[j + 1])
                    numerator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2 + +2], ry[j + 2])
                    numerator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j + 3])
                    numerator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4])
                    numerator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j + 5])

                    denominator_array5[0] = (ry[j], ry[j - 1], ry[j - 2 + 2], ry[j - 3 + 4], ry[j - k + 1 + 6])
                    denominator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j - k + 2 + 4])
                    denominator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - k + 3 + 2])
                    denominator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - k + 4])
                    denominator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j - k + 5])

                    phi_k5.append(np.linalg.det(numerator_array5) / np.linalg.det(denominator_array5))

                elif j == 2:
                    numerator_array5[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3 + 2], ry[j + 1])
                    numerator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j + 2])
                    numerator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j + 3])
                    numerator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4])
                    numerator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j + 5])

                    denominator_array5[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3 + 2], ry[j - k + 1 + 4])
                    denominator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - k + 2 + 2])
                    denominator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - k + 3])
                    denominator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - k + 4])
                    denominator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j - k + 5])

                    phi_k5.append(np.linalg.det(numerator_array5) / np.linalg.det(denominator_array5))

                elif j == 3:
                    numerator_array5[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j + 1])
                    numerator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j + 2])
                    numerator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j + 3])
                    numerator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4])
                    numerator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j + 5])

                    denominator_array5[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - k + 1 + 2])
                    denominator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - k + 2])
                    denominator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - k + 3])
                    denominator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - k + 4])
                    denominator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j - k + 5])
                    phi_k5.append(np.linalg.det(numerator_array5) / np.linalg.det(denominator_array5))

                elif j > 3:
                    numerator_array5[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j-3], ry[j + 1])
                    numerator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j-2], ry[j + 2])
                    numerator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j-1], ry[j + 3])
                    numerator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4])
                    numerator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j + 5])

                    denominator_array5[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j-3], ry[j - k + 1])
                    denominator_array5[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j-2], ry[j - k + 2])
                    denominator_array5[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j-1], ry[j - k + 3])
                    denominator_array5[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - k + 4])
                    denominator_array5[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 5])

                    phi_k5.append(np.linalg.det(numerator_array5) / np.linalg.det(denominator_array5))

            elif k == 6:
                numerator_array6 = np.zeros((k, k))
                denominator_array6 = np.zeros((k, k))

                if j == 0:
                    numerator_array6[0] = (ry[j],     ry[j - 1 + 2], ry[j - 2 + 4], ry[j - 3 + 6], ry[j - 4 + 8], ry[j + 1])
                    numerator_array6[1] = (ry[j + 1], ry[j],         ry[j - 1 + 2], ry[j - 2 + 4], ry[j - 3 + 6], ry[j + 2])
                    numerator_array6[2] = (ry[j + 2], ry[j + 1],     ry[j],         ry[j - 1 + 2], ry[j - 2 + 4], ry[j + 3])
                    numerator_array6[3] = (ry[j + 3], ry[j + 2],     ry[j + 1],     ry[j], ry[j - 1 + 2], ry[j + 4])
                    numerator_array6[4] = (ry[j + 4], ry[j + 3],     ry[j + 2],     ry[j + 1], ry[j], ry[j + 5])
                    numerator_array6[5] = (ry[j + 5], ry[j + 4],     ry[j + 3],     ry[j + 2], ry[j + 1], ry[j + 6])

                    denominator_array6[0] = (ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j - 3 + 6], ry[j - 4 + 8], ry[j - k + 1 + 10])
                    denominator_array6[1] = (ry[j + 1], ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j - 3 + 6], ry[j - k + 2 + 8])
                    denominator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j - k + 3 + 6])
                    denominator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1 + 2], ry[j - k + 4 + 4])
                    denominator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - k + 5 + 2])
                    denominator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j - k + 6])

                    phi_k6.append(np.linalg.det(numerator_array6)/np.linalg.det(denominator_array6))

                if j == 1:
                    numerator_array6[0] = (ry[j], ry[j - 1], ry[j - 2 + 2], ry[j - 3 + 4],ry[j-4 + 6], ry[j + 1])
                    numerator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2 + 2],ry[j-3+4], ry[j + 2])
                    numerator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2 +2], ry[j + 3])
                    numerator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 4])
                    numerator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 5])
                    numerator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 6])

                    denominator_array6[0] = (ry[j], ry[j - 1], ry[j - 2 + 2], ry[j - 3 + 4],ry[j-4 + 6], ry[j - k + 1 + 8])
                    denominator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2],ry[j-3+4], ry[j - k + 2 + 6])
                    denominator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2+2], ry[j - k + 3 + 4])
                    denominator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 4 + 2])
                    denominator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 5])
                    denominator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 6])

                    phi_k6.append(np.linalg.det(numerator_array6)/np.linalg.det(denominator_array6))

                if j==2:
                    numerator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3 + 2],ry[j-4 +4], ry[j + 1])
                    numerator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3 + 2], ry[j + 2])
                    numerator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 3])
                    numerator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 4])
                    numerator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 5])
                    numerator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 6])

                    denominator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3 + 2],ry[j-4 + 4], ry[j - k + 1 + 6])
                    denominator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3+2], ry[j - k + 2 + 4])
                    denominator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 3 + 2])
                    denominator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 4])
                    denominator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 5])
                    denominator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 6])

                    phi_k6.append(np.linalg.det(numerator_array6)/np.linalg.det(denominator_array6))

                if j==3:
                    numerator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4 + 2], ry[j + 1])
                    numerator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 2])
                    numerator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 3])
                    numerator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 4])
                    numerator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 5])
                    numerator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 6])

                    denominator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4 + 2], ry[j - k + 1 + 4])
                    denominator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 2 + 2])
                    denominator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 3])
                    denominator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 4])
                    denominator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 5])
                    denominator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 6])

                    phi_k6.append(np.linalg.det(numerator_array6)/np.linalg.det(denominator_array6))

                if j==4:
                    numerator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 1])
                    numerator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 2])
                    numerator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 3])
                    numerator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 4])
                    numerator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 5])
                    numerator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 6])

                    denominator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j - k + 1 + 2])
                    denominator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 2])
                    denominator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 3])
                    denominator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 4])
                    denominator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 5])
                    denominator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 6])

                    phi_k6.append(np.linalg.det(numerator_array6)/np.linalg.det(denominator_array6))

                elif j > 4:
                    numerator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 1])
                    numerator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 2])
                    numerator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 3])
                    numerator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 4])
                    numerator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 5])
                    numerator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 6])

                    denominator_array6[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j - k + 1])
                    denominator_array6[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 2])
                    denominator_array6[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 3])
                    denominator_array6[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 4])
                    denominator_array6[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 5])
                    denominator_array6[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 6])

                    phi_k6.append(np.linalg.det(numerator_array6) / np.linalg.det(denominator_array6))

            elif k == 7:
                numerator_array7 = np.zeros((k, k))
                denominator_array7 = np.zeros((k, k))

                if j == 0:
                    numerator_array7[0] = (ry[j], ry[j - 1 + 2], ry[j - 2 + 4], ry[j - 3 + 6], ry[j - 4 + 8], ry[j-5+10], ry[j + 1])
                    numerator_array7[1] = (ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6],ry[j-4+8], ry[j + 2])
                    numerator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j-3+6],ry[j + 3])
                    numerator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2],ry[j-2+4], ry[j + 4])
                    numerator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1+2], ry[j + 5])
                    numerator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 6])
                    numerator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 7])


                    denominator_array7[0] = (ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6], ry[j - 4+8], ry[j-5+10], ry[j - k + 1+12])
                    denominator_array7[1] = (ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6], ry[j-4+8],ry[j - k + 2+10])
                    denominator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j-3+6],ry[j - k + 3+8])
                    denominator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2],ry[j-2+4], ry[j - k + 4+6])
                    denominator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1+2], ry[j - k + 5+4])
                    denominator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 6+2])
                    denominator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 7])

                    phi_k7.append(np.linalg.det(numerator_array7) / np.linalg.det(denominator_array7))

                elif j == 1:
                    numerator_array7[0] = (ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4], ry[j - 4+6], ry[j-5+8], ry[j + 1])
                    numerator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4],ry[j-4+6], ry[j + 2])
                    numerator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j-3+4],ry[j + 3])
                    numerator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2+2], ry[j + 4])
                    numerator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 5])
                    numerator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 6])
                    numerator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 7])


                    denominator_array7[0] = (ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4], ry[j - 4+6], ry[j-5+8], ry[j - k + 1+10])
                    denominator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4], ry[j-4+6],ry[j - k + 2+8])
                    denominator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j-3+4],ry[j - k + 3+6])
                    denominator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2+2], ry[j - k + 4+4])
                    denominator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 5+2])
                    denominator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 6])
                    denominator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 7])

                    phi_k7.append(np.linalg.det(numerator_array7) / np.linalg.det(denominator_array7))

                elif j==2:
                    numerator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3+2], ry[j - 4+4], ry[j-5+6], ry[j + 1])
                    numerator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3+2],ry[j-4+4], ry[j + 2])
                    numerator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3+2],ry[j + 3])
                    numerator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 4])
                    numerator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 5])
                    numerator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 6])
                    numerator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 7])


                    denominator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3+2], ry[j - 4+4], ry[j-5+6], ry[j - k + 1 + 8])
                    denominator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3+2], ry[j-4+4],ry[j - k + 2 + 6])
                    denominator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3+2],ry[j - k + 3 + 4])
                    denominator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 4 + 2])
                    denominator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 5])
                    denominator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 6])
                    denominator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 7])

                    phi_k7.append(np.linalg.det(numerator_array7) / np.linalg.det(denominator_array7))

                elif j==3:
                    numerator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+2], ry[j-5+4], ry[j + 1])
                    numerator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4+2], ry[j + 2])
                    numerator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j + 3])
                    numerator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 4])
                    numerator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 5])
                    numerator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 6])
                    numerator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 7])


                    denominator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+2], ry[j-5+4], ry[j - k + 1 + 6])
                    denominator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j-4+2],ry[j - k + 2 + 4])
                    denominator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j - k + 3 + 2])
                    denominator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 4])
                    denominator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 5])
                    denominator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 6])
                    denominator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 7])
                    phi_k7.append(np.linalg.det(numerator_array7) / np.linalg.det(denominator_array7))

                elif j==4:
                    numerator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j-5+2], ry[j + 1])
                    numerator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 2])
                    numerator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j + 3])
                    numerator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 4])
                    numerator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 5])
                    numerator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 6])
                    numerator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 7])


                    denominator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j-5+2], ry[j - k + 1 + 4])
                    denominator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j-4],ry[j - k + 2 + 2])
                    denominator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j - k + 3])
                    denominator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 4])
                    denominator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 5])
                    denominator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 6])
                    denominator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 7])
                    phi_k7.append(np.linalg.det(numerator_array7) / np.linalg.det(denominator_array7))
                elif j ==5:
                    numerator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j-5], ry[j + 1])
                    numerator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 2])
                    numerator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j + 3])
                    numerator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 4])
                    numerator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 5])
                    numerator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 6])
                    numerator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 7])


                    denominator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j-5], ry[j - k + 1 + 2])
                    denominator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j-4],ry[j - k + 2])
                    denominator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j - k + 3])
                    denominator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 4])
                    denominator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 5])
                    denominator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 6])
                    denominator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 7])
                    phi_k7.append(np.linalg.det(numerator_array7) / np.linalg.det(denominator_array7))

                elif j > 5:
                    numerator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j-5], ry[j + 1])
                    numerator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 2])
                    numerator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j + 3])
                    numerator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 4])
                    numerator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 5])
                    numerator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 6])
                    numerator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 7])

                    denominator_array7[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j-5], ry[j - k + 1])
                    denominator_array7[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j-4],ry[j - k + 2])
                    denominator_array7[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j-3],ry[j - k + 3])
                    denominator_array7[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 4])
                    denominator_array7[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 5])
                    denominator_array7[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 6])
                    denominator_array7[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1], ry[j - k + 7])

                    phi_k7.append(np.linalg.det(numerator_array7) / np.linalg.det(denominator_array7))

            elif k==8:
                numerator_array8 = np.zeros((k, k))
                denominator_array8 = np.zeros((k, k))

                if j==0:
                    numerator_array8[0] = (ry[j], ry[j - 1 + 2], ry[j - 2+4], ry[j - 3+6], ry[j - 4+8], ry[j - 5+10], ry[j-6+12], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6], ry[j - 4+8],ry[j-5+10], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6],ry[j-4+8], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4],ry[j-3+6], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2],ry[j-2+4], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1+2], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6], ry[j - 4+8], ry[j - 5+10], ry[j-6+12], ry[j - k + 1 +14])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6], ry[j - 4+8],ry[j-5+10], ry[j - k + 2+12])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4], ry[j - 3+6],ry[j-4+8], ry[j - k + 3+10])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2], ry[j - 2+4],ry[j-3+6], ry[j - k + 4+8])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1+2],ry[j-2+4], ry[j - k + 5+6])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1+2], ry[j - k + 6+4])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7+2])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))

                if j==1:
                    numerator_array8[0] = (ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4], ry[j - 4+6], ry[j - 5+8], ry[j-6+10], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4], ry[j - 4+6],ry[j-5+8], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4],ry[j-4+6], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2],ry[j-3+4], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2+2], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4], ry[j - 4+6], ry[j - 5+8], ry[j-6+10], ry[j - k + 1+12])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4], ry[j - 4+6],ry[j-5+8], ry[j - k + 2+10])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2], ry[j - 3+4],ry[j-4+6], ry[j - k + 3+8])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2+2],ry[j-3+4], ry[j - k + 4+6])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2+2], ry[j - k + 5+4])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 6+2])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))

                elif j==2:
                    numerator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3+2], ry[j - 4+4], ry[j - 5+6], ry[j-6+8], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3+2], ry[j - 4+4],ry[j-5+6], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4+4], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+4], ry[j - 5+6], ry[j-6+8], ry[j - k + 1+10])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+4],ry[j-5+6], ry[j - k + 2+8])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4+4], ry[j - k + 3+6])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 4+4])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 5+2])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 6])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))

                elif j==3:
                    numerator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+2], ry[j - 5+4], ry[j-6+6], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+2],ry[j-5+4], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4+2], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+2], ry[j - 5+4], ry[j-6+6], ry[j - k + 1+8])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4+2],ry[j-5+4], ry[j - k + 2+6])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4+2], ry[j - k + 3+4])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 4+2])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 5])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 6])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))

                elif j==4:
                    numerator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5+2], ry[j-6+4], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5+2], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5+2], ry[j-6+4], ry[j - k + 1+6])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5+2], ry[j - k + 2+4])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j - k + 3+2])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 4])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 5])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 6])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))

                elif j==5:
                    numerator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5], ry[j-6+2], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5], ry[j-6+2], ry[j - k + 1+4])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5], ry[j - k + 2+2])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j - k + 3])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 4])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 5])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 6])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))

                elif j==6:
                    numerator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5], ry[j-6], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5], ry[j-6], ry[j - k + 1 +2])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5], ry[j - k + 2])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j - k + 3])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 4])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 5])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 6])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))


                elif j>6:
                    numerator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5], ry[j-6], ry[j + 1])
                    numerator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5], ry[j + 2])
                    numerator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j + 3])
                    numerator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j + 4])
                    numerator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j + 5])
                    numerator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j + 6])
                    numerator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j + 7])
                    numerator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j + 8])

                    denominator_array8[0] = (ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4], ry[j - 5], ry[j-6], ry[j - k + 1])
                    denominator_array8[1] = (ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3], ry[j - 4],ry[j-5], ry[j - k + 2])
                    denominator_array8[2] = (ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2], ry[j - 3],ry[j-4], ry[j - k + 3])
                    denominator_array8[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1], ry[j - 2],ry[j-3], ry[j - k + 4])
                    denominator_array8[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j - 1],ry[j-2], ry[j - k + 5])
                    denominator_array8[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[j-1], ry[j - k + 6])
                    denominator_array8[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[j - k + 7])
                    denominator_array8[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j - k + 8])
                    phi_k8.append(np.linalg.det(numerator_array8) / np.linalg.det(denominator_array8))
            elif k ==9:
                numerator_array9 = np.zeros((k, k))
                denominator_array9 = np.zeros((k, k))
                numerator_array9[0] = (ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j-6)], ry[abs(j -7)], ry[j + 1])
                numerator_array9[1] = (ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],ry[abs(j-5)], ry[abs(j -6)],ry[j + 2])
                numerator_array9[2] = (ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],ry[abs(j-4)], ry[abs(j -5)],ry[j + 3])
                numerator_array9[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],ry[abs(j-3)], ry[abs(j -4)],ry[j + 4])
                numerator_array9[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],ry[abs(j-2)], ry[abs(j -3)],ry[j + 5])
                numerator_array9[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[abs(j-1)], ry[abs(j -2)],ry[j + 6])
                numerator_array9[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[abs(j -1)],ry[j + 7])
                numerator_array9[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j],ry[j + 8])
                numerator_array9[8] = (ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3],ry[j+2], ry[j+1],ry[j + 9])


                for i in range(0,k):
                    denominator_array9[i][0:-1] = numerator_array9[i][0:-1]
                    denominator_array9[i][-1] = ry[abs(j - k + i + 1)]

                phi_k9.append(np.linalg.det(numerator_array9) / np.linalg.det(denominator_array9))

            elif k ==10:
                numerator_array10 = np.zeros((k, k))
                denominator_array10 = np.zeros((k, k))
                numerator_array10[0] = (ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j-6)], ry[abs(j -7)],ry[abs(j -8)], ry[j + 1])
                numerator_array10[1] = (ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],ry[abs(j-5)], ry[abs(j -6)],ry[abs(j -7)],ry[j + 2])
                numerator_array10[2] = (ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],ry[abs(j-4)], ry[abs(j -5)],ry[abs(j -6)],ry[j + 3])
                numerator_array10[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],ry[abs(j-3)], ry[abs(j -4)],ry[abs(j -5)],ry[j + 4])
                numerator_array10[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],ry[abs(j-2)], ry[abs(j -3)],ry[abs(j -4)],ry[j + 5])
                numerator_array10[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[abs(j-1)], ry[abs(j -2)],ry[abs(j -3)],ry[j + 6])
                numerator_array10[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[abs(j -1)],ry[abs(j -2)],ry[j + 7])
                numerator_array10[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j],ry[abs(j -1)],ry[j + 8])
                numerator_array10[8] = (ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3],ry[j+2], ry[j+1],ry[abs(j)],ry[j + 9])
                numerator_array10[9] = (ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1],ry[j + 10])

                for i in range(0,k):
                    denominator_array10[i][0:-1] = numerator_array10[i][0:-1]
                    denominator_array10[i][-1] = ry[abs(j - k + i + 1)]

                phi_k10.append(np.linalg.det(numerator_array10) / np.linalg.det(denominator_array10))

            elif k ==11:
                numerator_array11 = np.zeros((k, k))
                denominator_array11 = np.zeros((k, k))
                numerator_array11[0] = (ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j-6)], ry[abs(j -7)],ry[abs(j -8)],ry[abs(j -9)],  ry[j + 1])
                numerator_array11[1] = (ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],ry[abs(j-5)], ry[abs(j -6)],ry[abs(j -7)],ry[abs(j -8)],ry[j + 2])
                numerator_array11[2] = (ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],ry[abs(j-4)], ry[abs(j -5)],ry[abs(j -6)],ry[abs(j -7)],ry[j + 3])
                numerator_array11[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],ry[abs(j-3)], ry[abs(j -4)],ry[abs(j -5)],ry[abs(j -6)],ry[j + 4])
                numerator_array11[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],ry[abs(j-2)], ry[abs(j -3)],ry[abs(j -4)],ry[abs(j -5)],ry[j + 5])
                numerator_array11[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[abs(j-1)], ry[abs(j -2)],ry[abs(j -3)],ry[abs(j -4)],ry[j + 6])
                numerator_array11[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[abs(j -1)],ry[abs(j -2)],ry[abs(j -3)],ry[j + 7])
                numerator_array11[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j],ry[abs(j -1)],ry[abs(j -2)],ry[j + 8])
                numerator_array11[8] = (ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3],ry[j+2], ry[j+1],ry[abs(j)],ry[abs(j -1)],ry[j + 9])
                numerator_array11[9] = (ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1],ry[j],ry[j + 10])
                numerator_array11[10] = (ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j+2],ry[j+1],ry[j + 11])

                for i in range(0,k):
                    denominator_array11[i][0:-1] = numerator_array11[i][0:-1]
                    denominator_array11[i][-1] = ry[abs(j - k + i + 1)]

                phi_k11.append(np.linalg.det(numerator_array11) / np.linalg.det(denominator_array11))

            elif k ==12:
                numerator_array12 = np.zeros((k, k))
                denominator_array12 = np.zeros((k, k))
                numerator_array12[0] = (ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j-6)], ry[abs(j -7)],ry[abs(j -8)],ry[abs(j -9)],ry[abs(j-10)],  ry[j + 1])
                numerator_array12[1] = (ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],ry[abs(j-5)], ry[abs(j -6)],ry[abs(j -7)],ry[abs(j -8)],ry[abs(j-9)],ry[j + 2])
                numerator_array12[2] = (ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],ry[abs(j-4)], ry[abs(j -5)],ry[abs(j -6)],ry[abs(j -7)],ry[abs(j-8)],ry[j + 3])
                numerator_array12[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],ry[abs(j-3)], ry[abs(j -4)],ry[abs(j -5)],ry[abs(j -6)],ry[abs(j-7)],ry[j + 4])
                numerator_array12[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],ry[abs(j-2)], ry[abs(j -3)],ry[abs(j -4)],ry[abs(j -5)],ry[abs(j-6)],ry[j + 5])
                numerator_array12[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[abs(j-1)], ry[abs(j -2)],ry[abs(j -3)],ry[abs(j -4)],ry[abs(j-5)],ry[j + 6])
                numerator_array12[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[abs(j -1)],ry[abs(j -2)],ry[abs(j -3)],ry[abs(j-4)],ry[j + 7])
                numerator_array12[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j],ry[abs(j -1)],ry[abs(j -2)],ry[abs(j-3)],ry[j + 8])
                numerator_array12[8] = (ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3],ry[j+2], ry[j+1],ry[abs(j)],ry[abs(j -1)],ry[abs(j-2)],ry[j + 9])
                numerator_array12[9] = (ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1],ry[j],ry[abs(j-1)],ry[j + 10])
                numerator_array12[10] = (ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j+2],ry[j+1],ry[abs(j)],ry[j + 11])
                numerator_array12[11] = (ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j+3],ry[j+2],ry[abs(j+1)],ry[j + 12])

                for i in range(0,k):
                    denominator_array12[i][0:-1] = numerator_array12[i][0:-1]
                    denominator_array12[i][-1] = ry[abs(j - k + i + 1)]

                phi_k12.append(np.linalg.det(numerator_array12) / np.linalg.det(denominator_array12))

            elif k ==13:
                numerator_array13 = np.zeros((k, k))
                denominator_array13 = np.zeros((k, k))
                numerator_array13[0] = (ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j-6)], ry[abs(j -7)],ry[abs(j -8)],ry[abs(j -9)],ry[abs(j-10)],ry[abs(j-11)],  ry[j + 1])
                numerator_array13[1] = (ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],ry[abs(j-5)], ry[abs(j -6)],ry[abs(j -7)],ry[abs(j -8)],ry[abs(j-9)],ry[abs(j-10)],ry[j + 2])
                numerator_array13[2] = (ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],ry[abs(j-4)], ry[abs(j -5)],ry[abs(j -6)],ry[abs(j -7)],ry[abs(j-8)],ry[abs(j-9)],ry[j + 3])
                numerator_array13[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],ry[abs(j-3)], ry[abs(j -4)],ry[abs(j -5)],ry[abs(j -6)],ry[abs(j-7)],ry[abs(j-8)],ry[j + 4])
                numerator_array13[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],ry[abs(j-2)], ry[abs(j -3)],ry[abs(j -4)],ry[abs(j -5)],ry[abs(j-6)],ry[abs(j-7)],ry[j + 5])
                numerator_array13[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[abs(j-1)], ry[abs(j -2)],ry[abs(j -3)],ry[abs(j -4)],ry[abs(j-5)],ry[abs(j-6)],ry[j + 6])
                numerator_array13[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[abs(j -1)],ry[abs(j -2)],ry[abs(j -3)],ry[abs(j-4)],ry[abs(j-5)],ry[j + 7])
                numerator_array13[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j],ry[abs(j -1)],ry[abs(j -2)],ry[abs(j-3)],ry[abs(j-4)],ry[j + 8])
                numerator_array13[8] = (ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3],ry[j+2], ry[j+1],ry[abs(j)],ry[abs(j -1)],ry[abs(j-2)],ry[abs(j-3)],ry[j + 9])
                numerator_array13[9] = (ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1],ry[j],ry[abs(j-1)],ry[abs(j-2)],ry[j + 10])
                numerator_array13[10] = (ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j+2],ry[j+1],ry[abs(j)],ry[abs(j-1)],ry[j + 11])
                numerator_array13[11] = (ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j+3],ry[j+2],ry[abs(j+1)],ry[abs(j)],ry[j + 12])
                numerator_array13[12] = (ry[j + 12], ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j +4],ry[j + 3], ry[abs(j + 2)], ry[abs(j+1)],ry[j + 13])

                for i in range(0,k):
                    denominator_array13[i][0:-1] = numerator_array13[i][0:-1]
                    denominator_array13[i][-1] = ry[abs(j - k + i + 1)]

                phi_k13.append(np.linalg.det(numerator_array13) / np.linalg.det(denominator_array13))

            elif k ==14:
                numerator_array14 = np.zeros((k, k))
                denominator_array14 = np.zeros((k, k))
                numerator_array14[0] = (ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j-6)], ry[abs(j -7)],ry[abs(j -8)],ry[abs(j -9)],ry[abs(j-10)],ry[abs(j-11)],ry[abs(j-12)],ry[j + 1])
                numerator_array14[1] = (ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],ry[abs(j-5)], ry[abs(j -6)],ry[abs(j -7)],ry[abs(j -8)],ry[abs(j-9)],ry[abs(j-10)],ry[abs(j-11)],ry[j + 2])
                numerator_array14[2] = (ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],ry[abs(j-4)], ry[abs(j -5)],ry[abs(j -6)],ry[abs(j -7)],ry[abs(j-8)],ry[abs(j-9)],ry[abs(j-10)],ry[j + 3])
                numerator_array14[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],ry[abs(j-3)], ry[abs(j -4)],ry[abs(j -5)],ry[abs(j -6)],ry[abs(j-7)],ry[abs(j-8)],ry[abs(j-9)],ry[j + 4])
                numerator_array14[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],ry[abs(j-2)], ry[abs(j -3)],ry[abs(j -4)],ry[abs(j -5)],ry[abs(j-6)],ry[abs(j-7)],ry[abs(j-8)],ry[j + 5])
                numerator_array14[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[abs(j-1)], ry[abs(j -2)],ry[abs(j -3)],ry[abs(j -4)],ry[abs(j-5)],ry[abs(j-6)],ry[abs(j-7)],ry[j + 6])
                numerator_array14[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[abs(j -1)],ry[abs(j -2)],ry[abs(j -3)],ry[abs(j-4)],ry[abs(j-5)],ry[abs(j-6)],ry[j + 7])
                numerator_array14[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j],ry[abs(j -1)],ry[abs(j -2)],ry[abs(j-3)],ry[abs(j-4)],ry[abs(j-5)],ry[j + 8])
                numerator_array14[8] = (ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3],ry[j+2], ry[j+1],ry[abs(j)],ry[abs(j -1)],ry[abs(j-2)],ry[abs(j-3)],ry[abs(j-4)],ry[j + 9])
                numerator_array14[9] = (ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1],ry[j],ry[abs(j-1)],ry[abs(j-2)],ry[abs(j-3)],ry[j + 10])
                numerator_array14[10] = (ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j+2],ry[j+1],ry[abs(j)],ry[abs(j-1)],ry[abs(j-2)],ry[j + 11])
                numerator_array14[11] = (ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j+3],ry[j+2],ry[abs(j+1)],ry[abs(j)],ry[abs(j-1)],ry[j + 12])
                numerator_array14[12] = (ry[j + 12], ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j +4],ry[j + 3], ry[abs(j + 2)], ry[abs(j+1)],ry[abs(j)],ry[j + 13])
                numerator_array14[13] = (ry[j + 13], ry[j + 12], ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j +5],ry[j + 4], ry[abs(j + 3)], ry[abs(j+2)],ry[abs(j+1)],ry[j + 14])

                for i in range(0,k):
                    denominator_array14[i][0:-1] = numerator_array14[i][0:-1]
                    denominator_array14[i][-1] = ry[abs(j - k + i + 1)]

                phi_k14.append(np.linalg.det(numerator_array14) / np.linalg.det(denominator_array14))

            elif k ==15:
                numerator_array15 = np.zeros((k, k))
                denominator_array15 = np.zeros((k, k))
                numerator_array15[0] = (ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j-6)], ry[abs(j -7)],ry[abs(j -8)],ry[abs(j -9)],ry[abs(j-10)],ry[abs(j-11)],ry[abs(j-12)],ry[abs(j-13)],ry[j + 1])
                numerator_array15[1] = (ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],ry[abs(j-5)], ry[abs(j -6)],ry[abs(j -7)],ry[abs(j -8)],ry[abs(j-9)],ry[abs(j-10)],ry[abs(j-11)],ry[abs(j-12)],ry[j + 2])
                numerator_array15[2] = (ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],ry[abs(j-4)], ry[abs(j -5)],ry[abs(j -6)],ry[abs(j -7)],ry[abs(j-8)],ry[abs(j-9)],ry[abs(j-10)],ry[abs(j-11)],ry[j + 3])
                numerator_array15[3] = (ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],ry[abs(j-3)], ry[abs(j -4)],ry[abs(j -5)],ry[abs(j -6)],ry[abs(j-7)],ry[abs(j-8)],ry[abs(j-9)],ry[abs(j-10)],ry[j + 4])
                numerator_array15[4] = (ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],ry[abs(j-2)], ry[abs(j -3)],ry[abs(j -4)],ry[abs(j -5)],ry[abs(j-6)],ry[abs(j-7)],ry[abs(j-8)],ry[abs(j-9)],ry[j + 5])
                numerator_array15[5] = (ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],ry[abs(j-1)], ry[abs(j -2)],ry[abs(j -3)],ry[abs(j -4)],ry[abs(j-5)],ry[abs(j-6)],ry[abs(j-7)],ry[abs(j-8)],ry[j + 6])
                numerator_array15[6] = (ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],ry[j], ry[abs(j -1)],ry[abs(j -2)],ry[abs(j -3)],ry[abs(j-4)],ry[abs(j-5)],ry[abs(j-6)],ry[abs(j-7)],ry[j + 7])
                numerator_array15[7] = (ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],ry[j+1], ry[j],ry[abs(j -1)],ry[abs(j -2)],ry[abs(j-3)],ry[abs(j-4)],ry[abs(j-5)],ry[abs(j-6)],ry[j + 8])
                numerator_array15[8] = (ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3],ry[j+2], ry[j+1],ry[abs(j)],ry[abs(j -1)],ry[abs(j-2)],ry[abs(j-3)],ry[abs(j-4)],ry[abs(j-5)],ry[j + 9])
                numerator_array15[9] = (ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j+1],ry[j],ry[abs(j-1)],ry[abs(j-2)],ry[abs(j-3)],ry[abs(j-4)],ry[j + 10])
                numerator_array15[10] = (ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j+2],ry[j+1],ry[abs(j)],ry[abs(j-1)],ry[abs(j-2)],ry[abs(j-3)],ry[j + 11])
                numerator_array15[11] = (ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j+3],ry[j+2],ry[abs(j+1)],ry[abs(j)],ry[abs(j-1)],ry[abs(j-2)],ry[j + 12])
                numerator_array15[12] = (ry[j + 12], ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j +4],ry[j + 3], ry[abs(j + 2)], ry[abs(j+1)],ry[abs(j)],ry[abs(j-1)],ry[j + 13])
                numerator_array15[13] = (ry[j + 13], ry[j + 12], ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j + 6], ry[j +5],ry[j + 4], ry[abs(j + 3)], ry[abs(j+2)],ry[abs(j+1)],ry[abs(j)],ry[j + 14])
                numerator_array15[14] = (ry[j + 14], ry[j + 13], ry[j + 12], ry[j + 11], ry[j + 10], ry[j + 9], ry[j + 8], ry[j + 7], ry[j +6],ry[j + 5], ry[abs(j + 4)], ry[abs(j+3)],ry[abs(j+2)],ry[abs(j+1)],ry[j + 15])

                for i in range(0,k):
                    denominator_array15[i][0:-1] = numerator_array15[i][0:-1]
                    denominator_array15[i][-1] = ry[abs(j - k + i + 1)]

                phi_k15.append(np.linalg.det(numerator_array15) / np.linalg.det(denominator_array15))


    phi = []

    phi.append(phi_k1)
    phi.append(phi_k2)
    phi.append(phi_k3)
    phi.append(phi_k4)
    phi.append(phi_k5)
    phi.append(phi_k6)
    phi.append(phi_k7)
    phi.append(phi_k8)
    phi.append(phi_k9)
    phi.append(phi_k10)
    phi.append(phi_k11)
    phi.append(phi_k12)
    phi.append(phi_k13)
    phi.append(phi_k14)
    phi.append(phi_k15)

    phi = np.array(phi[:k1]).T.tolist()
    header_row = np.arange(1,k1+1,1)
    phi.insert(0,header_row)
    gpac_table = tabulate(phi, headers='firstrow', showindex=True)
    print(gpac_table)
    return gpac_table

gpac = cal_gpac(15,15,new_data)

#1st set: k = 8, j = 0
#2nd set: k = 12, j = 0

#Plotting ACF and PACF using statsmodel
acf = sm.tsa.stattools.acf(new_data, nlags=20)
pacf = sm.tsa.stattools.pacf(new_data, nlags=20)

fig = plt.figure(figsize=(6, 7))
fig.tight_layout(pad=4)
plt.subplot(2, 1, 1)
plot_acf(new_data, ax=plt.gca(), lags=20, title="ACF of Stationary Data")
plt.subplot(2, 1, 2)
plot_pacf(new_data, ax=plt.gca(), lags=20, title="PACF of Stationary Data")
plt.show()

#Levenberg Marquadt algorithm=================

#ARMA Parameter Estimation


def LM(y,na,nb,lags):
    model = sm.tsa.ARMA(y,(na,nb)).fit(trend='nc',disp=0)
    for i in range(na):
        print("AR Coefficient a{}".format(i), "is:", model.params[i])
    for i in range(nb):
        print("MA Coefficient b{}".format(i), "is:", model.params[i+na])
    print(model.summary())

    #Prediction
    y_train_set = y[:2415]
    y_test_set = y[2415:]
    model_hat = model.predict(start=0,end=len(y_train_set)-1)
    model_forecast = model.predict(start=2415, end = len(y)-1)

    #Residuals Testing and Chi-square test
    e = y_train_set - model_hat
    forecast_error = y_test_set - model_forecast
    re = calc_autocorrelation_coef(e,lags,f"ACF of Residuals for ARMA({na},{nb})")
    Q = len(e)*np.sum(np.square(re[lags:]))
    fore_re = calc_autocorrelation_coef(forecast_error.to_list(),lags,f"ARMA({na},{nb}) Forecast Error ACF")
    fore_Q = len(forecast_error)*np.sum(np.square(fore_re[lags:]))

    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1-alfa,DOF)

    #Printing Prediction Error Statistical Measures
    print(f"ARMA({na},{nb}) Prediction Error MSE:",calc_mse(calc_error_squared(e)))
    print(f"ARMA({na},{nb}) Prediction Error Q:", Q)
    print(f"ARMA({na},{nb}) Prediction Error Variance: ", np.var(e))
    print(f"ARMA({na},{nb}) Prediction Error Mean:",np.mean(e))

    #Printing Forecast Error Statistical Measures
    arma_forecast_error_mse = calc_mse(calc_error_squared(forecast_error))
    print(f"ARMA({na},{nb}) Forecast Error MSE:",arma_forecast_error_mse)
    print(f"ARMA({na},{nb}) Forecast Error Q:", fore_Q)
    arma_forecast_error_var = np.var(forecast_error)
    print(f"ARMA({na},{nb}) Forecast Error Variance: ", arma_forecast_error_var)
    arma_forecast_error_mean = np.mean(forecast_error)
    print(f"ARMA({na},{nb}) Forecast Error Mean:",arma_forecast_error_mean)

    if Q < chi_critical:
        print("Residual is white")
    else:
        print("The Residual is NOT white")

    if fore_Q < chi_critical:
        print("Forecast Error is white")
    else:
        print("Forecast Error is NOT white")

    lbvalue, pvalue = sm.stats.acorr_ljungbox(e, lags = [lags])
    print("lbvalue",lbvalue)
    print("pvalue",pvalue)

    #Plotting
    plt.figure()
    plt.plot(y,'r',label='True Data')
    plt.plot(model_hat,'b',label = "Prediction")
    plt.plot(model_forecast,'g',label='Forecast')
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title(f"ARMA({na},{nb}) Prediction and Forecast vs. Raw Data")
    plt.show()

    #Plotting Hstep
    plt.figure()
    plt.plot(y_test_set,'r',label='True Data')
    #plt.plot(model_hat,'b',label = "Prediction")
    plt.plot(model_forecast,'g',label='Forecast')
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title(f"ARMA({na},{nb}) Prediction and Forecast vs. Raw Data")
    plt.show()

    return arma_forecast_error_mse,fore_Q,arma_forecast_error_var,arma_forecast_error_mean

arma8_forecast_error_mse,arma8_forecast_error_Q,arma8_forecast_error_var,arma8_forecast_error_mean = LM(stock,8,0,20)
arma12_forecast_error_mse,arma12_forecast_error_Q,arma12_forecast_error_var,arma12_forecast_error_mean = LM(stock,12,0,20)

#Diagnostic Analysis======================================




#Base-models=============================================

y_train_list = y_train.to_list()
y_test_list = y_test.to_list() #604

#Predict Average Forecast
def calc_average_forecast(training_data,testing_data):
    forecast =[]
    for i in range(1,len(testing_data)):
        forecast.append(sum(training_data)/(len(training_data)-1))
    return forecast

average_forecast = calc_average_forecast(y_train_list,y_test_list) #603
average_forecast_error = calc_error(y_test,average_forecast)
average_forecast_error_squared =calc_error_squared(average_forecast_error)
average_forecast_error_mse = calc_mse(average_forecast_error_squared)#261615.53316583813
print("Average Forecast Error MSE: ",average_forecast_error_mse)
average_forecast_error_acf = calc_autocorrelation_coef(average_forecast_error,20,'Average Forecast Error ACF')
average_forecast_error_Q = calc_q(average_forecast_error_acf,len(average_forecast_error))
print("Average Forecast Error Q: ",average_forecast_error_Q)
average_forecast_error_variance = np.var(average_forecast_error)
print("Average Forecast Error Variance: ",average_forecast_error_variance)
average_forecast_error_mean = np.mean(average_forecast_error)
print("Average Forecast Error Mean: ",average_forecast_error_mean)

def calc_naive_forecast(training_data,test_data):
    forecast = []
    for i in range(0,len(test_data)):
        forecast.append(training_data[-1])
    return forecast

naive_forecast = calc_naive_forecast(y_train_list,y_test_list) #604
naive_forecast_error = calc_error(y_test,naive_forecast)
naive_forecast_error_squared =calc_error_squared(naive_forecast_error)
naive_forecast_error_mse = calc_mse(naive_forecast_error_squared)#39334.89404387418
print("Naive Forecast Error MSE: ",naive_forecast_error_mse)
naive_forecast_error_acf = calc_autocorrelation_coef(naive_forecast_error,20,'Naive Forecast Error ACF')
naive_forecast_error_Q = calc_q(naive_forecast_error_acf,len(naive_forecast_error))
print("Naive Forecast Error Q: ",naive_forecast_error_Q)
naive_forecast_error_variance = np.var(naive_forecast_error)
print("Naive Forecast Error Variance: ",naive_forecast_error_variance)
naive_forecast_error_mean = np.mean(naive_forecast_error)
print("Naive Forecast Error Mean: ",naive_forecast_error_mean)


#Calculate Drift Method Forecast
def calc_drift_forecast(training_set,test_set):
    forecast = []
    for i in range(0,len(test_set)):
        forecast.append(((training_set[-1] - training_set[0])/(len(training_set)-1))*((i+10)-1) + training_set[0])
    return forecast

drift_forecast = calc_drift_forecast(y_train_list,y_test_list) #604
drift_forecast_error = calc_error(y_test,drift_forecast)
drift_forecast_error_squared =calc_error_squared(drift_forecast_error)
drift_forecast_error_mse = calc_mse(drift_forecast_error_squared)#39334.89404387418
print("Drift Forecast Error MSE: ",drift_forecast_error_mse)
drift_forecast_error_acf = calc_autocorrelation_coef(drift_forecast_error,20,'Drift Forecast Error ACF')
drift_forecast_error_Q = calc_q(drift_forecast_error_acf,len(drift_forecast_error))
print("Drift Forecast Error Q: ",drift_forecast_error_Q)
drift_forecast_error_variance = np.var(drift_forecast_error)
print("Drift Forecast Error Variance: ",drift_forecast_error_variance)
drift_forecast_error_mean = np.mean(drift_forecast_error)
print("Drift Forecast Error Mean: ",drift_forecast_error_mean)


#Calculating SES
def calc_ses_method_prediction(training_set,alpha):
    prediction = []
    for i in range(0,len(training_set)):
        if i == 0:
            prediction.append((alpha*training_set[i]) + (1-alpha)*training_set[i])
        elif i>0:
            prediction.append((alpha * training_set[i]) + (1 - alpha) * prediction[i-1])
    return prediction
alpha = 0.5

ses_forecast = [calc_ses_method_prediction(y_train_list,alpha)[-1]]*len(y_test_list) #604
ses_forecast_error = calc_error(y_test,ses_forecast)
ses_forecast_error_squared =calc_error_squared(ses_forecast_error)
ses_forecast_error_mse = calc_mse(ses_forecast_error_squared)#39334.89404387418
print("SES Forecast Error MSE: ",ses_forecast_error_mse)
ses_forecast_error_acf = calc_autocorrelation_coef(ses_forecast_error,20,f'SES Forecast Error ACF (alpha={alpha})')
ses_forecast_error_Q = calc_q(ses_forecast_error_acf,len(ses_forecast_error))
print("SES Forecast Error Q: ",ses_forecast_error_Q)
ses_forecast_error_variance = np.var(ses_forecast_error)
print("SES Forecast Error Variance: ",ses_forecast_error_variance)
ses_forecast_error_mean = np.mean(ses_forecast_error)
print("SES Forecast Error Mean: ",ses_forecast_error_mean)


#Calculating Holt's Linear Method
def calc_holt_linear_forecast(y_train,y_test):
    aapl_holtt = ets.ExponentialSmoothing(y_train.values, trend='additive',damped=False,seasonal=None).fit()
    aapl_holtf = aapl_holtt.forecast(steps=len(y_test))
    aapl_holtf = pd.DataFrame(aapl_holtf).set_index(y_test.index)
    #aapl_holt_mse = np.square(np.subtract(y_test.values,np.ndarray.flatten(aapl_holtf.values))).mean()
    aapl_forecast_error_holt = calc_error(y_test,aapl_holtf.values)
    forecast_error_squared = calc_error_squared(aapl_forecast_error_holt)
    forecast_error_mse = calc_mse(forecast_error_squared)

    return aapl_holtf, forecast_error_mse, aapl_forecast_error_holt

holt_linear_forecast, holt_linear_forecast_error_mse, holtlinear_forecast_error = calc_holt_linear_forecast(y_train,y_test) #604
print("Holt-Linear Forecast Error MSE: ",holt_linear_forecast_error_mse)
holt_linear_forecast_error_acf = calc_autocorrelation_coef(holtlinear_forecast_error,20,'Holt-Linear Forecast Error ACF')
holt_linear_forecast_error_Q = calc_q(holt_linear_forecast_error_acf,len(holtlinear_forecast_error))
print("Holt-Linear Forecast Error Q: ",holt_linear_forecast_error_Q)
holt_linear_forecast_error_variance = np.var(holtlinear_forecast_error)
print("Holt-Linear Forecast Error Variance: ",holt_linear_forecast_error_variance)
holt_linear_forecast_error_mean = np.mean(holtlinear_forecast_error)
print("Holt-Linear Forecast Error Mean: ",holt_linear_forecast_error_mean)


#Plotting Base Models
fig = plt.figure(figsize=(16,10))
#fig.tight_layout(pad = 4)
ax1 = fig.add_subplot(3,2,1)
ax1.plot(X_train,y_train, label = 'Training dataset')
ax1.plot(X_test,y_test, label = 'Testing dataset')
ax1.plot(X_test[:-1],average_forecast, label = 'Average Method h-step forecast')
ax1.set_xticks(ax.get_xticks()[::1])
plt.xticks(rotation=45)
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Magnitude")
ax1.legend(loc = 'upper left')
ax1.set_title("Google Stock Close Price Average Forecasting")

ax2 = fig.add_subplot(3,2,2)
ax2.plot(X_train,y_train, label = 'Training dataset')
ax2.plot(X_test,y_test, label = 'Testing dataset')
ax2.plot(X_test,naive_forecast, label = 'Naive Method h-step forecast')
ax2.set_xticks(ax.get_xticks()[::1])
plt.xticks(rotation=45)
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Magnitude")
ax2.legend(loc = 'upper left')
ax2.set_title("Google Stock Close Price Naive Forecasting")

ax3 = fig.add_subplot(3,2,3)
ax3.plot(X_train,y_train, label = 'Training dataset')
ax3.plot(X_test,y_test, label = 'Testing dataset')
ax3.plot(X_test,drift_forecast, label = 'Drift Method h-step forecast')
ax3.set_xticks(ax.get_xticks()[::1])
plt.xticks(rotation=45)
ax3.set_xlabel("Time (t)")
ax3.set_ylabel("Magnitude")
ax3.legend(loc = 'upper left')
ax3.set_title("Google Stock Close Price Drift Forecasting")

ax4 = fig.add_subplot(3,2,4)
ax4.plot(X_train,y_train, label = 'Training dataset')
ax4.plot(X_test,y_test, label = 'Testing dataset')
ax4.plot(X_test,ses_forecast, label = 'SES Method h-step forecast')
ax4.set_xticks(ax.get_xticks()[::1])
plt.xticks(rotation=45)
ax4.set_xlabel("Time (t)")
ax4.set_ylabel("Magnitude")
ax4.legend(loc = 'upper left')
ax4.set_title(f"Google Stock Close Price SES Forecasting (alpha={alpha})")

ax5 = fig.add_subplot(3,2,5)
ax5.plot(X_train,y_train, label = 'Training dataset')
ax5.plot(X_test,y_test, label = 'Testing dataset')
ax5.plot(X_test,holt_linear_forecast, label = 'Holt-Linear Method h-step forecast')
ax5.set_xticks(ax.get_xticks()[::1])
plt.xticks(rotation=45)
ax5.set_xlabel("Time (t)")
ax5.set_ylabel("Magnitude")
ax5.legend(loc = 'upper left')
ax5.set_title("Google Stock Close Price Holt-Linear Forecasting")

ax6 = fig.add_subplot(3,2,6)
#fig,ax = plt.subplots(figsize = (10,8))
ax6.plot(X_train,y_train, label = 'Training dataset')
ax6.plot(X_test,y_test, label = 'Testing dataset')
ax6.plot(X_test,stock_hw_forecast, label = 'Holt-Winter Method h-step forecast')
ax6.set_xticks(ax.get_xticks()[::1])
plt.xticks(rotation=45)
ax6.set_xlabel("Time (t)")
ax6.set_ylabel("Magnitude")
ax6.legend(loc = 'upper left')
ax6.set_title("Google Stock Close Price Holt-Winter Forecasting")

fig.tight_layout(pad = 4)
plt.show()

#=========================================================

#Creating DataFrame of Each Model Statistical Measures

model_stats = pd.DataFrame({'Forecast Error Mean': [average_forecast_error_mean,naive_forecast_error_mean,
                                                    drift_forecast_error_mean,ses_forecast_error_mean,
                                                    holt_linear_forecast_error_mean,hw_forecast_error_mean,
                                                    mlr_forecast_error_mean,arma8_forecast_error_mean,
                                                    arma12_forecast_error_mean],
                            'Forecast Error Variance':[average_forecast_error_variance,naive_forecast_error_variance,
                                                       drift_forecast_error_variance,ses_forecast_error_variance,
                                                       holt_linear_forecast_error_variance,hw_forecast_error_variance,
                                                       mlr_forecast_error_variance,arma8_forecast_error_var,
                                                       arma12_forecast_error_var],
                            'Forecast Error MSE':[average_forecast_error_mse,naive_forecast_error_mse,
                                                  drift_forecast_error_mse,ses_forecast_error_mse,
                                                  holt_linear_forecast_error_mse,stock_hw_forecast_error_mse,
                                                  mlr_forecast_error_mse,arma8_forecast_error_mse,
                                                  arma12_forecast_error_mse],
                            'Forecast Error Q':[average_forecast_error_Q,naive_forecast_error_Q,
                                                  drift_forecast_error_Q,ses_forecast_error_Q,
                                                  holt_linear_forecast_error_Q,hw_forecast_error_Q,
                                                  stock_mlr_q,arma8_forecast_error_Q,arma12_forecast_error_Q]},
                           index = ['Average','Naive','Drift','Simple Exponential Smoothing','Holt-Linear',
                                    'Holt-Winters','Multiple Linear Regression','ARMA(8,0)','ARMA(12,0)'])

print(model_stats)