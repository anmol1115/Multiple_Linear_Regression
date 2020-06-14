#importing relevant libraries and files
import pandas as pd
import numpy as np
train=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Train.csv')
test=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Test.csv')

#assigning training and test sets
X_train=train.iloc[:,:-1].values
y_train=train.iloc[:,-1].values
X_test=test.iloc[:,:].values

#creating regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

#checking the accuracy of the model
import statsmodels.regression.linear_model as sm
X_opt=X_train[:,[0,1,2,3,4]]
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt.astype(float)).fit()
regressor_OLS.summary()

#The key takeaway from the last section is the p-value for each feature is well 
#within the the range. Along with that both R-squared and Adjusted R-squared are 
#pretty close to 1.