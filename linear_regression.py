#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.svm import SVR
#from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def sickness_list():
  return ["ARTHRITIS (Sickness)", "BPHIGH (Sickness)", "CANCER (Sickness)",
          "CASTHMA (Sickness)", "CHD (Sickness)", "COPD (Sickness)",
          "DIABETES (Sickness)", "HIGHCHOL (Sickness)", "KIDNEY (Sickness)",
          "MHLTH (Sickness)", "PHLTH (Sickness)", "STROKE (Sickness)",
          "TEETHLOST (Sickness)"]

def prevention_list():
  return ["ACCESS (Not Prevention)", "BPMED (Not Prevention)",
          "CHECKUP (Not Prevention)", "CHOLSCREEN (Not Prevention)",
          "COLON_SCREEN (Not Prevention)", "COREM (Not Prevention)",
          "COREW (Not Prevention)", "DENTAL (Not Prevention)",
          "MAMMOUSE (Not Prevention)", "PAPTEST (Not Prevention)"]

def negative_behavior_list():
  return ["BINGE (Negative_Behavior)", "CSMOKING (Negative_Behavior)",
          "LACK_PHYSICAL_ACTIVE (Negative_Behavior)",
          "OBESITY (Negative_Behavior)", "SLEEP (Negative_Behavior)"]

def others_list():
  return ["Population"]


if __name__ == "__main__":

  df = pd.read_csv("500_Cities_Better_Health_Transpose_Not_Prevention.csv",
                   sep=",")
  #df.head()
  #print(df.head().keys)
  dataAttributes = []
  dataAttributes = dataAttributes+sickness_list()
  dataAttributes = dataAttributes+prevention_list()
  dataAttributes = dataAttributes+negative_behavior_list()
  dataAttributes = dataAttributes+others_list()

  for attributeTarget in dataAttributes:
    y = df[attributeTarget]

    for attributePredictor in dataAttributes:
      if (attributePredictor == attributeTarget):
        continue
      x = df[attributePredictor]

      x_train, x_test, y_train, y_test = (
          train_test_split(x, y, test_size=0.3, random_state=4)
      )


      regr = linear_model.LinearRegression()
      if ((x_train.ndim < 2) or (x_test.ndim < 2)):
          # single column
          regr.fit(x_train.values.reshape(-1,1), y_train.values)
          y_pred_regr = regr.predict(x_test.values.reshape(-1,1))
      else:
          #multiple columns process
          regr.fit(x_train.values, y_train.values)
          y_pred_regr = regr.predict(x_test.values)


      try:
        coefficient = ("%.2f" % regr.coef_)
      except Exception as e:
        coefficient = "NULL"

      try:
        absoluteError = ("%.2f" 
                          % mean_absolute_error(y_test.values, y_pred_regr))
      except Exception as e:
        absoluteError = "NULL"

      try:
        squaredError = ("%.2f" 
                         % mean_squared_error(y_test.values, y_pred_regr))
      except Exception as e:
        squaredError = "NULL"

      try:
        correlation = ("%.2f" 
                        % np.sqrt(r2_score(y_test.values, y_pred_regr)))
      except Exception as e:
        correlation = "NULL"

      try:
        variance = ("%.2f" 
                     % r2_score(y_test.values, y_pred_regr))
      except Exception as e:
        variance = "NULL"

      print("\"{0}\",\"{1}\",{2},{3},{4},{5},{6}"
            .format(attributeTarget, attributePredictor, coefficient,
                    absoluteError, squaredError, correlation, variance))

    #lasso = linear_model.Lasso(alpha=0.001)
    #lasso.fit(x_train.values.reshape(-1,1), y_train.values)
    #svr = SVR(C=8, epsilon=0.2, gamma=0.5)
    #svr.fit(x_train.values.reshape(-1,1), y_train.values)
    #y_pred_lasso = lasso.predict(x_test.values.reshape(-1,1))
    #y_pred_svr = svr.predict(x_test.values.reshape(-1,1))

    #print("~~ Lasso ~~")
    #print("Mean absolute error: %.2f"
    #      % mean_absolute_error(y_test.values, y_pred_lasso))
    #print("Mean squared error: %.2f"
    #      % mean_squared_error(y_test.values, y_pred_lasso))
    #print("Correlation: %.2f" % np.sqrt(r2_score(y_test.values, y_pred_lasso)))
    #print("Variance score: %.2f" % r2_score(y_test.values, y_pred_lasso))

    #print("~~ SVR ~~")
    #print("Mean absolute error: %.2f"
    #      % mean_absolute_error(y_test.values, y_pred_svr))
    #print("Mean squared error: %.2f"
    #      % mean_squared_error(y_test.values, y_pred_svr))
    #print("Correlation: %.2f" % np.sqrt(r2_score(y_test.values, y_pred_svr)))
    #print("Variance score: %.2f" % r2_score(y_test.values, y_pred_svr))


#mode
#round(max(set(pierDiffList), key=pierDiffList.count), 3)
#median
#round(np.median(pierDiffList), 3)
#mean
#round(np.mean(pierDiffList), 3)
#variance
#round(np.var(pierDiffList), 3)
#standard deviation
#round(np.std(pierDiffList), 3)


