# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:01:35 2023

@author: rkb19187
"""
import matplotlib.pyplot as plt
import pandas, sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
 
parameters = pandas.read_csv("Judred.csv", index_col=0)
#print(parameters)
targets = pandas.read_csv("APs.csv", index_col = 0)
#print(targets)

Forcefield = "2.1"
targets = targets[targets["FF"] == Forcefield]
targets.index = targets["pep"]

X_train, X_val, y_train, y_val = train_test_split(parameters, targets, test_size=0.33, random_state=9876, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=9876, shuffle=True)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# Add kernal trick
X_train = np.e ** -(0.2*X_train)**2
X_test = np.e ** -(0.2*X_test)**2
X_val = np.e ** -(0.2*X_val)**2

BestHyperparameters = pandas.DataFrame(columns = ["fit_intercept", "positive", "Test data RMSE"])
iteration = 0
for fit_intercept in [True, False]:
    for positive in [True, False]:
        Hyperparameters = {"fit_intercept": fit_intercept,
                           "positive": positive}
        
        model = LinearRegression(**Hyperparameters)

        #Train on the training data
        model.fit(X_train, y_train["mean"])
        
        # Test on the Test data
        predictions = model.predict(X_test)
        
        BestHyperparameters.loc[iteration] = [fit_intercept, positive, mean_squared_error(y_test["mean"], predictions, squared=False)]
        iteration +=1
        
      
BestHyperparameters = BestHyperparameters.sort_values("Test data RMSE", ascending=False)
print(BestHyperparameters)

Hyperparameters = {"fit_intercept": BestHyperparameters.iloc[-1]["fit_intercept"],
                   "positive": BestHyperparameters.iloc[-1]["positive"]}
        
# Validate on the never before seen in any way validation data
predictions = model.predict(X_val)
RMSE = mean_squared_error(y_val["mean"], predictions, squared=False)

plt.scatter(predictions, y_val["mean"])
plt.plot([1,2.7], [1,2.7], lw=1, c="black")
plt.title(f"RMSE = {round(RMSE,3)}")
plt.xlabel("Predicted AP")
plt.ylabel("True AP")
plt.show()








