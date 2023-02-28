# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:01:35 2023

@author: rkb19187
"""
import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
 
parameters = pandas.read_csv("Judred.csv", index_col=0)
#print(parameters)
targets = pandas.read_csv("APs.csv", index_col = 0)
#print(targets)

Forcefield = "2.1"
targets = targets[targets["FF"] == Forcefield]
targets.index = targets["pep"]

X_train, X_val, y_train, y_val = train_test_split(parameters, targets, test_size=0.33, random_state=9876, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=9876, shuffle=True)


BestHyperparameters = pandas.DataFrame(columns = ["fit_intercept", "positive", "Test data r2"])
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
        
        BestHyperparameters.loc[iteration] = [fit_intercept, positive, r2_score(y_test["mean"], predictions)]
        iteration +=1
        
      
BestHyperparameters = BestHyperparameters.sort_values("Test data r2")
print(BestHyperparameters)

Hyperparameters = {"fit_intercept": BestHyperparameters.iloc[-1]["fit_intercept"],
                   "positive": BestHyperparameters.iloc[-1]["positive"]}
        
# Validate on the never before seen in any way validation data
predictions = model.predict(X_val)
r2 = r2_score(y_val["mean"], predictions)

plt.scatter(predictions, y_val["mean"])
plt.plot([1,2.7], [1,2.7], lw=1, c="black")
plt.title(f"r2 = {round(r2,1)}")
plt.xlabel("Predicted AP")
plt.ylabel("True AP")
plt.show()