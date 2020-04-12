import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split


TRAIN_DATA_FILE = "data/train.csv"
TEST_DATA_FILE = "data/test.csv"
SAMPLE_SUBMISSION_FILE = "data/sample_submission.csv"
index = 1
RESULTS_FILE = "data/submission" + str(index) + ".csv"

data = pd.read_csv(TRAIN_DATA_FILE)
print(data.head())  



# Utile:
# LotArea, OverallQual(1-10), OverallCond(1-9), BsmtFinSF1, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, 
# GrLivArea(334-5642), Fireplaces(0-3), GarageArea(0-1418)

# CentralAir - Y/N, 

# BsmtQual Gd, TA, Ex, N/a, etc?

# YearBuilt,
# HouseStyle- 1story50%, 2 story30%


#Precis inutile:
# Street, Utilities, RoofMatl 98% 1%, 

# Alley?


data_selected = data[["LotArea", "OverallQual", "OverallCond", "TotalBsmtSF", "GrLivArea", "Fireplaces", "GarageArea"]]
data_predicted = data[["SalePrice"]]
# equivalent to GrLivArea:
# data_selected["FlrSFTotal"] = data["1stFlrSF"] + data["2ndFlrSF"]

print(data_selected.head())
# print(data_selected.info)
print(data_selected.size)

print(data_predicted.head())
# print(data_predicted.info)
print(data_predicted.size)
# 1460 rinduri total
# print(data_predicted.iloc[14])

X = data_selected
y = data_predicted

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("X_train size:")
print(X_train.size)
print("y_train size:")
print(y_train.size)

print("X_test size:")
print(X_test.size)
print("Y_test size:")
print(y_test.size)




############################
# Train model
#############################
import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
reg.score(X, y)

y_predicted = reg.predict(X_test)

score_test  = reg.score(X_test, y_test)
print("Score test:")
print(score_test)
print(y_predicted[0:10])
print(y_test[0:10])




#  Test on test data from kaggle:

data_test = pd.read_csv(TEST_DATA_FILE)

final_data = pd.DataFrame(data_test["Id"], columns =['Id', 'SalePrice'])
# final_data["Id"] = data_test["Id"]


data_test= data_test[["LotArea", "OverallQual", "OverallCond", "TotalBsmtSF", "GrLivArea", "Fireplaces", "GarageArea"]]
print(data_test.head())
# print(data_test["Id"])

final_data["SalePrice"] = reg.predict(data_test)

print(final_data.head())
# print(final_data.head())

# final_data = pd.dataframe["Id", "SalePrice"]
# final_data["SalePrice"] = y_test
# final_data["Id"] = test_Ids

# df1 = pd.read_csv(SAMPLE_SUBMISSION_FILE)
# print(df1.head())  
# predicted data format:
# Id, SalePrice
