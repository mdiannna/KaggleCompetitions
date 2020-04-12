import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression


TRAIN_DATA_FILE = "data/train.csv"
TEST_DATA_FILE = "data/test.csv"
SAMPLE_SUBMISSION_FILE = "data/sample_submission.csv"
index = "3"
RESULTS_FILE = "data/submissions/" + str(index) + ".csv"

def read_data():
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
	print(data_selected.size)

	print(data_predicted.head())
	print(data_predicted.size)
	# 1460 rinduri total
	# print(data_predicted.iloc[14])
	return data_selected, data_predicted

X, y = read_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=42, shuffle=True)

print("X_train size:")
print(X_train.size)
print("y_train size:")
print(y_train.size)

print("X_test size:")
print(X_test.size)
print("Y_test size:")
print(y_test.size)


X_test = X_test.interpolate()


############################
# Train model
#############################

model = LinearRegression().fit(X_train, y_train)

def predictModel(X_test, model):
	y_predicted = model.predict(X_test)

	score_test  = model.score(X_test, y_test)
	print("Score test:")
	print(score_test)
	print(y_predicted[0:10])
	print(y_test[0:10])
	print("MSE:")
	print(mean_squared_error(y_test, y_predicted))
	return y_predicted

y_predicted = predictModel(X_test, model)

def calcError(y_test, y_predicted):
	print("Root Mean Squared Logarithmic Error (RMSE):")
	rmse = math.sqrt(mean_squared_log_error(y_predicted, y_test))
	print(rmse)
	return rmse

# calcError(y_test, y_predicted)



def predictSubmission(model):
	data_test = pd.read_csv(TEST_DATA_FILE)

	final_data = pd.DataFrame(data_test["Id"], columns =['Id', 'SalePrice'])
	# final_data["Id"] = data_test["Id"]

	data_test= data_test[["LotArea", "OverallQual", "OverallCond", "TotalBsmtSF", "GrLivArea", "Fireplaces", "GarageArea"]]
	# data_test = data_test.fillna(0)
	data_test = data_test.interpolate()

	# TODO: remove NaNs before prediction
	final_data["SalePrice"] = model.predict(data_test)

	print(final_data.head())

	final_data.to_csv(RESULTS_FILE, index=False)

predictSubmission(model)