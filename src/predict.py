import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
data = pd.read_csv("c://git/arbeit/data/mpg.csv",sep=";")

x_train, x_test, y_train, y_test = train_test_split(data.drop(["mpg"],axis=1),data["mpg"])

knn = KNeighborsRegressor()
knn.fit(x_train, y_train)

filename = "c://git/arbeit/data/finalized_model.sav"

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

