from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math as m


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "Predict.html")


def result(request):
    housing_data = pd.read_csv("C:/Users/SADVIKA/Downloads/Housing.csv")
    housing_data = housing_data.drop('furnishingstatus', axis=1)
    varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    def binary_map(x):
        return x.map({'yes': 1, "no": 0})

    housing_data[varlist] = housing_data[varlist].apply(binary_map)

    X = housing_data.drop('price', axis='columns')
    Y = housing_data['price']


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8= float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])
    var11 = float(request.GET['n11'])


    pred = model.predict(np.array([var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11]).reshape(1,-1))
    pred = round(pred[0])



    price = "The predicted price is $" + str(pred)

    return render(request, "Predict.html", {"result2":price})
