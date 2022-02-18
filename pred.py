import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xg

def prediction(Fcatgeory, Ftype, Fyear, Fmonth):
    
    # Read CSV
    df = pd.read_csv("./data.csv")
    df = df.iloc[:,:5]
    # Converting string variable to int. LabelEncoder could have been used here
    le = preprocessing.LabelEncoder()
    le.fit(df["MONATSZAHL"])
    df["MONATSZAHL"] = le.transform(df["MONATSZAHL"])

    le1 = preprocessing.LabelEncoder()
    le1.fit(df["AUSPRAEGUNG"])
    df["AUSPRAEGUNG"] = le1.transform(df["AUSPRAEGUNG"])

    # Removing Summe from MONAT Column
    df = df[~df["MONAT"].isin(["Summe"])]
    # Extracting only Month value
    df["MONAT"] = df["MONAT"].apply(lambda x : x[4:])
    # Dropping Na
    df.dropna(inplace = True)

    # X,y values
    X = df.iloc[:,:4].values
    y = df.iloc[:,-1].values

    Fcatgeory = le.transform([Fcatgeory])[0]
    Ftype = le1.transform([Ftype])[0]

    input = np.array([Fcatgeory, Ftype, Fyear, Fmonth])
    input = np.reshape(input, (1,4))
    # Advanced Ensemble Learning Model - xGBoost 

    # Instantiation
    xgb_r = xg.XGBRegressor(n_estimators = 10, seed = 123)
    # Fitting the model
    xgb_r.fit(X, y)
    # Predict the model
    pred = xgb_r.predict(input)

    return pred
    