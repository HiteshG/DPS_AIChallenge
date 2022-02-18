import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xg

# Read CSV
df = pd.read_csv("./data.csv")
df = df.iloc[:,:5]
# Converting string variable to int. LabelEncoder could have been used here
df["MONATSZAHL"] = df["MONATSZAHL"].map({"Alkoholunfälle": "0", "Fluchtunfälle": "1","Verkehrsunfälle": "2"})
df["AUSPRAEGUNG"] = df["AUSPRAEGUNG"].map({"insgesamt": "0", "Verletzte und Getötete": "1","mit Personenschäden": "2"})

# Removing Summe from MONAT Column
df = df[~df["MONAT"].isin(["Summe"])]
# Extracting only Month value
df["MONAT"] = df["MONAT"].apply(lambda x : x[4:])
# Dropping Na
df.dropna(inplace = True)

# Grouping by MONATSZAHL and taking final sum
cat1 = df.groupby(["MONATSZAHL"], as_index = False)["WERT"].sum()
sns.barplot(x = "MONATSZAHL", y ="WERT", data = cat1)

# X,y values
X = df.iloc[:,:4].values
y = df.iloc[:,-1].values

# For Modelling splitting train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle = True)

# Base Model R2 score = 64.47
# reg = LinearRegression().fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# r2_score(y_test, y_pred)

# Advanced Ensemble Learning Model - xGBoost with R2 = 98.95

# Instantiation
xgb_r = xg.XGBRegressor(n_estimators = 10, seed = 123)
# Fitting the model
xgb_r.fit(X_train, y_train)
# Predict the model
pred = xgb_r.predict(X_test)

r2_score(y_test, pred)