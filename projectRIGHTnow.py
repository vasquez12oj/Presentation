#Jupyter Notebook Program:



import pickle

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error



# -----------------------------

# 1. Load data

# -----------------------------

data = pd.read_csv("cust_latest.csv")



# -----------------------------

# 2. Clean and prepare date

# -----------------------------

data["PO Date"] = pd.to_datetime(data["PO Date"])



data["day"] = data["PO Date"].dt.day

data["month"] = data["PO Date"].dt.month

data["year"] = data["PO Date"].dt.year



# -----------------------------

# 3. Define X and y

# -----------------------------

target_col = "Bill Qty"



feature_cols = [

"day",

"month",

"year",

"Brand Name",

"Material Family"

]



X = data[feature_cols]

y = data[target_col]



# -----------------------------

# 4. Identify column types

# -----------------------------

numeric_features = ["day", "month", "year"]

categorical_features = ["Brand Name", "Material Family"]



preprocessor = ColumnTransformer(

transformers=[

("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),

("num", "passthrough", numeric_features)

]

)



# -----------------------------

# 5. Train/test split

# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.33, random_state=42

)



# -----------------------------

# 6. Create models

# -----------------------------

lin_model = Pipeline([

("preprocessor", preprocessor),

("model", LinearRegression())

])



rf_model = Pipeline([

("preprocessor", preprocessor),

("model", RandomForestRegressor(

n_estimators=100,

max_depth=8,

random_state=42

))

])



svr_model = Pipeline([

("preprocessor", preprocessor),

("model", SVR(C=10, epsilon=0.1, kernel="rbf"))

])



voting_model = VotingRegressor([

("lin", lin_model),

("rf", rf_model),

("svr", svr_model)

])



models = {

"LinearRegression": lin_model,

"RandomForest": rf_model,

"SVR": svr_model,

"VotingRegressor": voting_model

}



# ----------------------    -------

# 7. Train and eval    uate

# ----------------------    -------

for name, model in models.items    ():

model.fit(X_train, y_train)

preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)

print(f"{name} MSE: {mse:.4f}")



# ---------------------    --------

# 8.     Save model

# --------    ---------------------

wit    h open("predicted_sales_model.pkl", "wb") as f:

pickle.dump({
    
"model": voting_model,

"features": feature_cols,

"brand_options": sorted(data["Brand Name"].dropna().unique()),

"material_options": sorted(data["Material Family"].dropna().unique())

}, f)



print("Saved predicted_sales_model.pkl")
