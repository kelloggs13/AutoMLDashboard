
import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import pygwalker as pyg
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from plotnine import *
import altair as alt

st.set_page_config(layout = "wide")

with st.sidebar:
  input_data = st.file_uploader('uplod data')
  input_filename, input_file_extension = os.path.splitext(input_data.name)

def read_data():
  if input_file_extension == ".csv":
    df = pd.read_csv(input_data)
  elif input_file_extension == ".xlsx":
    df = pd.read_excel(input_data)
  else:
    st.warning("Uploaded file must be either .csv or .xlsx")
  return df

if input_data is not None:
  df_input = read_data()
  
# show un-processed data
st.header("Uploaded Data")
st.write(df_input.head(3))

# DataViz with pygwalker
db_pyg = pyg.walk(df_input, return_html=True)

with st.sidebar:
  btn_show_pygwalker = st.checkbox("Explore Uploaded Data")
  select_target = st.selectbox('choose target', df_input.columns)

placeholder = st.empty()
if btn_show_pygwalker:
    with placeholder:
      components.html(db_pyg, scrolling=True, height = 1000)


# Add the target label and pop target-column 
df_input["target"] = df_input[select_target]
df_input.drop(select_target, axis = 1, inplace = True)
first_column = df_input.pop('target')
df_input.insert(0, 'target', first_column)


# Split data into train and test
X = df_input.drop("target", axis=1).copy()
y = df_input["target"].copy() 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=25)

# encode all remaining character variables
vars_categorical = X.select_dtypes(include="O").columns.to_list()
vars_remainder = X_train.select_dtypes(exclude="O").columns.to_list()
ct = ColumnTransformer([("encoder", OrdinalEncoder(), vars_categorical)],remainder="passthrough",)
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)
X_train = pd.DataFrame(X_train, columns=vars_categorical+vars_remainder)
X_test = pd.DataFrame(X_test, columns=vars_categorical+vars_remainder)

# show processed data
st.header("Processed Data")
col_l, col_r = st.columns([1, 9])
with col_l:
  st.write(y_train.head(3))
with col_r:
  st.write(X_train.head(3))




def fit_eval_model(model):
  mod = model
  mod.fit(X_train, y_train)
  importance = mod.feature_importances_
  importance = pd.Series(importance, index=X_test.columns).sort_values(ascending = False)
  importance = pd.DataFrame(importance)
  importance["feature"] = importance.index
  importance["model"] = mod
  importance.columns = ["feature_importance", "feature", "model"]

  preds = mod.predict(X_test)
  eval = classification_report(y_test, preds)
  st.write(importance)
  st.write(eval)
  st.write(pd.DataFrame(preds).head())
  return importance, eval, preds

fm_dectree = fit_eval_model(DecisionTreeClassifier())
fm_gradboost = fit_eval_model(GradientBoostingClassifier())
fm_rforest = fit_eval_model(RandomForestClassifier())


# Making predictions with each model
# tree_preds = tree.predict(X_test)
# randomforest_preds = randomforest.predict(X_test)
# gradientboosting_preds = gradientboosting.predict(X_test)

# Store model predictions in a dictionary
# this makes it's easier to iterate through each model
# and print the results. 
#model_preds = {
#    "Decision Tree": tree_preds,
#    "Random Forest": randomforest_preds,
#    "Gradient Booksting": gradientboosting_preds
#}

#for model, preds in model_preds.items():
#    st.write(f"{model} Results:\n{classification_report(y_test, preds)}")



          
