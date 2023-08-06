# future topics
# - onehotencoding
# - readme (intro?) page
# -- how to use, prerequisites, limitations[comma separated, binary classification], outlook
# - blog article (utility, learnigns, comparision with kaggle-daatsets and -scores, test-datasets)
# - sidebar. gude user by numbered steps
# - targets in upload file shoudl be 1(pos case) or 0 fpr neg case
# - confusion matrix
# - foramt change in eval-kpi as percent

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
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from plotnine import *
import altair as alt
from sklearn import metrics
import numpy as np
from sklearn import preprocessing

st.set_page_config(layout = "wide")

with st.sidebar:
  input_data = st.file_uploader('upload data')

def read_data():
  if input_file_extension == ".csv":
    df = pd.read_csv(input_data)
  elif input_file_extension == ".xlsx":
    df = pd.read_excel(input_data)
  else:
    st.warning("Uploaded file must be either .csv or .xlsx")
  return df

if input_data is not None:
  input_filename, input_file_extension = os.path.splitext(input_data.name)
  df_input = read_data()
    
  # show un-processed data
  st.header("Uploaded Data")
  st.dataframe(df_input, hide_index = True)
  
  # DataViz with pygwalker
  db_pyg = pyg.walk(df_input, return_html=True)
  
  with st.sidebar:
    btn_show_pygwalker = st.checkbox("Explore Uploaded Data")
    select_target = st.selectbox('Choose Target', df_input.columns)
    target_unique = np.unique(df_input[select_target])
    #select_pos_case = st.selectbox('Select positive Outcome', target_unique)
   # select_neg_case = np.array2string(target_unique[target_unique != select_pos_case])
   # st.write(select_pos_case)
    #st.write(select_neg_case)

    btn_fit_models = st.checkbox("Start it up")
  
  placeholder = st.empty()
  if btn_show_pygwalker:
      with placeholder:
        components.html(db_pyg, scrolling=True, height = 1000)
  
  if btn_fit_models:
    # Add the target label and pop target-column 
    df_input["target"] = df_input[select_target]
    df_input.drop(select_target, axis = 1, inplace = True)
    first_column = df_input.pop('target')
    df_input.insert(0, 'target', first_column)
    
    # Split data into train and test
    X = df_input.drop("target", axis=1).copy()
    y = df_input["target"].copy() 
   # st.write(y)
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
  #  st.write(y)

#    y = y.replace({select_pos_case: 1, select_neg_case: 0}) # sklearn.metric oftetimes requires numeric binary target
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
    if False:
      st.header("Data used for Model Fitting")
      col_l, col_r = st.columns([1, 9])
      with col_l:
        st.dataframe(y_train.head(3))
      with col_r:
        st.dataframe(X_train.head(3))
  
    st.header("Fitted Models")
    
    def fit_eval_model(model):
      mod = model
      mod_str = str(mod)
      mod.fit(X_train, y_train)
      importance = mod.feature_importances_
      importance = pd.Series(importance, index=X_train.columns).sort_values(ascending = False)
      importance = pd.DataFrame(importance)
      importance["feature"] = importance.index
      importance.columns = ["feature_importance", "feature"]
      y_train_pred = mod.predict(X_train)
      y_test_pred = mod.predict(X_test)
      
      acc_train = metrics.accuracy_score(y_train, y_train_pred)
      f1_train =  metrics.f1_score(y_train, y_train_pred, average = "binary")
      auc_train = metrics.roc_auc_score(y_train, y_train_pred, average = "macro")

      acc_test = metrics.accuracy_score(y_test, y_test_pred)
      f1_test =  metrics.f1_score(y_test, y_test_pred, average = "binary")
      auc_test = metrics.roc_auc_score(y_test, y_test_pred, average = "macro")
      df_eval = pd.DataFrame({
        ' ':['Train','Test','Change'],
        'Accuracy':[acc_train, acc_test, (acc_test/acc_train)-1],
        'F1':[f1_train, f1_test, (f1_test/f1_train)-1],
        'AUC':[auc_train, auc_test, (auc_test/auc_train)-1]
      })
      return mod_str, df_eval, importance, y_test_pred
     
    col_l, col_m, col_r = st.columns([1, 1, 1])
    
    with col_l:
      fm_dectree = fit_eval_model(DecisionTreeClassifier())
      st.write(fm_dectree[0])
      st.dataframe(fm_dectree[1], hide_index = True)
      st.dataframe(fm_dectree[2], hide_index = True)
    with col_m:
      fm_gradboost = fit_eval_model(GradientBoostingClassifier())
      st.write(fm_gradboost[0])
      st.dataframe(fm_gradboost[1], hide_index = True)
      st.dataframe(fm_gradboost[2], hide_index = True)
    with col_r:
      fm_rforest = fit_eval_model(RandomForestClassifier())
      st.write(fm_rforest[0])
      st.dataframe(fm_rforest[1], hide_index = True)
      st.dataframe(fm_rforest[2], hide_index = True)
      
    with st.sidebar:
        st.button('sd')
