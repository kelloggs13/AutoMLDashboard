# future topics
# - onehotencoding
# - readme (intro?) page
# -- how to use, prerequisites, limitations[comma separated, binary classification], outlook
# - blog article (utility, learnigns, comparision with kaggle-daatsets and -scores, test-datasets)
# - sidebar. gude user by numbered steps
# - targets in upload file shoudl be 1(pos case) or 0 fpr neg case
# - confusion matrix 
#   -> add explanation of 0, 1 --> mabye to at beginning anywayxss? 
#   -> readability!!!
# - foramt change in eval-kpi as percent
# dump (and test-read) model as pickle: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://discuss.streamlit.io/t/download-pickle-file-of-trained-model-using-st-download-button/27395
# limit targetvariables in selctor to "objects"  // things with 2 distinct values and no missing values

import os
import streamlit as st
from streamlit_toggle import st_toggle_switch
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import pygwalker as pyg
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from plotnine import *
import altair as alt
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
import pickle


import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


st.set_page_config(layout = "wide")

def read_data(file):
  input_filename, input_file_extension = os.path.splitext(file.name)
  if input_file_extension == ".csv":
    df = pd.read_csv(input_data)
  elif input_file_extension == ".xlsx":
    df = pd.read_excel(input_data)
  else:
    st.warning("Uploaded file must be either .csv or .xlsx")
  return df

st.write("1. Upload Data File (CSV or XLSX)")
input_data = st.file_uploader(' ')

if input_data is not None:
  df_input = read_data(input_data)
  
  st.write("2. Choose Target for Classification Model")
  column_select_target = df_input.columns.tolist()
  column_select_target = [None] + column_select_target
  select_target = st.selectbox(' ', column_select_target)
  
  if select_target is not None: # ---- PRE-PROCESSING ----
    
    # Add the target label and pop target-column 
    df_input["target"] = df_input[select_target]
    df_input.drop(select_target, axis = 1, inplace = True)
    first_column = df_input.pop('target')
    df_input.insert(0, 'target', first_column)
    
    X = df_input.drop("target", axis=1).copy()
    y = df_input["target"].copy() 
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    
    # encode all remaining character variables
    vars_categorical = X.select_dtypes(include="O").columns.to_list()
    vars_remainder = X.select_dtypes(exclude="O").columns.to_list()
    ct = ColumnTransformer([("encoder", OrdinalEncoder(), vars_categorical)],remainder="passthrough",)
    ct.fit(X)
    X = ct.transform(X)
    X = pd.DataFrame(X, columns=vars_categorical+vars_remainder)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=25)
    
    # show uploaded data
    st.header("Uploaded Data")
    st.dataframe(df_input, hide_index = True)
    # show pre-processed data
    st.header("Pre-Processed Features")
    st.dataframe(X, hide_index = True)
    
    st.write("3. Choose Fitting or Scoring")
    select_mode = st.selectbox(' ', [None, "Fitting", "Scoring"])

    if select_mode == "Fitting":
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
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        return mod_str, df_eval, importance, conf_matrix, y_test_pred, mod
       
      col_l, col_m, col_r = st.columns([1, 1, 1])
      
      with col_l:
        fm_dectree = fit_eval_model(DecisionTreeClassifier())
        st.write(fm_dectree[0])
        st.dataframe(fm_dectree[1], hide_index = True)
        st.write(fm_dectree[3])
        st.dataframe(fm_dectree[2], hide_index = True)
        st.download_button("Download Model", data=pickle.dumps(fm_dectree[5]),file_name=f"{fm_dectree[0]}.pkl")
      with col_m:
        fm_gradboost = fit_eval_model(GradientBoostingClassifier())
        st.write(fm_gradboost[0])
        st.dataframe(fm_gradboost[1], hide_index = True)
        st.write(fm_gradboost[3])
        st.dataframe(fm_gradboost[2], hide_index = True)
        st.download_button("Download Model", data=pickle.dumps(fm_gradboost[5]),file_name=f"{fm_gradboost[0]}.pkl")
    
      with col_r:
        fm_rforest = fit_eval_model(RandomForestClassifier())
        st.write(fm_rforest[0])
        st.dataframe(fm_rforest[1], hide_index = True)
        st.write(fm_rforest[3])
        st.dataframe(fm_rforest[2], hide_index = True)
        st.download_button("Download Model", data=pickle.dumps(fm_rforest[5]),file_name=f"{fm_rforest[0]}.pkl")

    if select_mode == "Scoring":
      # Streamlit app title
      st.title("Model Scoring and Confusion Matrix")
      
      # Upload the pickled model
      model_file = st.file_uploader("Upload Pickled Model", type=["pkl"])
      
      # Upload the input data (Excel or CSV)
      input_data_file = st.file_uploader("Upload Input Data (Excel or CSV)", type=["csv", "xlsx"])
      
      if model_file and input_data_file:
          try:
              # Load the pickled model
              with model_file as f:
                  model = pickle.load(f)
      
              # Load the input data
              input_data = pd.read_csv(input_data_file) if input_data_file.name.endswith(".csv") else pd.read_excel(input_data_file)
      
              # Display uploaded input data
              st.write("Uploaded Input Data:")
              st.write(input_data)
      
              # Select target variable
              target_variable = st.selectbox("Select Target Variable", input_data.columns)
      
              # Split data into features and target
              X = input_data.drop(target_variable, axis=1)
              y = input_data[target_variable]
      
              # Train-test split for scoring
              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
              # Model scoring
              y_pred = model.predict(X_test)
      
              # Calculate confusion matrix
              cm = confusion_matrix(y_test, y_pred)
      
              # Display confusion matrix
              st.write("Confusion Matrix:")
              st.write(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))
      
          except Exception as e:
              st.error("An error occurred: " + str(e))
  
