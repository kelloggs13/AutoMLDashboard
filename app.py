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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from plotnine import *
import altair as alt
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
import pickle
from datetime import datetime


exec(open('functions.py').read())

st.set_page_config(layout = "wide")

st.sidebar.subheader("Inputs")

form_input = st.sidebar.form("inputs")
input_data = form_input.file_uploader('Upload Data File (CSV or XLSX)')
inputs_submitted = form_input.form_submit_button("Go!")

st.sidebar.write(datetime.now())

if input_data is not None:
  
  df_input = read_data(input_data)

  column_select_target = df_input.columns.tolist()
  select_target = form_input.selectbox("Choose Target for Classification Model", column_select_target)
  
  # show uploaded data
  st.subheader("Uploaded Data")
  show_all_data_uploaded = st.checkbox("Show all data", key = "alldata_uploaded")
  if show_all_data_uploaded:
    st.dataframe(df_input, hide_index = True)
  if not show_all_data_uploaded:
    st.dataframe(df_input.head(3), hide_index = True)

  if inputs_submitted:
    df_input["target"] = df_input[select_target]
    df_input.drop(select_target, axis = 1, inplace = True)
    first_column = df_input.pop('target')
    df_input.insert(0, 'target', first_column)
    st.subheader("Counts of Target")
    st.write(df_input.target.value_counts())

    # pre-process
    X, y = preprocess_data_classif(df_input)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=25)
    
    # show pre-processed data
    st.subheader("Pre-Processed Features")
    show_all_data_preproc = st.checkbox("Show all data", key = "alldata_preproc")
    if show_all_data_preproc:
      st.dataframe(X, hide_index = True)
    if not show_all_data_preproc:
      st.dataframe(X.head(3), hide_index = True)

    st.write("todo: 'fitting y on X1, X2, etc.'")

    st.subheader("Fitted Models")
    
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    fm = fit_eval_model_classif(RandomForestClassifier())
    
    st.write(fm[0])
    st.dataframe(fm[1], hide_index = True)
    st.write(fm[3])
    st.dataframe(fm[2], hide_index = True)
    st.sidebar.download_button("Download Model", data=pickle.dumps(fm[5]),file_name=f"{fm[0]}.pkl")
    
    st.sidebar.button("Reset", on_click = st.experimental_rerun)

    if 1 == 2: # "Scoring"
      
      # Upload the pickled model
      model_file = st.sidebar.file_uploader("Upload Pickled Model", type=["pkl"])
      
      if model_file is not None:
        
        # Load the pickled model
        with model_file as f:
          model_scoring = pickle.load(f)
    
        # Model scoring
        y_scored = model_scoring.predict(X)
    
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_scored)
    
        # Display confusion matrix
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))
      
        # download predictions
        data_pred = df_input.copy()
        data_pred['Predictions'] = y_scored
        st.write(data_pred)
        st.write("todo: target and orig.target need to be comparable")
        st.write("fix download button")
        st.sidebar.download_button("Download Scored Data", data=data_pred)
  
