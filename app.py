
# final toolset:
# - preprocessing funcitons
# - dashboard without prepcorcesing that can classif/regregssion and #
# - supports multicalss classifaction

# todo:
# - strings one-hot encoden => heisst auch dass strings bei welchen label encoding besser wäre( weil es eine
# reihenfolge gibt) bereits or dem dashboard als integer definiert sein müssen.

# clean up imports
# clean up / remove preproc
# add readme

# preprocessign muss
# - bei test und train gleich erfolgen
# - aich beim predicten erfolgen, gemäss dem selben vrogehen wie biem trainieren des modells
# --> beides im dashboard (??)


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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier

exec(open('functions.py').read())

st.set_page_config(layout = "wide")

st.sidebar.subheader("Inputs")

input_data = st.sidebar.file_uploader('Upload Data File (CSV or XLSX)')

current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.write(f'<span style="color: grey;">{current_datetime}</span>', unsafe_allow_html=True)


if input_data is not None:
  df_input = read_data(input_data)

  column_select_target = df_input.columns.tolist()
  column_select_target = [" "] + column_select_target
  select_target = st.sidebar.selectbox("Choose Target for Classification", column_select_target)

  if select_target != " ":
    df_input["target"] = df_input[select_target]
    df_input.drop(select_target, axis = 1, inplace = True)
    first_column = df_input.pop('target')
    df_input.insert(0, 'target', first_column)
    
    st.header("Inputs", divider = "red")

    col_data_1, col_data_2 = st.columns([1, 5])
    
    with col_data_1:
      st.subheader("Counts of Target")
      st.write(df_input.target.value_counts())
  
    with col_data_2:
      st.subheader("Uploaded Data")
      st.dataframe(df_input, hide_index = False)

    # pre-process
    X, y = preprocess_data_classif(df_input)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=25)

    # evaluate and explain models    
    col_fm_1, col_fm_2, col_fm_3 = st.columns([1, 1, 1])
    with col_fm_1:
      fit_and_describe(RandomForestClassifier())
    with col_fm_2:
      fit_and_describe(GradientBoostingClassifier())
    with col_fm_3:
      fit_and_describe(AdaBoostClassifier())
  
  

      
