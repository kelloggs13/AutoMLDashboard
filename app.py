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

st.set_page_config(layout = "wide")

tab_fit, tab_score = st.tabs(["Find Model", "Score Data"])

with tab_fit:
  
  param_col1, param_col2, param_col3 = st.columns([1, 1, 1])
  
  with param_col1:
    st.write("1. Upload Data File (CSV or XLSX)")
    input_data = st.file_uploader(' ')
  
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
      
    # DataViz with pygwalker
    db_pyg = pyg.walk(df_input, return_html=True)
    
    st.write("---")
#    components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)
      
    # show un-processed data
    st.header("Uploaded Data")
    btn_show_pygwalker = st.checkbox("Explore Uploaded Data")
    st.dataframe(df_input, hide_index = True)
    
    with param_col2:
      st.write("2. Choose Target for Classification Model")
      select_target = st.selectbox(' ', df_input.columns)
      target_unique = np.unique(df_input[select_target])
    with param_col3:
      st.write("3. Fit Models")
      btn_fit_models = st_toggle_switch(label="Go!",
                                  default_value=False,
                                  label_after=True,
                                  inactive_color="#D3D3D3",  # optional
                                  active_color="#11567f",  # optional
                                  track_color="#29B5E8",  # optional
                              )
     
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
      lb = preprocessing.LabelBinarizer()
      y = lb.fit_transform(y)
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

     
   
with tab_score:
  st.write("sd")
