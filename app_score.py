
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc
import streamlit as st
import pandas as pd
import pickle
import tempfile

exec(open('functions.py').read())

# Function to load the pre-trained model from a .pkl file
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

# Main Streamlit app
st.title("Model Scoring App")

# Upload data file
uploaded_data = st.file_uploader("Upload a CSV or Excel file with data", type=["csv", "xlsx"])

# Upload model file
model_file = st.file_uploader("Upload a .pkl model file", type=["pkl"])

if uploaded_data is not None and model_file is not None:
    if uploaded_data.name.endswith('.csv'):
        # Load CSV data into a DataFrame
        df_input = pd.read_csv(uploaded_data)
    else:
        # Load Excel data into a DataFrame
        df_input = pd.read_excel(uploaded_data)

    column_select_target_scoring = df_input.columns.tolist()
    column_select_target_scoring = [" "] + column_select_target_scoring
    select_target_scoring = st.sidebar.selectbox("Choose Target for Scoring", column_select_target_scoring)
    
    if select_target_scoring:
        df_input["select_target_scoring"] = df_input[select_target_scoring]
        df_input.drop(select_target_scoring, axis=1, inplace=True)
    
        X = df_input.drop("select_target_scoring", axis=1).copy()
        y = df_input["select_target_scoring"].copy() 
        
        
        # pre-process
        X = preprocess_features(X)
      
        st.write(y)
        st.write(X)
      
        # Save the uploaded model to a temporary file
        temp_model_file = tempfile.NamedTemporaryFile(delete=False)
        temp_model_file.write(model_file.read())
        temp_model_file.close()
      
        # Load the model
        model = load_model(temp_model_file.name)
  
        # Check if the model is loaded
        if model is not None:
            st.write("Model successfully loaded!")
  
            # Score the data using the model
            predictions = model.predict(X)
  
            # Display the results
            st.subheader("Model Predictions:")
            st.write(predictions)
