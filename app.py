
import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import pygwalker as pyg
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

with st.sidebar:
  select_target = st.selectbox('choose target', df_input.columns)

# Add the target label
df_input["target"] = df_input[select_target]
df_input.drop(select_target, axis = 1, inplace = True)
first_column = df_input.pop('target')
df_input.insert(0, 'target', first_column)
st.write(df_input.head(3))

# Split data into features and label 
X = df_input.drop("target", axis=1).copy()
y = df_input["target"].copy() 

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=25)


# DataViz with pygwalker
db_pyg = pyg.walk(df_input, return_html=True)

btn_show_pygwalker = st.checkbox("Explore Data")

placeholder = st.empty()
if btn_show_pygwalker:
    with placeholder:
      components.html(db_pyg, scrolling=True, height = 1000)


col_l, col_r = st.columns([1, 2])

def fit_eval_model(model):
  mod = model
  mod.fit(X_train, y_train)
  importance = mod.feature_importances_
  for i,v in enumerate(importance):
    st.write('Feature: %0d, Score: %.5f' % (i,v))
    
st.write("dectree")
fit_eval_model(DecisionTreeClassifier())
st.write("randFor")
fit_eval_model(RandomForestClassifier())
st.write("other stuff")

# Instnatiating the models 
svm = SVC()
tree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
gradientboosting = GradientBoostingClassifier()

# Training the models 
svm.fit(X_train, y_train)
tree.fit(X_train, y_train)
randomforest.fit(X_train, y_train)
gradientboosting.fit(X_train, y_train)

# get feature importances
# get importance
importance = gradientboosting.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 st.write('Feature: %0d, Score: %.5f' % (i,v))

# Making predictions with each model
svm_preds = svm.predict(X_test)
tree_preds = tree.predict(X_test)
randomforest_preds = randomforest.predict(X_test)
gradientboosting_preds = gradientboosting.predict(X_test)

# Store model predictions in a dictionary
# this makes it's easier to iterate through each model
# and print the results. 
model_preds = {
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds,
    "Random Forest": randomforest_preds,
    "Gradient Booksting": gradientboosting_preds
}

for model, preds in model_preds.items():
    st.write(f"{model} Results:\n{classification_report(y_test, preds)}")
