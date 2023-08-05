
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import pygwalker as pyg
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(layout = "wide")

breastcancer = load_breast_cancer() 

# Convert data to pandas dataframe
breastcancer_df = pd.DataFrame(breastcancer.data, columns=breastcancer.feature_names)

# Add the target label
breastcancer_df["target"] = breastcancer.target
st.write(breastcancer_df)

# Take a preview
breastcancer_df.head()

db_pyg = pyg.walk(breastcancer_df, return_html=True)
components.html(db_pyg, height=1000, scrolling=True)


# Split data into features and label 
X = breastcancer_df.drop("target", axis=1).copy()
y = breastcancer_df["target"].copy() 

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=25)

# Check the splits are correct
st.write(f"Train size: {round(len(X_train) / len(X) * 100)}% \n\ Test size: {round(len(X_test) / len(X) * 100)}%")

# Instnatiating the models 
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
gradientboosting = GradientBoostingClassifier()

# Training the models 
logistic_regression.fit(X_train, y_train)
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
log_reg_preds = logistic_regression.predict(X_test)
svm_preds = svm.predict(X_test)
tree_preds = tree.predict(X_test)
randomforest_preds = randomforest.predict(X_test)
gradientboosting_preds = gradientboosting.predict(X_test)

# Store model predictions in a dictionary
# this makes it's easier to iterate through each model
# and print the results. 
model_preds = {
    "Logistic Regression": log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds,
    "Random Forest": randomforest_preds,
    "Gradient Booksting": gradientboosting_preds
}

for model, preds in model_preds.items():
    st.write(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
