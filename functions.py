
def read_data(file):
  input_filename, input_file_extension = os.path.splitext(file.name)
  if input_file_extension == ".csv":
    df = pd.read_csv(input_data)
  elif input_file_extension == ".xlsx":
    df = pd.read_excel(input_data)
  else:
    st.warning("Uploaded file must be either .csv or .xlsx")
  return df

def preprocess_data_classif(df):
  # split X, y   
  X = df.drop("target", axis=1).copy()
  y = df["target"].copy() 
  # encode  character model features
  vars_categorical = X.select_dtypes(include="O").columns.to_list()
  vars_remainder = X.select_dtypes(exclude="O").columns.to_list()
  ct = ColumnTransformer([("encoder", OrdinalEncoder(), vars_categorical)], remainder="passthrough",)
  ct.fit(X)
  X = ct.transform(X)
  X = pd.DataFrame(X, columns=vars_categorical+vars_remainder)
  # encode target as binary
  # lb = preprocessing.LabelBinarizer()
  # y = lb.fit_transform(y)
  return X, y

        
def fit_and_describe(mod):
  mod_name = str(mod).replace("Classifier()", "")
  st.header(mod_name, divider = "red")
  mod.fit(X_train, y_train)
  y_train_pred = mod.predict(X_train)
  y_test_pred = mod.predict(X_test)

  accuracy_train = accuracy_score(y_train, y_train_pred)
  accuracy_test = accuracy_score(y_test, y_test_pred)
  f1_train = f1_score(y_train, y_train_pred, average='micro')
  f1_test = f1_score(y_test, y_test_pred, average='micro')

  df_metrics = pd.DataFrame({
    '':["Train", "Test"]
   ,'Accuracy':[accuracy_train, accuracy_test]
   ,'F1':[f1_train, f1_test]
  })

  st.subheader("Metrics")
  st.write(df_metrics)

  st.subheader("Confusion Matrix")
  st.write(confusion_matrix(y_test, y_test_pred))

  importances = mod.feature_importances_
  indices = np.argsort(importances)[::-1]
  feature_importance = mod.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  fig = plt.figure(figsize=(12, 12))
  plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
  plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
  st.subheader("Feature Importance")
  st.pyplot(fig)
  
  # Download button
  st.download_button(
        "Download Model",
        data = pickle.dumps(mod),
        file_name = mod_name + ".pkl",
    )
