
def read_data(file):
  input_filename, input_file_extension = os.path.splitext(file.name)
  if input_file_extension == ".csv":
    df = pd.read_csv(input_data)
  elif input_file_extension == ".xlsx":
    df = pd.read_excel(input_data)
  else:
    st.warning("Uploaded file must be either .csv or .xlsx")
  return df

def preprocess_data(df):
  # split X, y   
  X = df.drop("target", axis=1).copy()
  y = df["target"].copy() 
  
  # encode  character model features
  vars_categorical = X.select_dtypes(include="O").columns.to_list()
  vars_remainder = X.select_dtypes(exclude="O").columns.to_list()
  ct = ColumnTransformer([("encoder", OrdinalEncoder(), vars_categorical)],remainder="passthrough",)
  ct.fit(X)
  X = ct.transform(X)
  X = pd.DataFrame(X, columns=vars_categorical+vars_remainder)
  
  return X, y

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
