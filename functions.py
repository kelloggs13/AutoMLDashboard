
@st.cache_data
def read_data(file):
  input_filename, input_file_extension = os.path.splitext(file.name)
  if input_file_extension == ".csv":
    df = pd.read_csv(input_data)
  elif input_file_extension == ".xlsx":
    df = pd.read_excel(input_data)
  else:
    st.warning("Uploaded file must be either .csv or .xlsx")
  return df


def preprocess_features(data, onehot_encode_threshold=2):
    """
    Preprocesses a DataFrame with model features by imputing missing values, encoding categorical columns,
    and standardizing numerical columns.

    Args:
        data (pd.DataFrame): Input DataFrame with mixed numerical and categorical features.
        onehot_encode_threshold (int, optional): Threshold for one-hot encoding. Default is 2.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Label encoders for categorical columns.
        dict: One-hot encoders for high cardinality columns.
        
    Example Usage:
      data = pd.DataFrame({
          'Age': [25, 30, 35, None, 40],
          'Gender': ['Male', 'Female', 'Male', 'Other', 'Male'],
          'Income': [50000, 60000, 75000, 80000, 90000],
          'Level': ["A1", "A2", "A1", "A1", "A2"]
      })
      
      preprocessed_data = preprocess_data(data
      
      print(data)
      print(preprocessed_data)
    """
    
    imputer = SimpleImputer(strategy='mean') 
    scaler = StandardScaler() 

    label_encoders = {}
    onehot_encoders = {}

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int', 'float']).columns.tolist()

    # Impute missing values with the mean for numerical columns
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

    for col in categorical_cols:
        # Check if one-hot encoding is needed based on unique values
        if data[col].nunique() > onehot_encode_threshold:
            # Create a one-hot encoder
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_features = encoder.fit_transform(data[[col]])
            column_names = [f"{col}_{val}" for val in encoder.categories_[0][1:]]
            encoded_df = pd.DataFrame(encoded_features, columns=column_names)
            data = pd.concat([data, encoded_df], axis=1)

            # Drop the original categorical column
            data.drop(col, axis=1, inplace=True)
            
        # Create a label encoder for low cardinality categorical columns    
        else: 
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # Standardize the remaining numerical columns using the scaler
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data


def fit_and_describe(mod):
  mod_name = str(mod).replace("Classifier()", "")
  st.subheader(mod_name)
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

  st.write("Metrics")
  st.write(df_metrics)

  st.write("Confusion Matrix Test Data")
  st.write(confusion_matrix(y_test, y_test_pred))

  importances = mod.feature_importances_
  indices = np.argsort(importances)[::-1]
  feature_importance = mod.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  fig = plt.figure(figsize=(12, 12))
  plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
  plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
  st.write("Feature Importance Test Data")
  st.pyplot(fig)
  
  # Download button
  st.download_button(
        "Download Model",
        data = pickle.dumps(mod),
        file_name = mod_name + ".pkl",
    )
