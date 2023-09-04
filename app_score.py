
st.set_page_config(layout = "wide")

# Function to load the pre-trained model from a .pkl file
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

# Main Streamlit app
st.title("Model Scoring App")

# Upload data file
uploaded_data = st.sidebar.file_uploader("Upload a CSV or Excel file with data", type=["csv", "xlsx"])

# Upload model file
model_file = st.sidebar.file_uploader("Upload a .pkl model file", type=["pkl"])

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
    
    if select_target_scoring != " ":
      df_input["select_target_scoring"] = df_input[select_target_scoring]
      df_input.drop(select_target_scoring, axis=1, inplace=True)
  
      X = df_input.drop("select_target_scoring", axis=1).copy()
      y = df_input["select_target_scoring"].copy() 
      
      
      # pre-process
      X_orig = X.copy()
      X = preprocess_features(X)
    
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
        df_pred = pd.DataFrame(predictions)
        df_pred.columns = ["prediction"]
        st.write(df_pred.prediction.value_counts())
        
        df_pred_features = pd.concat([df_pred, X_orig], axis=1)

        def download_link(df, filename):
          df.to_excel(filename, index=False)
          with open(filename, 'rb') as f:
              data = f.read()
          st.download_button(
              label="Download Predictions as Excel File",
              data=data,
              file_name=filename,
              key='excel-download'
          )
      
      # Display the DataFrame
      st.dataframe(df_pred_features)
      
      # Add the download button
      download_link(df_pred_features, 'predictions.xlsx')
