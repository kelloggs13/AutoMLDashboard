
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


    X = df_input.drop("target", axis=1).copy()
    y = df_input["target"].copy() 

    # pre-process
    print(X.head())
    X = preprocess_features(X)
    print(X.head())

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
  
  
