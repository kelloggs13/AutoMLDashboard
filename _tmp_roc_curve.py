  
  st.write("ROC Curve Test Data")
  # Get predicted probabilities for the positive class
  y_prob = mod.predict_proba(X_test)[:, 1] # check: which is the positive class?
  fpr, tpr, thresholds = roc_curve(y_test, y_prob)
  roc_auc = auc(fpr, tpr)
  fig = plt.figure(figsize=(12, 12))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  st.pyplot(fig) # ValueError: y_true takes value in {'Negative', 'Positive'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.
