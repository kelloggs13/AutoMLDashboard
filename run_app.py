# add readme

# - blog article (utility, learnigns, comparision with kaggle-daatsets and -scores, test-datasets)
# - sidebar. gude user by numbered steps

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


exec(open('functions.py').read())
exec(open('app.py').read())
