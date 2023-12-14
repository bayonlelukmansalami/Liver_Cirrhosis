import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import r2_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
numCols = df_train.select_dtypes([np.number]).columns.drop(["id"])
catCols = df_train.select_dtypes('object').columns.drop('Status')

train = pd.get_dummies(df_train, columns=catCols, drop_first=True, dtype=int)
test = pd.get_dummies(df_test, columns=catCols, drop_first=True, dtype=int)

train['Status'] = df_train['Status'].map({"D":0, "C":1, "CL":2})

X = train.drop(columns=['id','Status'], axis =1)
y = train['Status'] 
test1 = test.drop('id',axis=1)

mm = MinMaxScaler()
X_mm = mm.fit_transform(X)
test_mm = mm.transform(test1)

sample_submission = pd.read_csv('sample_submission.csv')

xg = XGBClassifier(booster='gbtree', max_depth=3, learning_rate=0.0916440028226021, n_estimators=575, 
                   min_child_weight=1, subsample=0.5593359384273137, colsample_bylevel= 0.6551948667255165, 
                   colsample_bytree=0.3604143952025156, colsample_bynode= 0.9868067661365549, 
                   reg_alpha=0.6985074556560505,reg_lambda= 0.04977711432256354, eval_metric='mlogloss')

xg.fit(X_mm, y)

# Streamlit app
st.title('Liver Disease Prediction App')

# Display the DataFrame
st.write('## Dataset')
st.write(df)

# Sidebar inputs
st.sidebar.title('Prediction Inputs')


n_days = st.sidebar.slider('N_Days', min_value=41, max_value=4795, value=182)
age = st.sidebar.slider('Age', min_value=9598, max_value=28650, value=10000)
Bilirubin = st.sidebar.slider('Bilirubin', min_value=0.3, max_value=28, value=20)
Cholesterol = st.sidebar.slider('Cholesterol', min_value=9598, max_value=28650, value=10000)
Albumin = st.sidebar.slider('Albumin', min_value=9598, max_value=28650, value=10000)
Copper = st.sidebar.slider('Copper', min_value=9598, max_value=28650, value=10000)
Alk_Phos = st.sidebar.slider('Alk_Phos', min_value=9598, max_value=28650, value=10000)
SGOT = st.sidebar.slider('SGOT', min_value=9598, max_value=28650, value=10000)
Tryglicerides = st.sidebar.slider('Tryglicerides', min_value=9598, max_value=28650, value=10000)
Platelets = st.sidebar.slider('Platelets', min_value=9598, max_value=28650, value=10000)
Prothrombin = st.sidebar.slider('Prothrombin', min_value=9598, max_value=28650, value=10000)
Stage = st.sidebar.slider('Stage', min_value=9598, max_value=28650, value=10000)
Drug_Placebo = st.sidebar.slider('Drug_Placebo', min_value=9598, max_value=28650, value=10000)
Sex_M = st.sidebar.slider('Sex_M', min_value=0, max_value=1, value=1)
Ascites_Y	 = st.sidebar.slider('Ascites_Y	', min_value=9598, max_value=28650, value=10000)
Hepatomegaly_Y = st.sidebar.slider('Hepatomegaly_Y', min_value=9598, max_value=28650, value=10000)
Spiders_Y = st.sidebar.slider('Spiders_Y', min_value=9598, max_value=28650, value=10000)
Edema_S = st.sidebar.slider('Edema_S', min_value=9598, max_value=28650, value=10000)
Edema_Y = st.sidebar.slider('Edema_Y', min_value=9598, max_value=28650, value=10000)





# Create a DataFrame for prediction
user_input = pd.DataFrame({
    'N_Days': [n_days],
    'Age': [Age],
    'Bilirubin': [Bilirubin],
    'Cholesterol': [Cholesterol],
    'Albumin': [Albumin],
    'Copper': [Copper],
    'Alk_Phos': [Alk_Phos],
    'SGOT': [SGOT],
    'Tryglicerides': [Tryglicerides	],
    'Platelets': [Platelets],
    'Prothrombin': [Prothrombin],
    'Stage': [Stage],
    'Drug_Placebo': [Drug_Placebo],
    'Sex_M': [Sex_M],
    'Ascites_Y': [Ascites_Y],
    'Hepatomegaly_Y': [Hepatomegaly_Y],
    'Spiders_Y': [Spiders_Y],
    'Edema_S': [Edema_S],
    'Edema_Y': [Edema_Y]
    
})
ss = StandardScaler()
user_input = ss.fit_transform(user_input)
# Make prediction
prediction = model.predict(user_input)[0]
if prediction == 0:
    return "Status_D patient was deceased at N_Days"
elif prediction == 1:
    return "Status_C patient was alive at N_Days"
else:
    return "Status_CL patient was alive at N_Days due to liver a transplant"
#prediction_label = le.inverse_transform([prediction])[0]

# Display prediction
st.write('## Prediction')
st.write(f'The predicted status is: {prediction}')