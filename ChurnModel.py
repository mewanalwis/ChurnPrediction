#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Connecting to existing Azure Machine Learning workspace
import os
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

ws = Workspace.get(name='ws_employee_churn', auth=None, subscription_id='9c43f403-ce9b-4e3a-8f7f-21b1688e2873', resource_group='rg_datascience')
cluster_type = os.environ.get("AML_COMPUTE_CLUSTER_TYPE", "CPU")
compute_target = ws.get_default_compute_target(cluster_type)


# In[ ]:


#Read and pre-process data
import pandas as pd
import numpy as np

hr = pd.read_csv('HR_Data.csv')
col_names = hr.columns.tolist()

col_names
cat_vars=['Age Bucket', 'Decade', 'Gender', 'Marital Status', 'Cost Center', 'Managed By', 'Designation', 'Employee Band', 'Experience']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(hr[var], prefix=var,drop_first=False)
    hr1=hr.join(cat_list)
    hr=hr1

hr.drop(hr.columns[[0,1,2,3,4,5,6,7,8,9]], axis=1, inplace=True)
hr.columns.values

hr_vars=hr.columns.values.tolist()
y=['Left']
X=[i for i in hr_vars if i not in y]


from sklearn.model_selection import train_test_split
from sklearn.utils import resample
#
# Convert dataframe into numpy objects and split them into
# train and test sets: 80/20
X = hr.loc[:, hr.columns != "Left"].values
y = hr.loc[:, hr.columns == "Left"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)

X_train_u, y_train_u = resample(X_train[y_train == 1],
                                y_train[y_train == 1],
                                replace=True,
                                n_samples=X_train[y_train == 0].shape[0],
                                random_state=1)
X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))
print("Upsampled shape:", X_train_u.shape, y_train_u.shape)


from sklearn.pipeline import make_pipeline


# In[ ]:


# Build and register Random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from azureml.core import Run

run = Run.get_context()
rf = RandomForestClassifier(n_estimators=10,max_features = "sqrt",
                            min_samples_leaf=5,criterion="entropy",class_weight="balanced")
rf.fit(X_train_d, y_train_d)

# Place the model file in the outputs folder.
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=rf, filename='outputs/emp_churn_model_rf.pkl')

# register the model Azure 
model = Model.register(model_path = "outputs/emp_churn_model_rf.pkl",
                       model_name = "emp_churn_model_rf",
                       tags = {'area': "employee churn", 'type': "random forest"},
                       description = "Random forest model to predict employee churn",
                       workspace = ws)
print(model.name, model.id, model.version, sep='\t')


# In[ ]:


#Build and register Logistic regression model
from sklearn.linear_model import LogisticRegression
from azureml.core import Run
from sklearn.externals import joblib
from azureml.core.model import Model

run = Run.get_context()

logmod = LogisticRegression(solver = "liblinear",C=0.5,penalty="l2",fit_intercept=True,class_weight="balanced")
logmod.fit(X_train_u, y_train_u)

from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logmod.predict(X_test))))

# Place the model file in the outputs folder.
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=logmod, filename='outputs/emp_churn_model.pkl')

# register the model Azure 
model = Model.register(model_path = "outputs/emp_churn_model.pkl",
                       model_name = "emp_churn_model",
                       tags = {'area': "employee churn", 'type': "regression"},
                       description = "Logistic regression model to predict employee churn",
                       workspace = ws)
print(model.name, model.id, model.version, sep='\t')


# In[ ]:


print(run.get_file_names())


# In[ ]:


from azureml.core import Workspace
from azureml.core.model import Model
import os 

model=Model(ws, 'emp_churn_model')
model.download(target_dir=os.getcwd(), exist_ok=True)

# verify the downloaded model file
file_path = os.path.join(os.getcwd(), "emp_churn_model.pkl")

os.stat(file_path)


# In[ ]:


#Code snippet to check the web service
import requests
import json
import pandas as pd
from azureml.core.model import Model
from sklearn.externals import joblib


hr = pd.read_csv('predictions.csv')
scoring_uri = 'http://690e1634-fd94-4f55-a10e-bc3a57004dd2.southeastasia.azurecontainer.io/score'
# If the service is authenticated, set the key
key = '<your key>'


# Convert to JSON string
input_data = hr.to_json(orient='records')

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)


# In[ ]:


import pandas as pd
#input_df = pd.read_json(input_df, orient='records')
input_df_encoded = input_df

columns_encoded = ['Age Bucket_20', 'Age Bucket_30', 'Age Bucket_40',
       'Age Bucket_Teen', 'Decade_60', 'Decade_70', 'Decade_80',
       'Decade_90', 'Gender_F', 'Gender_M', 'Marital Status_Married',
       'Marital Status_Unmarried', 'Cost Center_Abbott - A',
       'Cost Center_Abbott - B', 'Cost Center_Abbott - C',
       'Cost Center_CPL', 'Cost Center_Cadila Pharma',
       'Cost Center_Cassel', 'Cost Center_East', 'Cost Center_Elite',
       'Cost Center_Endo', 'Cost Center_FDC', 'Cost Center_Fourrts',
       'Cost Center_Galderma', 'Cost Center_Glenmark Cuticare',
       'Cost Center_Glenmark General', 'Cost Center_Hilton',
       'Cost Center_Indus', 'Cost Center_Innova', 'Cost Center_Lifescan',
       'Cost Center_Magna', 'Cost Center_North', 'Cost Center_North ',
       'Cost Center_Novo HI', 'Cost Center_SKF', 'Cost Center_Surelife ',
       'Cost Center_Unison', 'Cost Center_Zydus Alidac',
       'Cost Center_Zydus Allidac', 'Cost Center_Zydus CND',
       'Cost Center_Zydus CND ', 'Cost Center_Zydus Cadila',
       'Cost Center_Zydus Cardiva', 'Cost Center_Zydus Respicare',
       'Managed By_Principle Managed', 'Managed By_SHL Managed',
       'Designation_Assistant Field Manager', 'Designation_Field Manager',
       'Designation_Medical Advicer',
       'Designation_Medical Marketing Executive',
       'Designation_Medical Representative',
       'Designation_Senior Field Manager',
       'Designation_Senior Medical Marketing Executive',
       'Designation_Senior Product Specialist', 'Employee Band_5',
       'Employee Band_6', 'Employee Band_7', 'Experience_Fresher',
       'Experience_Non Pharma', 'Experience_Other', 'Experience_Pharma']
        
cat_vars=['Age Bucket', 'Decade', 'Gender', 'Marital Status', 'Cost Center', 'Managed By', 'Designation', 'Employee Band', 'Experience']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(input_df[var], prefix=var,drop_first=False)
    hr1=input_df.join(cat_list)
    input_df=hr1
for column_encoded in columns_encoded:
    if not column_encoded in input_df.columns:
        input_df_encoded[column_encoded] = 0
        
print(input_df_encoded.columns)
print(input_df.columns)


# In[ ]:


print(input_df_encoded)
df_cd = pd.merge(input_df_encoded, input_df, how='inner')
df_cd


# In[ ]:




