#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Connecting to Azure Workspace
import os
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

ws = Workspace.get(name='ws_employee_churn', auth=None, subscription_id='9c43f403-ce9b-4e3a-8f7f-21b1688e2873', resource_group='rg_datascience')
cluster_type = os.environ.get("AML_COMPUTE_CLUSTER_TYPE", "CPU")
compute_target = ws.get_default_compute_target(cluster_type)


# In[2]:


from azureml.core.model import Model

model=Model(ws, 'emp_churn_model')
model.download(target_dir=os.getcwd(), exist_ok=True)

# verify the downloaded model file
file_path = os.path.join(os.getcwd(), "emp_churn_model.pkl")
os.stat(file_path)


# In[3]:


# Writing the score file
%%writefile score.py
import json
import pickle
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('emp_churn_model')
    model = joblib.load(model_path)

def run(input_df):
    try:
        input_df = pd.read_json(input_df, orient='records')
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
        
        cat_vars=['Age Bucket', 'Decade', 'Gender', 'Marital Status', 'Cost Center', 'Managed By', 'Designation', 
                  'Employee Band', 'Experience']
        for var in cat_vars:
            cat_list='var'+'_'+var
            cat_list = pd.get_dummies(input_df[var], prefix=var,drop_first=False)
            hr1=input_df.join(cat_list)
            input_df=hr1
        for column_encoded in columns_encoded:
            if not column_encoded in input_df.columns:
                input_df_encoded[column_encoded] = 0
        
        df_cd = pd.merge(input_df_encoded, input_df, how='inner')
        df_cd.drop(df_cd.columns[[0,1,2,3,4,5,6,7,8,9]], axis=1, inplace=True)
        
        result = model.predict(df_cd)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error


# In[4]:


#Settng the enviornment dependancies
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())


# In[5]:


with open("myenv.yml","r") as f:
    print(f.read())


# In[6]:


#Defining ACI configurations
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "Churn",  "method" : "sklearn"}, 
                                               description='Predict Employee Churn with sklearn')


# In[7]:


#Deploy the model in Azure
%%time
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage

# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="myenv.yml")

service = Webservice.deploy_from_model(workspace=ws,
                                       name='emp-churn-model-ci',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)


# In[ ]:


print(service.get_logs())


# In[8]:


print(service.scoring_uri)


# In[ ]:


print(service.state)

