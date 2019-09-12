#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

#supress python expiration warnings
warnings.filterwarnings("ignore")


hr = pd.read_csv('HR_Data.csv')
col_names = hr.columns.tolist()

# Get number of positve and negative examples
pos = hr[hr["Left"] == 1].shape[0]
neg = hr[hr["Left"] == 0].shape[0]
print("Positive examples = {}".format(pos))
print("Negative examples = {}".format(neg))
print("Proportion of positive to negative examples = {:.2f}%".format((pos / neg) * 100))
sns.countplot(hr["Left"])
plt.xticks((0, 1), ["Didn't leave", "Left"])
plt.xlabel("Left")
plt.ylabel("Count")
plt.title("Class counts");
plt.show()


# In[3]:


print(col_names)


# In[48]:


#fig, axes = plt.subplots(2, 3, figsize=(10, 4))

pd.crosstab(hr["Age Bucket"],hr.Left).plot(kind='bar')
plt.title('Turnover Frequency for Age Bucket')
plt.xlabel('Age Bucket')
plt.ylabel('Frequency of Turnover')

pd.crosstab(hr["Decade"],hr.Left).plot(kind='bar')
plt.title('Turnover Frequency for Decade')
plt.xlabel('Decade')
plt.ylabel('Frequency of Turnover')

pd.crosstab(hr["Gender"],hr.Left).plot(kind='bar')
plt.title('Turnover Frequency for Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency of Turnover')

pd.crosstab(hr["Marital Status"],hr.Left).plot(kind='bar')
plt.title('Turnover Frequency for Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency of Turnover')

pd.crosstab(hr["Managed By"],hr.Left).plot(kind='bar')
plt.title('Turnover Frequency for Managed By')
plt.xlabel('Managed By')
plt.ylabel('Frequency of Turnover')

pd.crosstab(hr["Experience"],hr.Left).plot(kind='bar')
plt.title('Turnover Frequency for Experience')
plt.xlabel('Experience')
plt.ylabel('Frequency of Turnover')
plt.show()


# In[4]:


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
# 
# Split train and test set: 80/20
X = hr.loc[:, hr.columns != "Left"].values
y = hr.loc[:, hr.columns == "Left"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)
# Upsample minority class
X_train_u, y_train_u = resample(X_train[y_train == 1],
                                y_train[y_train == 1],
                                replace=True,
                                n_samples=X_train[y_train == 0].shape[0],
                                random_state=1)
X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))
# Downsample majority class
X_train_d, y_train_d = resample(X_train[y_train == 0],
                                y_train[y_train == 0],
                                replace=True,
                                n_samples=X_train[y_train == 1].shape[0],
                                random_state=1)
X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))


print("\n")
print("Original shape:", X_train.shape, y_train.shape)
print("Upsampled shape:", X_train_u.shape, y_train_u.shape)
print("Downsampled shape:", X_train_d.shape, y_train_d.shape)
print("\n")


# In[5]:


# Load common libraries
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[6]:


############ Identify the best sample method  ###############################

from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

methods_data = {"Original": (X_train, y_train),
                "Upsampled": (X_train_u, y_train_u),
                "Downsampled": (X_train_d, y_train_d)}
for method in methods_data.keys():
    
    
    pip_logmod = make_pipeline(StandardScaler(),
                           LogisticRegression(solver = "liblinear",class_weight="balanced"))
    hyperparam_range = np.arange(0.5, 20.1, 0.5)
    hyperparam_grid = {"logisticregression__penalty": ["l1", "l2"],
                   "logisticregression__C":  hyperparam_range,
                   "logisticregression__fit_intercept": [True, False]
                  }
    
    gs_logmodel = GridSearchCV(pip_logmod,
                           hyperparam_grid,
                           scoring="accuracy",
                           cv=10,
                           n_jobs=-1)
    gs_logmodel.fit(methods_data[method][0], methods_data[method][1])
    
    print("\n")
    print(f"\033[1m\033[0mThe best      hyperparameters:\n{'-' * 25}")
    for hyperparam in gs_logmodel.best_params_.keys():
        print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_logmodel.best_params_[hyperparam])
    print(f"\033[1m\033[94mBest 10-folds CV f1-score: {gs_logmodel.best_score_ * 100:.2f}%.")

print("\n")


# In[17]:


bench_mark_results=[]
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
pip_logmod = make_pipeline(StandardScaler(),
                           LogisticRegression(solver = "liblinear",class_weight="balanced"))
#hyperparam_range = np.arange(0.5, 20.1, 0.5)
hyperparam_range = [.01, .1, .5 ,1, 10, 100, 1000]
hyperparam_grid = {"logisticregression__penalty": ["l1", "l2"],
               "logisticregression__C":  hyperparam_range,
               "logisticregression__fit_intercept": [True, False]
              }

gs_logmodel = GridSearchCV(pip_logmod,
                       hyperparam_grid,
                       scoring="accuracy",
                       cv=10,
                       n_jobs=-1)

startTime = time.time()

gs_logmodel.fit(X_train_d, y_train_d)

endTime = time.time()
elapsedTime=endTime-startTime
    
print("\n")
print(f"\033[1m\033[0mThe best      hyperparameters:\n{'-' * 25}")
for hyperparam in gs_logmodel.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_logmodel.best_params_[hyperparam])
print(f"\033[1m\033[94mBest 10-folds CV f1-score: {gs_logmodel.best_score_ * 100:.2f}%.")
print("\n")

"precision and recall"
from sklearn.metrics import classification_report
print(classification_report(y_test, gs_logmodel.predict(X_test)))

y_pred = gs_logmodel.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('logistic_regression')

f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n")

bench_mark_results.append({'Classifier':'Logistic Regression',
                                 'Precision[Test](%)':str(round(precision,2)*100),
                                 'Accuracy[Test](%)':str((round(accuracy,2)*100)),   
                                 'F1-Score[Test](%)':str(round(f1,2)*100),
                                 'Recall[Test](%)':str(round(recall,2)*100),
                                 'Elapsed_Time(Sec.)':str(round(elapsedTime,2))})


# In[8]:


########### Build random forest classifier #################################
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

pip_rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=500,
                                              class_weight="balanced",
                                              random_state=123))

hyperparam_grid = {
    "randomforestclassifier__n_estimators": [10, 50, 100, 500],
    "randomforestclassifier__max_features": ["sqrt", "log2", 0.4, 0.5],
    "randomforestclassifier__min_samples_leaf": [1, 3, 5],
    "randomforestclassifier__criterion": ["gini", "entropy"]}

gs_rf = GridSearchCV(pip_rf,
                     hyperparam_grid,
                     scoring="f1",
                     cv=10,
                     n_jobs=-1)

startTime = time.time()
gs_rf.fit(X_train_d, y_train_d)
endTime = time.time()
elapsedTime=endTime-startTime

print("\n")
print("\n")
print(f"\033[1m\033[0mThe best hyperparameters:")
for hyperparam in gs_rf.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_rf.best_params_[hyperparam])
    
print(f"\033[1m\033[94mBest 10-folds CV f1-score: {gs_rf.best_score_ * 100:.2f}%.")

"Precision and Recall"
print("\n")

from sklearn.metrics import classification_report
print(classification_report(y_test, gs_rf.predict(X_test)))

y_pred = gs_rf.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')

f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n")

bench_mark_results.append({'Classifier':'Random Forrest',
                                 'Precision[Test](%)':str(round(precision,2)*100),
                                 'Accuracy[Test](%)':str((round(accuracy,2)*100)),   
                                 'F1-Score[Test](%)':str(round(f1,2)*100),
                                 'Recall[Test](%)':str(round(recall,2)*100),
                                 'Elapsed_Time(Sec.)':str(round(elapsedTime,2))})


# In[9]:


############## Build SVM classifier #########################################
from sklearn import svm
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

steps = [('scaler', StandardScaler()), ('SVM', svm.SVC(kernel="linear",probability=True))]
pip_svm = Pipeline(steps)

hyperparam_grid = {'SVM__C':[0.001,0.1,10,100], 'SVM__gamma':[0.1,0.01]}
gs_svm = GridSearchCV(pip_svm,
                     param_grid=hyperparam_grid,
                     #scoring="f1",
                     cv=5)


startTime = time.time()
gs_svm.fit(X_train_d, y_train_d)
endTime = time.time()
elapsedTime=endTime-startTime

print("\n")
print(f"\033[1m\033[0mThe best hyperparameters:")
for hyperparam in gs_svm.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_svm.best_params_[hyperparam])

print("\n")
print(" ")
print(f"\033[1m\033[94mThe 10-folds CV f1-score is: {np.mean(gs_svm.score(X_train_u, y_train_u)) * 100:.2f}%")

"precision and recall"
print("\n")
from sklearn.metrics import classification_report
print(classification_report(y_test, gs_svm.predict(X_test)))

y_pred = gs_svm.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM')
plt.savefig('svm')

f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n")

bench_mark_results.append({'Classifier':'SVM',
                                 'Precision[Test](%)':str(round(precision,2)*100),
                                 'Accuracy[Test](%)':str((round(accuracy,2)*100)),   
                                 'F1-Score[Test](%)':str(round(f1,2)*100),
                                 'Recall[Test](%)':str(round(recall,2)*100),
                                 'Elapsed_Time(Sec.)':str(round(elapsedTime,2))})


# In[10]:


############# Build Gradient Boosting classifier ########################
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

pip_gb = make_pipeline(StandardScaler(),
                       GradientBoostingClassifier(loss="deviance",
                                                  random_state=123))
hyperparam_grid = {"gradientboostingclassifier__max_features": ["log2", 0.5],
                   "gradientboostingclassifier__n_estimators": [100, 300, 500],
                   "gradientboostingclassifier__learning_rate": [0.001, 0.01, 0.1],
                   "gradientboostingclassifier__max_depth": [1, 2, 3]}
gs_gb = GridSearchCV(pip_gb,
                      param_grid=hyperparam_grid,
                      scoring="f1",
                      cv=10,
                      n_jobs=-1)

startTime = time.time()
gs_gb.fit(X_train_u, y_train_u)
endTime = time.time()
elapsedTime=endTime-startTime

print("\n")
print(f"\033[1m\033[0mThe best hyperparameters:\n{'-' * 25}")
for hyperparam in gs_gb.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_gb.best_params_[hyperparam])
print(f"\033[1m\033[94mBest 10-folds CV f1-score: {gs_gb.best_score_ * 100:.2f}%.")

print("\n")
y_pred = gs_gb.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Gradiaent Boosting')
plt.savefig('gb')

f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n")

bench_mark_results.append({'Classifier':'Gradient Boosting',
                                 'Precision[Test](%)':str(round(precision,2)*100),
                                 'Accuracy[Test](%)':str((round(accuracy,2)*100)),   
                                 'F1-Score[Test](%)':str(round(f1,2)*100),
                                 'Recall[Test](%)':str(round(recall,2)*100),
                                 'Elapsed_Time(Sec.)':str(round(elapsedTime,2))})


# In[11]:


pd.DataFrame(bench_mark_results)


# In[19]:


y_pred_rf = gs_rf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_rf)
auc = metrics.roc_auc_score(y_test, y_pred_rf)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr,tpr,label="RF, auc="+str(auc))
plt.legend(loc=4)

y_pred_lr = gs_logmodel.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_lr)
auc = metrics.roc_auc_score(y_test, y_pred_lr)
plt.plot(fpr,tpr,label="LR, auc="+str(auc))
plt.legend(loc=4)

y_pred_svm = gs_svm.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_svm)
auc = metrics.roc_auc_score(y_test, y_pred_svm)
plt.plot(fpr,tpr,label="SVM, auc="+str(auc))
plt.legend(loc=4)

y_pred_gb = gs_gb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_gb)
auc = metrics.roc_auc_score(y_test, y_pred_gb)
plt.plot(fpr,tpr,label="GB, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




