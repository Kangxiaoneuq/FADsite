import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from catboost import CatBoostClassifier

with open('FADsite_seq.pkl', 'rb') as f:
    model = pickle.load(f)
    
ori = pd.read_csv(r'test6.csv')
test_X = ori.iloc[:,:-1]
test_label = ori.iloc[:,-1]
resample_pred = model.predict(np.array(test_X))
y_score = model.predict_proba(np.array(test_X))[:,1]
fpr,tpr,threshold = metrics.roc_curve(test_label, y_score)
roc_auc = metrics.auc(fpr,tpr)
print("The results for FADsite_seq on Test6")
print('Pre:',metrics.precision_score(test_label,resample_pred))
print('f1:',metrics.f1_score(test_label,resample_pred))
print('Acc:',metrics.accuracy_score(test_label,resample_pred))
print('AUC:',roc_auc)
print('MCC:',matthews_corrcoef(test_label,resample_pred))    
print('')

ori = pd.read_csv(r'test4.csv')
test_X = ori.iloc[:,:-1]
test_label = ori.iloc[:,-1]
resample_pred = model.predict(np.array(test_X))
y_score = model.predict_proba(np.array(test_X))[:,1]
fpr,tpr,threshold = metrics.roc_curve(test_label, y_score)
roc_auc = metrics.auc(fpr,tpr)
print("The results for FADsite_seq on Test4")
print('Pre:',metrics.precision_score(test_label,resample_pred))
print('f1:',metrics.f1_score(test_label,resample_pred))
print('Acc:',metrics.accuracy_score(test_label,resample_pred))
print('AUC:',roc_auc)
print('MCC:',matthews_corrcoef(test_label,resample_pred))