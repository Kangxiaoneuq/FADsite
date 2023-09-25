import copy
import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from scipy.spatial.distance import pdist

ori = pd.read_excel(r'test6.xlsx')
name = ori.iloc[:6,0]
sequence = ori.iloc[:6,1]
label = ori.iloc[:,2]
df_seq_inf = ori.iloc[:,3:]

Name = []
for i in range(len(sequence)):
    for j in range(len(sequence[i])):
        Name.append(name[i])
path = "..\\pdb\\"

df_empty = pd.DataFrame()

for n in range(len(name)):    
    x = []
    y = []
    z = []
    NO_aminoacid = []
    for line in open(path + name[n] +'.pdb'):
        list = line.split()
        if list[0] == 'ATOM':
            NO_aminoacid.append(list[5])
            x.append(float(list[6]))
            y.append(float(list[7]))
            z.append(float(list[8]))
    number_of_atom = len(x)
    number_of_aminoacid = 1
    Rev_NO_aminoacid = [1]*number_of_atom
    for i in range(2,len(x)):
        if NO_aminoacid[i]!=NO_aminoacid[i-1]:
            number_of_aminoacid = number_of_aminoacid +1
            for j in range(i,len(x)):
                Rev_NO_aminoacid[j] = number_of_aminoacid
            
    contact = np.zeros((number_of_aminoacid, number_of_aminoacid)).astype('int64')
    for i in range(len(x)):
        for j in range(len(x)):
            if abs(Rev_NO_aminoacid[i]-Rev_NO_aminoacid[j]) > 1:
                a=[x[i],y[i],z[i]]
                b=[x[j],y[j],z[j]]
                X=np.vstack([a,b]) 
                d_ij = pdist(X)
                if d_ij <= 4.5 :
                    contact[Rev_NO_aminoacid[i]-1][Rev_NO_aminoacid[j]-1] = 1
    G=nx.Graph(contact)
    degree = []
    degree.extend(nx.degree_centrality(G).values())

    son_of_site = []
    for i in range(len(contact)):
        site = []
        for j in range(len(contact)):
            if contact[i][j] == 1:
                site.append(j+1)
        son_of_site.append(site)
    
    a1 = []
    for i in range(len(son_of_site)):
        b1 = []
        for j in range(len(son_of_site[i])):
            b1.append(degree[son_of_site[i][j]-1])
        a1.append(b1)
    
    degree_of_son = []
    for i in range(len(a1)):
        degree_of_son.append(sum(a1[i]))
    info_ = []
    for j in range(len(Name)):
        if Name[j] == name[n]: 
            info_.append(df_seq_inf.iloc[j,:])
            
    info = pd.DataFrame(info_)
    info.reset_index(drop = True,inplace = True)
    
    info2 = []
    for i in range(len(son_of_site)):
        info1 = []
        for j in range(len(son_of_site[i])):
            info1.append(info.loc[son_of_site[i][j]-1])
        info2.append(info1)
    
    info51 = []
    for i in range(len(info2)):
        info31 = []
        for j in range(len(info2[i])):
            for k in range(len(info2[i][j])):
                info31.append(info2[i][j][k] * a1[i][j])
        info51.append(info31)
    
    info7 = []
    for i in range(len(info51)):
        x1 = np.array(info51[i])
        x2 = int(len(x1)/46)
        x3 = np.reshape(x1, (x2,46))
        info7.append(np.sum(x3, axis=0))

    df_e = pd.DataFrame(info7)
    df_empty = pd.concat([df_empty,df_e],ignore_index=True)

with open('FADsite.pkl', 'rb') as f:
    model = pickle.load(f)
    
test_X = np.hstack((df_seq_inf,df_empty)) 
test_label = label
resample_pred = model.predict(np.array(test_X))
y_score = model.predict_proba(np.array(test_X))[:,1]
fpr,tpr,threshold = metrics.roc_curve(test_label, y_score)
roc_auc = metrics.auc(fpr,tpr)
print("The results for FADsite on Test6")
print('Pre:',metrics.precision_score(test_label,resample_pred))
print('f1:',metrics.f1_score(test_label,resample_pred))
print('Acc:',metrics.accuracy_score(test_label,resample_pred))
print('AUC:',roc_auc)
print('MCC:',matthews_corrcoef(test_label,resample_pred))
      