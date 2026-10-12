"""Reproduce the shipped research classifier. Run from any directory.
Data: UCI Heart Disease (Cleveland), Janosi et al. (1989), CC BY 4.0.
https://doi.org/10.24432/C52P4X
This is an educational classifier, not a clinically validated medical device.
"""
from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, brier_score_loss
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'model/processed.cleveland.data'
features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
numeric = ['age','trestbps','chol','thalach','oldpeak']
categorical = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
categories = [[0,1],[1,2,3,4],[0,1],[0,1,2],[0,1],[1,2,3],[0,1,2,3],[3,6,7]]
data = pd.read_csv(source, names=features+['num'], na_values='?')
clean = data.dropna().copy()
assert len(data)==303 and len(clean)==297
assert clean.duplicated().sum()==0
X, y = clean[features], (clean['num']>0).astype(int)
train_idx,test_idx = train_test_split(np.arange(len(clean)),test_size=.2,random_state=42,stratify=y)
pre = ColumnTransformer([('numeric',StandardScaler(),numeric),('categorical',OneHotEncoder(categories=categories,sparse_output=False,handle_unknown='error'),categorical)])
# Algorithm and hyperparameters fixed in advance; no model selection on holdout.
pipeline = Pipeline([('pre',pre),('classifier',LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000,random_state=42))])
pipeline.fit(X.iloc[train_idx],y.iloc[train_idx])
probs = pipeline.predict_proba(X.iloc[test_idx])[:,1]
pred = (probs>=.5).astype(int)
tn,fp,fn,tp = confusion_matrix(y.iloc[test_idx],pred,labels=[0,1]).ravel()
accuracy = accuracy_score(y.iloc[test_idx],pred)
n=len(test_idx); z=1.96; denom=1+z*z/n
center=(accuracy+z*z/(2*n))/denom
half=z*np.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/denom
scaler=pre.named_transformers_['numeric']; clf=pipeline.named_steps['classifier']
means=pre.transform(X.iloc[train_idx]).mean(axis=0)
model={
 'version':'pulse-cleveland-lr-1.0.0','algorithm':'L2-regularized logistic regression','threshold':.5,
 'numeric':numeric,'categorical':categorical,'categories':categories,
 'means':scaler.mean_.tolist(),'scales':scaler.scale_.tolist(),
 'weights':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),'encodedTrainingMean':means.tolist(),
 'ranges':{k:[float(X[k].min()),float(X[k].max())] for k in numeric},
 'source':{'name':'UCI Heart Disease — Cleveland','url':'https://archive.ics.uci.edu/dataset/45/heart+disease','doi':'https://doi.org/10.24432/C52P4X','citation':'Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository.','license':'CC BY 4.0','sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
 'training':{'sourceRows':303,'completeRows':297,'excludedRows':6,'trainRows':len(train_idx),'testRows':len(test_idx),'randomSeed':42,'split':'80/20 stratified','preprocessing':'Training-only standardization of 5 numeric features; one-hot encoding of 8 categorical features. Complete cases only.','hyperparameters':{'C':1.0,'solver':'lbfgs','max_iter':2000},'sklearnVersion':sklearn.__version__,'pythonVersion':platform.python_version(),'trainPositive':int(y.iloc[train_idx].sum()),'trainFemale':int((X.iloc[train_idx].sex==0).sum()),'trainMale':int((X.iloc[train_idx].sex==1).sum())},
 'metrics':{'accuracy':float(accuracy),'accuracyCI95':[float(center-half),float(center+half)],'rocAuc':float(roc_auc_score(y.iloc[test_idx],probs)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'brier':float(brier_score_loss(y.iloc[test_idx],probs)),'majorityBaselineAccuracy':float(max((y.iloc[test_idx]==0).mean(),(y.iloc[test_idx]==1).mean())),'confusion':{'trueNegative':int(tn),'falsePositive':int(fp),'falseNegative':int(fn),'truePositive':int(tp)}},
 'target':'Dataset num > 0 (disease class) versus num = 0 (no-disease class). Coronary angiographic disease status; not all heart disease and not future risk.',
 'limitations':['Small, historical Cleveland clinical cohort; not representative of general screening populations.','Single internal random holdout only; no prospective or external clinical validation.','Scores have not been externally calibrated and are not reliable personal disease probabilities.','Sex is binary in the source data; subgroup performance and fairness have not been established.','Requires clinical test results; do not guess or substitute missing measurements.','False negatives and false positives occur. Do not use to diagnose, rule out disease, change medication, or delay care.']
}
(ROOT/'dist/model.json').write_text(json.dumps(model,indent=2)+'\n')
fixtures=[{'inputs':{k:float(X.iloc[i][k]) for k in features},'probability':float(p)} for i,p in zip(test_idx,probs)]
(ROOT/'tests/parity-fixtures.json').write_text(json.dumps(fixtures,indent=2)+'\n')
(ROOT/'model/split.json').write_text(json.dumps({'trainOriginalRowIds':clean.index[train_idx].tolist(),'testOriginalRowIds':clean.index[test_idx].tolist()},indent=2)+'\n')
print(json.dumps({'training':model['training'],'metrics':model['metrics'],'ranges':model['ranges']},indent=2))
