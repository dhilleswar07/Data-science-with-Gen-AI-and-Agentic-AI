import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\DATASCIENCE WITH GEN AI & AGENTIC AI\logit classification.csv")

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler  
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# Training the Logistic_regressor model on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion_matrix:\n",cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print("classifier report:\n",cr)

bias = classifier.score(x_train,y_train)
print("bias:",bias)

variance= classifier.score(x_test, y_test)
print("variance:",variance)

# ================== FUTURE PREDICTION ============================

dataset1 = pd.read_csv(r"E:\DATASCIENCE WITH GEN AI & AGENTIC AI\final1.csv")

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred = pd.DataFrame()

d2 ['y_pred1'] = classifier.predict(M)
d2.to_csv('final1.csv')

# to get the path
import os
print("Path:",os.getcwd())

#--

from sklearn.metrics import roc_auc_score, roc_curve
y_pred_prob = classifier.predict_proba(x_test)[:,1]

auc_score = roc_auc_score(y_test,y_pred_prob)
print("AUC_Score:",auc_score)

fpr, tpr,thresholds = roc_curve(y_test,y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'Logisitc Regression (AUC = {auc_score:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


 






