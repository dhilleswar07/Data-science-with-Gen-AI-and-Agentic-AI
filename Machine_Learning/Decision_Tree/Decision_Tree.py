
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\DATASCIENCE WITH GEN AI & AGENTIC AI\logit classification.csv")

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=  StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)
'''

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0,max_depth=4,n_estimators=30,criterion='entropy')
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