# XG BOOST

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\DATASCIENCE WITH GEN AI & AGENTIC AI\Churn_Modelling.csv")

x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

#labelencoding the 'Gender' Column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2]= le.fit_transform(x[:,2])

print(x)

# OneHot encoding the 'Geography' Column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')

x= np.array(ct.fit_transform(x))


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



from xgboost import XGBClassifier
classifier = XGBClassifier()
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


# Apply K-Fold CrossValidation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X= x_train,y=y_train,cv=5)
print("Accuracy:{:.2f}%".format(accuracies.mean()*100))
#print("Standard Deviation: {")
