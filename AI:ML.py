#import libraries
import pandas as pd

#importing dataset
dataset_Destination = pd.read_csv('destinations.csv')
dataset_sample_submission = pd.read_csv('sample_submission.csv')
dataset_train = pd.read_csv('test.csv')
dataset_test = pd.read_csv('train.csv')

#remove colums to make test and train data have same columns
for col in dataset_test:
    if(col in dataset_train):
        pass
    else:
        del dataset_test[col]
for col in dataset_train:
    if(col in dataset_test):
        pass
    else:
        del dataset_train[col]

#split dependent and indepandent data
X = dataset_train.iloc[:, :]
y = dataset_train.iloc[:, 16]
X_test = dataset_test.iloc[:, :]
y_test = dataset_test.iloc[:, 16]
del X[y.name]
del X_test[y_test.name]

#categorical data
from sklearn.preprocessing import LabelEncoder#,OneHotEncoder
LEX_obj = LabelEncoder()
for col in X.columns:
    X[col] = LEX_obj.fit_transform(X[col].astype(str))
for col in X_test.columns:
    X_test[col] = LEX_obj.fit_transform(X_test[col].astype(str))
    
X = X.values
X_test = X_test.values
y = y.values
y_test = y_test.values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

#missing data
from sklearn.preprocessing import Imputer
imputer_obj = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer_obj = imputer_obj.fit(X)
X = imputer_obj.transform(X)
X_test = imputer_obj.transform(X_test)

# prepare configuration for cross validation test harness
seed = 7

#import algo for model selection
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# prepare models
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier(n_neighbors=3)))

bestAccuracy = 1

# evaluate each model in turn
for name,model in models:
    m = model
    print("\n",name,"**** processing ****")
    m.fit(X,y)
    y_pred = m.predict(X_test[20000:])
    if(met.accuracy_score(y_test[20000:],y_pred)<bestAccuracy):
        bestModel = m
        bestModelName = name
        bestAccuracy = met.accuracy_score(y_test[20000:],y_pred)
    print("\n",name,"Accuracy Score",met.accuracy_score(y_test[20000:],y_pred))
    
print("\nBest model is",bestModelName,"With Accuracy",bestAccuracy)
finalpred = bestModel.predict(X_test[:100])
print(finalpred)
print(y_test[:100])
#y_pred = m.predict(X_test)
#print(name,"Accuracy Score",met.accuracy_score(y_test,y_pred))
    
#alternate method to evalute algo
'''takes longer time to execute
from sklearn import model_selection as mod_sel
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    print(name,"****alternate method processing")
    kfold = mod_sel.KFold(n_splits=10, random_state=seed)
    cv_results = mod_sel.cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
'''