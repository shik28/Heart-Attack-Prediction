import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import time

import warnings
warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 35)

#Exploratory analysis
#Load the dataset and do some quick exploratory analysis.

data = pd.read_csv('HeartAttack_data.csv', index_col=False)

print("\n\n\nShape of the heart attack data = ", end="")
print( data.shape)
print("\n\n\nHeart attack description : \n")
print( data.describe())
print(data.dtypes)
#data.rename(columns={'num':'num'},inplace=True)
print(data.columns)

data.replace('?',np.nan,inplace=True)

data["thalach"] = pd.to_numeric(data["thalach"], downcast="float")
data["age"] = pd.to_numeric(data["age"], downcast="float")
data["sex"] = pd.to_numeric(data["sex"], downcast="float")
data["cp"] = pd.to_numeric(data["cp"], downcast="float")
data["trestbps"] = pd.to_numeric(data["trestbps"], downcast="float")
data["chol"] = pd.to_numeric(data["chol"], downcast="float")
data["restecg"] = pd.to_numeric(data["restecg"], downcast="float")
data["exang"] = pd.to_numeric(data["exang"], downcast="float")
data["fbs"] = pd.to_numeric(data["fbs"], downcast="float")
del data['slope']
del data['ca']
del data['thal']
#data=data.astype('float64')
print("\n\n\nSample Heart attack data set(30) :- \n\n", data.head(30) )

print(data.dtypes)

#data=data.fillna(data.median(),inplace=True)
for col in['fbs','trestbps','chol','restecg','thalach','exang']:
    data[col].fillna(data[col].mode()[0],inplace=True)
print("\n\n\nSample Heart attack data set(30) :- \n\n", data.head(30) )

plt.hist(data['num'])
plt.title('Num')
plt.show()


data.plot(kind='density', subplots=True, layout=(4,3), sharex=False, legend=False, fontsize=1)
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr() )
ax1.grid(True)
plt.title('Attributes Correlation')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()


Y = data['num'].values
X = data.drop('num', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=25)


pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))


results = []
names = []
num_folds=10


print("\n\n\nAccuracies of algorithm after scaled dataset\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model = GaussianNB()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm

end = time.time()
print( "\n\nGUASSIAN NB Training Completed. It's Run Time: %f" % (end-start))


# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by GaussianNB Machine Learning Algorithms")
print("\nAccuracy score = %f\n" % accuracy_score(Y_test, predictions))
#print("\n")
print("Confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))

from sklearn.externals import joblib
filename =  "finalized_HeartAttack_model.sav"
joblib.dump(model, filename)
print( "Best Performing Model dumped successfully into a file by Joblib")
