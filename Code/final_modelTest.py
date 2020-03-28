import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
dataset = pd.read_csv("C:\\Users\\ugur_\\Desktop\\MSCBD\\Thesis\\layer2\\mini_final.csv")

X = dataset
X = X.drop(columns = "type")

X = X.to_numpy()
y = dataset[["type"]]
#OnehotEncodingg class numeric-nominal
y = pd.get_dummies(y)
y = y.to_numpy()

#Split the Data into Training and Testing sets with test size as #30%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True)


####RANDOM FOREST
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
##

##DescisionTreeClassifier
# training a DescisionTreeClassifier 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 500).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
print("Descision Tree Classifier Accuracy:",dtree_model.score(X_test, y_test))
tree.plot_tree(dtree_model) 
##

##KNN classifier 
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)  
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print("kNN Accuracy:",accuracy) 
##

##OneVsRest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
print("OneVsRest LinearSVC Accuracy:",clf.score(X_test,y_test))

clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)
print("OneVsRest SVC Accuracy:",clf.score(X_test,y_test))
##



#Final Model
model = Sequential()
model.add(Dense(300, activation ='relu', input_dim=38))#1100
model.add(Dense(150,activation = 'relu' ))#relu
model.add(Dense(3, activation = 'softmax'))

# For a multi-class classification problem
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
pk =model.fit(X_train,y_train,batch_size = 100,epochs=450,verbose = 0,validation_split=0.2)
score,acc = model.evaluate(X_test,y_test,batch_size = 100,verbose=2)
print("Final model Accuracy: ",acc)
##

#MLP test optimizer
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
pk =model.fit(X_train,y_train,batch_size = 100,epochs=450,verbose = 0,validation_split=0.2)
score,acc = model.evaluate(X_test,y_test,batch_size = 100,verbose=2)
print("rmsprop accuracy: ",acc)
#
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
pk =model.fit(X_train,y_train,batch_size = 100,epochs=450,verbose = 0,validation_split=0.2)
score,acc = model.evaluate(X_test,y_test,batch_size = 100,verbose=2)
print("adaDelta accuracy: ",acc)
##END- test






