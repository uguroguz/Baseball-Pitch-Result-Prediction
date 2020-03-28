import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

#Model
model = Sequential()
model.add(Dense(300, activation ='relu', input_dim=38))#1100
model.add(Dense(150,activation = 'relu' ))#relu
model.add(Dense(3, activation = 'softmax'))

# For a multi-class classification problem
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# Train the model, iterating on the data in batches of 32 samples
pk =model.fit(X_train,y_train,batch_size = 100,epochs=450,verbose = 0,validation_split=0.2)
# evaluate the model
_, train_acc = model.evaluate(X_train, y_train,batch_size = 100, verbose=0)
_, test_acc = model.evaluate(X_test, y_test,batch_size = 100, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
#prediction
predict_y = model.predict(X_test)
predict_y= np.around(predict_y,decimals = 5)

score,acc = model.evaluate(X_test,y_test,batch_size = 100,verbose=2)
print("Test accuracy: ",acc)

# Save the model
model_json = model.to_json()
with open("./model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("./weight.h5")
model.save('model.h5')
np.save('xtest',X_test)
np.save('ytest',y_test)
np.save('xtrain',X_train)
np.save('ytrain',y_train)

print("success")


