import pandas as pd
from sklearn.preprocessing import StandardScaler
heart=pd.read_csv("heart.csv")


X=heart.iloc[:,0:13]
y=heart.iloc[:,13:14]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout
from keras import models
#Creating a pipeline
model = models.Sequential()
#1st hidden layer with input layer
model.add(Dense(units=145,activation="relu",input_dim=13))
#2nd hidden layer
model.add(Dense(units=120,activation="relu",))
#3rd hidden layer
model.add(Dense(units=70,activation="relu",))
#output layer
#model.add(Dense(units=1,activation="sigmoid"))
model.add(Dense(1, activation=tf.nn.sigmoid))

#model Summary
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model_his=model.fit(X_train,y_train,validation_split=0.1, batch_size=55,epochs=50,verbose=1)


y_pred=model.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)


print(classification_report(y_test,y_pred))


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model_his.history['accuracy'])
plt.plot(model_his.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(model_his.history['loss'])
plt.plot(model_his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('heart.h5',model_his)


print("model created")