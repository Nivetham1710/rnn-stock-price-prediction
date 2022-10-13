# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Read the csv file and create the Data frame using pandas.
### STEP 2:
Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train.

### STEP 3:
Create a model with the desired number of nuerons and one output neuron.
### STEP 4:
Follow the same steps to create the Test data. But make sure you combine the training data with the test data.
### STEP 5:
Make Predictions and plot the graph with the Actual and Predicted values.


## PROGRAM
~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
dataset_train.tail()
dataset_train.iloc[1255:1257]
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shapesc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
## why 60 means we wnat the 60 values to predict the values
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  ##i=60 and we have to put only one values through all input in the y train array
  y_train_array.append(training_set_scaled[i,0])
  ## To convert into the numpy array
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
## 60 is the number of inputs we have to put through eachu values
X_train.shape
length = 60
n_features = 1
model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(length,n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam',loss='mse')
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
dataset_test.head()
test_set.shape
test_set = dataset_test.iloc[:,1:2]
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
dataset_total.shape
inputs = dataset_total.values
inputs
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
##increasing the shape here
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
#### inverse help us to convert the scaled price to the practical price
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
~~~




## OUTPUT

### True Stock Price, Predicted Stock Price vs time
![image](https://user-images.githubusercontent.com/94155183/195599193-92122dc7-dcd6-4cf0-a056-3351c71cbe37.png)

### Mean Square Error
/home/sec/Downloads/m1(2).png

## RESULT
Thus, we have successfully created a Simple RNN model for Stock Price Prediction.
