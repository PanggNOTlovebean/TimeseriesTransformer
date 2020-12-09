import os ,datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as  pd
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

batch_size = 32
seq_len = 49

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

" data download  https://finance.yahoo.com/quote/IBM/history?period1=950400&period2=1594512000&interval=1d&filter=history&frequency=1d"

df = pd.read_csv('./input/szzs.csv', delimiter=',', usecols=['Date', 'Close'])



# Apply moving average with a window of 10 days to all columns
df[[ 'Close']] = df[[ 'Close']].rolling(10).mean() 

# Drop all rows with NaN values
df.dropna(how='any', axis=0, inplace=True) 


'''Calculate percentage change'''


df['Close'] = df['Close'].pct_change() # Create arithmetic returns column


df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values


'''Normalize price columns'''

min_return = min(df[['Close']].min(axis=0))
max_return = max(df[['Close']].max(axis=0))

# Min-max normalize price columns (0-1 range)

df['Close'] = (df['Close'] - min_return) / (max_return - min_return)


'''Create training, validation and test split'''
print(df)
times = sorted(df.index.values)
last_30pct = sorted(df.index.values)[-int(0.3*len(times))] # Last 30% of series

df_train = df[(df.index < last_30pct)]  # Training data are 80% of total data
df_val = df[(df.index >= last_30pct)]
df_test = df[(df.index >= last_30pct)]
test_tick=df[(df.index >= last_30pct)]['Date'][49:]
# Remove date column
df_train.drop(columns=['Date'], inplace=True)
df_val.drop(columns=['Date'], inplace=True)
df_test.drop(columns=['Date'], inplace=True)

# Convert pandas columns into arrays

train_data = df_train.values
val_data = df_val.values
test_data = df_test.values



# Training data
X_train, y_train = [], []
for i in range(seq_len, len(train_data)):
  X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
  y_train.append(train_data[:, 0][i]) #Value of 4th column (Close Price) of df-row 128+1
X_train, y_train = np.array(X_train), np.array(y_train)

###############################################################################

# Validation data
X_val, y_val = [], []
for i in range(seq_len, len(val_data)):
    X_val.append(val_data[i-seq_len:i])
    y_val.append(val_data[:, 0][i])
X_val, y_val = np.array(X_val), np.array(y_val)

###############################################################################

# Test data
X_test, y_test = [], []
for i in range(seq_len, len(test_data)):
    X_test.append(test_data[i-seq_len:i])
    y_test.append(test_data[:, 0][i])    
X_test, y_test = np.array(X_test), np.array(y_test)
print(X_train)
print(X_train.shape,y_train.shape)
print(y_train)
pangg=np.array([1,2,3,])
print(pangg.shape)
import model
model = model.create_model()
print(X_train)
print(y_train)
callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding_avg.hdf5', 
                                              monitor='val_loss', 
                                              save_best_only=True, 
                                              verbose=1)
history = model.fit(X_train, y_train, 
                    batch_size=batch_size,  
                    epochs=1, 
                    steps_per_epoch=len(X_train)/batch_size,
                    callbacks=[callback],
                    validation_data=(X_val, y_val)) 
model.load_weights('Transformer+TimeEmbedding_avg.hdf5')

y=model.predict(X_test)
fig=plt.figure()
import datetime
test_tick=test_tick.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))


true_y=y*(max_return - min_return)+min_return
true_ytest=y_test*(max_return - min_return)+min_return
df = pd.read_csv('./input/szzs.csv', delimiter=',', usecols=['Date','Close'])

real_close=df[(df.index >= last_30pct)]['Close'][48:-1].values
pre_close=[]

for i in range(len(real_close)):
	predict_close=real_close[i]*(1+true_y[i])
	pre_close.append(predict_close)

plt.plot(test_tick[-50:],df[(df.index >= last_30pct)]['Close'][49:].values[-50:],label="true",linewidth=1)
plt.plot(test_tick[-50:],pre_close[-50:],label="predict",linewidth=1)

plt.legend()
plt.show()