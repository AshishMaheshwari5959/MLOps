import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

dataset = pd.read_csv('weight-height.csv')

y = dataset['Weight']
X = dataset[['Height']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = Sequential()

model.add(Dense(units=1 , input_shape=(1,)  ))

earlystop = EarlyStopping(monitor = 'loss', min_delta = 0.001,patience = 3,verbose = 0,restore_best_weights = True)
checkpoint = ModelCheckpoint("weight.h5",monitor="loss",mode="min",save_best_only = True,verbose=0)

callbacks = [earlystop]

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.1) ,metrics=['accuracy'] )
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks = callbacks,epochs=5,verbose=0)

def compile(x):
    global c
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=x) ,metrics=['accuracy'] )
    hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks = callbacks,epochs=20,verbose=0)
    a = hist.history['loss'][-1]
    b = hist.history['loss'][-2]
    c = abs(a-b)
    return c
c=2
l =[]
s=0
while c > 0.0001 :
    if c >= 1:
        compile(0.1*c)
    elif 1 > c >= 0.1:
        compile(0.1*c)
    elif 0.1 > c >= 0.01:
        compile(0.1*c)
    elif 0.01 > c >= 0.001:
        compile(0.1*c)
    l.append(model.evaluate(X_train,y_train,verbose=0)[0])
    s=s+1
    if s >4 :
        p = l[-1]
        q = l[-2]
        r = l[-3]
        if p>q and p>r :
            break
        if l[-1] == min(l):
            model.save('weight.h5')
        
model = load_model('weight.h5')
print(model.evaluate(X_train,y_train,verbose=0)[0])
