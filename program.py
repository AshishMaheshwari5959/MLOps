import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
sys.stderr = stderr

model = Sequential()

model.add(Convolution2D( filters = 32, kernel_size=(3,3), activation = 'relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units =128, activation = 'relu'))
model.add(Dense(units=4, activation = 'softmax'))
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

save = sys.stdout
sys.stdout = open("output.txt", "w+")

train_datagen = ImageDataGenerator( rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory( 'Faces/train/', target_size=(64, 64), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory( 'Faces/validation/', target_size=(64, 64), batch_size=32, class_mode='categorical')

sys.stdout.close()
sys.stdout = save

history = model.fit( training_set, steps_per_epoch=100, epochs=10, validation_data=test_set, validation_steps=20, verbose=0)

save = sys.stdout
sys.stdout = open("accuracy.txt", "w+")
print(100 * history.history['val_accuracy'][-1])
sys.stdout.close()
sys.stdout = save

print ("Accuracy of the trained model is : {} %".format ( 100 * history.history['val_accuracy'][-1])) 
model.save('program.h5')
