import numpy as np;
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

def displayDigit(index, pixeldata):
	plt.imshow(pixeldata[index,:,:,1], cmap=cm.gray)

traindata_path = 'mnist_train.csv'
traindata = np.loadtxt(traindata_path, delimiter=',', skiprows=1)
testdata_path = 'mnist_test.csv'
testdata = np.loadtxt(testdata_path, delimiter=',', skiprows=1)

n_of_train = traindata[:,0].size
n_of_test = testdata[:,0].size
#in order to use keras model, one-hot encoding for the outputs ex: [{0,0,0,1}, {0,1,0,0}] NOT [3,1]
# y_train==np.zeros( (n_of_train,10) )
# y_test=np.zeros( (n_of_test,10) )
# y_train=[np.arange(9), traindata[:,0].reshape(n_of_train, 1).astype(int)] = 1
# y_test[np.arange(9), testdata[:,0].reshape(n_of_test, 1).astype(int)] = 1

y_train = to_categorical(traindata[:,0], num_classes = 10)
y_test = to_categorical(testdata[:,0], num_classes = 10)
#shape into Dense shape (training examples, y_test) 
x_train = np.array(traindata[:, 1:785]).reshape(n_of_train, 784)
x_test = np.array(testdata[:, 1:785]).reshape(n_of_test, 784)

#rescale 0-255 to 0-1
x_train = x_train.astype('float32') / 255 
x_test = x_test.astype('float32') / 255

model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Dense(40, input_shape = (784,), activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

model.compile(optimizer= 'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10, batch_size=32)
y_hat = model.predict(x_test)

model.summary()

perfmetrics = model.evaluate(x=x_test, y=y_test, verbose=0)
print('Test loss:', perfmetrics[0])
print('Test accuracy:', perfmetrics[1])

save_model(model, 'my_model.h5', overwrite= True, include_optimizer = True)  
model = load_model('my_model.h5')
perfmetrics2 = model.evaluate(x=x_test, y=y_test, verbose=0)
print('Test loss:', perfmetrics2[0])
print('Test accuracy:', perfmetrics2[1])

plt.show()




