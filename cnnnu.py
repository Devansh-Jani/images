from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 #for loading image


nu = cv2.imread('IMG_20180226_134414.jpg')
nu=cv2.resize(nu,(750,750))
print(nu.shape)
cv2.imshow('image',nu)

cv2.waitKey(0)
cv2.destroyAllWindows()


def visualize_nu(model, nu):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    nu_batch = np.expand_dims(nu,axis=0)
    conv_nu = model.predict(nu_batch)
    conv_nu = np.squeeze(conv_nu, axis=0)
    print(conv_nu.shape)
    cv2.imshow('image',conv_nu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

model = Sequential()
print("\nConvolving 5 times with 1 filter")
for i in range(5):
	model.add(Convolution2D(1,    # number of filter layers
							(3,    # y dimension of kernel (we're going for a 3x3 kernel)
							3),    # x dimension of kernel
							#strides=1,
							#use_bias=True,
							input_shape=nu.shape))
							
	#nu_batch = np.expand_dims(nu,axis=0)
	#conv_nu = model.predict(nu_batch)
	visualize_nu(model, nu)


print("\nConvolving 1 time with RELU:")
model = Sequential()
model.add(Convolution2D(1,    # number of filter layers
                        (3,    # y dimension of kernel (we're going for a 3x3 kernel)
                        3),    # x dimension of kernel
                        input_shape=nu.shape))

nu_batch = np.expand_dims(nu,axis=0)
model.add(Activation('relu'))
conv_nu = model.predict(nu_batch)
visualize_nu(model, nu)

print("\nApplying Max Pooling:")
model.add(MaxPooling2D(pool_size=(2,2)))
conv_nu = model.predict(nu_batch)
visualize_nu(model, nu)

print("\nConvolving 2 times with 2 RELUs and one pooling layer")
# 3 filters in both conv layers
model = Sequential()
model.add(Convolution2D(3,    # number of filter layers
                        (3,    # y dimension of kernel (we're going for a 3x3 kernel)
                        3),    # x dimension of kernel
                        input_shape=nu.shape))
# Lets activate then pool!
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(3,    # number of filter layers
                        (3,    # y dimension of kernel (we're going for a 3x3 kernel)
                        3),    # x dimension of kernel
                        input_shape=nu.shape))
# Lets activate then pool!
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

visualize_nu(model, nu)






