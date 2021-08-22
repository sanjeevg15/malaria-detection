from keras.models import Sequential
from keras import Model
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dropout, Dense, Input, GlobalAveragePooling2D
from utils import gaussian_blur
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

# Build a deep neural network model
model_3l = Sequential()
model_3l.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_3l.add(BatchNormalization())
model_3l.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_3l.add(MaxPool2D(pool_size=(2,2)))
model_3l.add(BatchNormalization()) 
model_3l.add(Dropout(0.4))
model_3l.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model_3l.add(MaxPool2D(pool_size=(2,2)))
model_3l.add(BatchNormalization())
model_3l.add(Dropout(0.4))
model_3l.add(Flatten())
model_3l.add(Dense(2, activation='softmax'))

model_3lp = Sequential()
model_3lp.add(gaussian_blur)
model_3lp.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_3lp.add(BatchNormalization())
model_3lp.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_3lp.add(MaxPool2D(pool_size=(2,2)))
model_3lp.add(BatchNormalization()) 
model_3lp.add(Dropout(0.4))
model_3lp.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model_3lp.add(MaxPool2D(pool_size=(2,2)))
model_3lp.add(BatchNormalization())
model_3lp.add(Dropout(0.4))
model_3lp.add(Flatten())
model_3lp.add(Dense(2, activation='softmax'))

resnet50 = ResNet50(include_top=False, input_shape=(100,100,3), weights=None)
vgg16 = VGG16(include_top=False, input_shape=(100,100,3), weights=None)

for layer in vgg16.layers:
    layer.trainable = False
for layer in resnet50.layers:
    layer.trainable = False

inputs = Input(shape=(100,100,3))
v = vgg16(inputs, training=False)
r = resnet50(inputs, training=False)
v = GlobalAveragePooling2D()(v)
r = GlobalAveragePooling2D()(r)
outputs_v = Dense(2, activation='softmax')(v)
outputs_r = Dense(2, activation='softmax')(r)
model_v = Model(inputs, outputs_v)
model_r = Model(inputs, outputs_r)