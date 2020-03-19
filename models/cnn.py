from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dropout, Dense, Flatten


def getmodel(width, height, channel, classes):
    model = Sequential()
    input_shape = (width, height, channel)
    chandim = -1

    # 如果第一次进通道=>init
    if K.image_data_format() == "channel_first":
        input_shape = (channel, height, channel)
        chandim = -1

    # model add layer    conv->relu->pool
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chandim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (conv->relu)*2->pool
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chandim))
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chandim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # FC fc->relu->softmax classfier
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chandim))
    model.add(Dropout(0.5))
    # model.add(Activation('sigmoid'))
    model.add(Dense(classes))  # 此处的classes是最后可以分出来的几种类
    model.add(Activation('softmax'))

    return model
