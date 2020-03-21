from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, BatchNormalization, Concatenate
from keras.models import Model
from keras import backend as K


def _Conv2D(x, _filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(_filter, kernel_size, strides=strides, padding=padding, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Inception(x, _filter):
    # 1*1 需要用(1,1)的kernel
    branch_1 = _Conv2D(x, _filter, (1, 1), padding='same', strides=(1, 1), name=None)

    # 3*3 需要用一次(1,1)和一次(3,3)
    branch_3 = _Conv2D(x, _filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch_3 = _Conv2D(branch_3, _filter, (3, 3), padding='same', strides=(1, 1), name=None)

    # 5*5 需要用两次(1,1)
    branch_5 = _Conv2D(x, _filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch_5 = _Conv2D(branch_5, _filter, (1, 1), padding='same', strides=(1, 1), name=None)

    # pool 用(3,3)的池化池
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _Conv2D(branch_pool, _filter, (1, 1), padding='same', strides=(1, 1), name=None)

    tem = Concatenate(axis=3)([branch_1, branch_3, branch_5, branch_pool])
    return tem


def getmodel(width, height, channel, classes):
    input_shape = Input(shape=(width, height, channel))
    # K.set_image_data_format('channels_first')
    # googlenet 不用顺序化的模型，通过api函数实现
    # vggnet input 为224
    # 什么时候写strides是需要用元组形式，什么时候不需要元组形式
    x = _Conv2D(input_shape, 64, (7, 7), strides=(2, 2), padding='same')
    # MaxPooling2D()
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = _Conv2D(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)
    x = Inception(x, 120)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)
    x = Inception(x, 208)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(1000, activation='softmax')(x)
    model = Model(input_shape, x, name='inception')
    model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# if __name__ == '__main__':
#     getmodel(224, 224, 3, 2)
