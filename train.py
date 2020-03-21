import loadData
import config
import keras.utils as up_utils
from sklearn.model_selection import train_test_split
import models.Googlenet as googlenet
from keras.optimizers import Adam
import models.cnn as cnn


if __name__ == '__main__':
    loadData.generate_dataset()
    data = config.data
    label = config.label

    label = up_utils.to_categorical(label)

    # split train_set and test_set
    (trainX, testX, trainY, testY) = train_test_split(
        data,
        label,
        shuffle=True,
        test_size=0.25,
        random_state=40
    )

    # basic cnn
    model = cnn.getmodel(width=data.shape[1], height=data.shape[2], channel=3, classes=2)

    # googleNet
    # model = googlenet.getmodel(width=data.shape[1], height=data.shape[2], channel=3, classes=2)

    # AlexNet
    # model =

    # vgg16net
    # model =


    # select optimizer
    opt = Adam(lr=config.INIT_LR,decay=config.INIT_LR/config.EPOCHS)
    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=['accuracy']
    )
    H = model.fit(
        trainX,
        trainY,
        batch_size=32,
        epochs=config.EPOCHS,
        validation_data=(testX,testY)
    )


