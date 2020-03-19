import config
import cv2
import pandas as pd
import numpy as np


def generate_dataset(size):
    # label
    config.label = pd.read_csv(config.label_path).iloc[:, 1].values

    # dataset
    minx = miny = 10000
    for i in range(100):
        image = cv2.imread(config.data_path + str(i + 1) + config.data_format)
        minx = min(minx, image.shape[0])
        miny = min(miny, image.shape[1])
        print(image.shape)
    if (size - minx) % 2 != 0:
        minx -= 1
    if (size - miny) % 2 != 0:
        miny -= 1

    print(minx)
    print(miny)
    # format fixed size
    for i in range(100):
        img = cv2.imread(config.data_path + str(i + 1) + config.data_format)
        cut_img = img[int((img.shape[0] - minx) / 2):int((img.shape[0] - minx) / 2 + minx),
                  int((img.shape[1] - miny) / 2):int((img.shape[1] - miny) / 2 + miny)]
        cut_img = cv2.copyMakeBorder(cut_img, int((size - minx) / 2), int((size - minx) / 2), int((size - miny) / 2),
                                     int((size - miny) / 2),
                                     cv2.BORDER_REPLICATE)
        # cut_img = img[int((img.shape[0] - minx) / 2):int((img.shape[0] - minx) / 2 + minx),
        #           int((img.shape[1] - miny) / 2):int((img.shape[1] - miny) / 2 + miny)]
        cv2.imwrite(config.processed_data_path + str(i + 1) + config.data_format, cut_img)

    for i in range(100):
        img = cv2.imread(config.processed_data_path + str(i + 1) + config.data_format)
        config.data.append(np.array(img))
    config.data = np.array(config.data)
    print(config.data[1].shape)
    print(config.data[2].shape)
    # print(config.label)
