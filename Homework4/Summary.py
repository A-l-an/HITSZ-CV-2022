import glob
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from LDA import MyLDA
from KPCA import MyKPCA
from PCA import MyPCA
import matplotlib.pyplot as plt


def read_img(path, width, height):
    cate = [path + x for x in os.listdir(path)]
    cate.sort(key=lambda x: (x[:24], int(x[24:])))  # 将文件名的数字字符串转化为数字再排序
    imgs = []
    labels = []

    for index, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (width, height))
            imgs.append(np.array(img).flatten('F'))  # 化为一维
            labels.append(index + 1)

    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


if __name__ == '__main__':
    path = 'Grp13Dataset/'
    width = 70
    height = 80
    imgs, labels = read_img(path, width, height)
    X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels, test_size=0.25)

    PCA_x, PCA_y, KPCA_x, KPCA_y, LDA_x, LDA_y = [], [], [], [], [], []
    for i in range(1, 99):
        model = MyPCA(n_components=i)
        model.train(X_train, Y_train, 0.99)
        model.predict(X_test, Y_test)
        acc = model.predict(X_test, Y_test)
        PCA_x.append(i)
        PCA_y.append(acc)

        model = MyKPCA(n_components=i)
        model.train(X_train, Y_train, 0.99)
        model.predict(X_test, Y_test)
        acc = model.predict(X_test, Y_test)
        KPCA_x.append(i)
        KPCA_y.append(acc)

        model = MyLDA(n_components=i)
        model.train(X_train, Y_train, 0.99)
        model.predict(X_test, Y_test)
        acc = model.predict(X_test, Y_test)
        LDA_x.append(i)
        LDA_y.append(acc)

    # print("n_components: " + str(model.n_components))
    plt.xlabel('Number of principal components')
    plt.ylabel('Accuracy')
    plt.plot(PCA_x, PCA_y, "r--", label="PCA")
    plt.plot(KPCA_x, KPCA_y, "b-.", label="KPCA")
    plt.plot(LDA_x, LDA_y, "g-", label="LDA")

    plt.show()
