import glob
import os
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split


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


class MyLDA:
    def __init__(self, n_components=1):
        self.model = None
        self.n_components = n_components
        self.trained_imgs = None
        self.train_labels = None

    def train(self, imgs, labels, precision):
        avg = np.mean(imgs, axis=0)  # 平均脸
        diff = imgs - avg            # 差值脸
        cov = np.cov(diff)           # 协方差矩阵

        # 求特征值和特征向量，构造特征脸空间
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_val = np.sort(eig_val)
        # 根据贡献率选取前p个
        eig_sum = np.sum(eig_val)
        eig_cum = np.cumsum(eig_val)
        rate = eig_cum / eig_sum                    # 贡献率
        # n_components = np.argmax(rate > precision) + 1
        n_components = min(41 - 1, self.n_components)    # LDA: min(n_features, n_classes - 1)

        self.model = LDA(n_components=n_components)
        self.trained_imgs = self.model.fit_transform(imgs, labels)  # 标准化正则化
        # self.n_components = n_components

        trained_imgs = []
        trained_labels = []
        label_size = np.unique(labels).size
        # 计算类平均脸
        for i in range(1, label_size + 1):
            index = np.where(labels == i)                   # 找到该标签(类)的所有索引
            img_avg = np.mean(self.trained_imgs[index], axis=0)
            trained_imgs.append(img_avg)
            trained_labels.append(i)

        self.trained_imgs = np.array(trained_imgs)
        self.train_labels = np.array(trained_labels)

    def predict(self, test_imgs, test_labels):
        pre_labels = []
        tested_imgs = self.model.transform(test_imgs)

        # 找到距离待识别人脸最近待的类平均脸
        for i in range(len(tested_imgs)):
            diff = self.trained_imgs - tested_imgs[i]
            squ = np.square(diff)
            distance = np.sum(squ, axis=1)
            min_distance = np.argmin(distance)
            pre_labels.append(self.train_labels[min_distance])

        # 计算准确率
        cnt = 0
        labels_len = len(pre_labels)
        for i in range(labels_len):
            if test_labels[i] == pre_labels[i]:
                cnt = cnt + 1
        acc = cnt / labels_len
        # print("LDA accuracy:", acc)
        return acc


if __name__ == '__main__':
    path = 'Grp13Dataset/'
    width = 70
    height = 80
    imgs, labels = read_img(path, width, height)
    X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels, test_size=0.25)

    model = MyLDA()
    model.train(X_train, Y_train, 0.99)
    model.predict(X_test, Y_test)
    print("n_components: " + str(model.n_components))
