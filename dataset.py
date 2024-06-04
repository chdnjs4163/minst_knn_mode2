import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(x_test_flat.shape)
print(x_train_flat.shape)


from sklearn.neighbors import KNeighborsClassifier
# 모델 생성
knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
knn.fit(x_train_flat, y_train)
prediction = knn.score(x_test_flat,y_test)
print(prediction)

class MinstModel():
    def __init__(self):
        (x_train, y_train), (_, _) = mnist.load_data()
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        self.knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
        self.knn.fit(x_train_flat, y_train)

    def output(self,pic):
        pic_flat = pic.reshape(1, -1).astype('float32') / 255.0
        return self.knn.predict(pic)
