from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    iris = load_iris()

    pca = PCA(n_components=2)
    pca.fit(iris.data)

    newData = pca.transform(iris.data)

    plt.scatter(newData[iris.target == 0, 0], newData[iris.target == 0, 1], c='red', marker='o', edgecolor='red')
    plt.scatter(newData[iris.target == 1, 0], newData[iris.target == 1, 1], c='green', marker='*', edgecolor='green')
    plt.scatter(newData[iris.target == 2, 0], newData[iris.target == 2, 1], c='blue', marker='s', edgecolor='blue')

    plt.legend(['setosa', 'versicolor', 'virginica'])

    plt.savefig('./files/pca.png')


if __name__ == '__main__':
    main()
