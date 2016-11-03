#import matplotlib.pyplot as plt
from matplotlib.pyplot import clf, gray, colorbar, xlabel, ylabel, title, savefig
from matplotlib.pyplot import boxplot, pie, legend, scatter, gca, clabel, contour
from matplotlib.pyplot import axis, plot, hist2d, figure, subplot
from numpy import percentile
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from numpy import arange
from numpy import linspace
from numpy import histogram
from numpy import histogram2d


def get_elevation():
    from scipy.misc import imread
    image_file = '../data/n36_w082_1arc_v2_uint16_boone.tif'
    #image_file = 'cs5710/data/n36_w082_1arc_v2_uint16_boone.tif'
    img = imread(image_file)
    img *= 3

    return img

def main():
    iris = load_iris()


    # Figure 3.9
    counts, xedges, yedges, img = hist2d(iris.data[:, 2], iris.data[:, 3], bins=3)

    #plt.imshow(img)
    gray() # Use greyscale

    # Title and axis labels
    title('petal width (cm) vs. petal length (cm) Histogram')
    xlabel('petal length (cm)')
    ylabel('petal width (cm)')
    colorbar()

    savefig('./files/Fig3.9.png')


    # Figure 3.11
    clf()
    boxplt = boxplot([iris.data[:, 0], iris.data[:, 1], iris.data[:, 2], iris.data[:, 3]], labels=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    savefig('./files/Fig3.11.png')

    # Figure 3.12, with three sub figures
    clf()
    boxpltSetosa = boxplot([iris.data[iris.target == 0, 0], iris.data[iris.target == 0, 1], iris.data[iris.target == 0, 2], iris.data[iris.target == 0, 3]], labels=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    xlabel('setosa')
    savefig('./files/Fig3.12-0.png')

    clf()
    boxpltSetosa = boxplot([iris.data[iris.target == 1, 0], iris.data[iris.target == 1, 1], iris.data[iris.target == 1, 2], iris.data[iris.target == 1, 3]], labels=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    xlabel('versicolor')
    savefig('./files/Fig3.12-1.png')

    clf()
    boxpltSetosa = boxplot([iris.data[iris.target == 2, 0], iris.data[iris.target == 2, 1], iris.data[iris.target == 2, 2], iris.data[iris.target == 2, 3]], labels=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    xlabel('virginica')
    savefig('./files/Fig3.12-2.png')


    # Figure 3.13 - Piechart (pychart? haha)
    clf()
    counts, bin_edges = histogram(iris.target, bins=3)
    piechart = pie(counts, labels=['setosa', 'versicolor', 'virginica'])
    axis('square')

    savefig('./files/Fig3.13.png')


    # Figure 3.14, 0-3
    # Calculate the percentile for each point

    for i in range(4):
        clf()
        percentiles = []
        myRange = arange(0, 1, .001)
        for yvalue in myRange:
            percentiles.append(percentile(iris.data[:, i], yvalue * 100, interpolation='nearest'))

        plot(percentiles, myRange)

        if i == 0:
            title('sepal length (cm)')
        elif i == 1:
            title('sepal width (cm)')
        elif i == 2:
            title('petal length (cm)')
        elif i == 3:
            title('petal width (cm)')

        xlabel("x")
        ylabel("F(x)")
        savefig('./files/Fig3.14-' + str(i) + '.png')



    # Figure 3.15
    myRange = arange(0, 110, 10)
    sepalLengths = []
    sepalWidths = []
    petalLengths = []
    petalWidths = []

    for i in myRange:
        sepalLengths.append(percentile(iris.data[:, 0], i, interpolation='nearest'))
        sepalWidths.append(percentile(iris.data[:, 1], i, interpolation='nearest'))
        petalLengths.append(percentile(iris.data[:, 2], i, interpolation='nearest'))
        petalWidths.append(percentile(iris.data[:, 3], i, interpolation='nearest'))

    clf()

    fig = figure()
    ax = fig.add_subplot(111)

    ax.plot(myRange, sepalLengths, marker='o')
    ax.plot(myRange, sepalWidths, marker='s')
    ax.plot(myRange, petalLengths, marker='^')
    ax.plot(myRange, petalWidths, marker='d')


    legend(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], loc=2)
    xlabel('Percentile')
    ylabel('Value (centimeters)')
    savefig('./files/Fig3.15.png')


    # Figure 3.16 - 4x4
    for i in range(4):
        for j in range(4):
            # Skip this diagram if i == j
            if i == j:
                continue

            clf()
            scatter(iris.data[iris.target == 0,j], iris.data[iris.target == 0,i], c='red', marker='x')
            scatter(iris.data[iris.target == 1,j], iris.data[iris.target == 1,i], c='green', marker='+')
            scatter(iris.data[iris.target == 2,j], iris.data[iris.target == 2,i], c='blue', marker='o')

            if i == 0:
                ylabel("sepal length (cm)")
            elif i == 1:
                ylabel("sepal width (cm)")
            elif i == 2:
                ylabel("petal length (cm)")
            elif i == 3:
                ylabel("petal width (cm)")

            if j == 0:
                xlabel("sepal length (cm)")
            elif j == 1:
                xlabel("sepal width (cm)")
            elif j == 2:
                xlabel("petal length (cm)")
            elif j == 3:
                xlabel("petal width (cm)")

            legend(['setosa', 'versicolor', 'virginica'])

            savefig('files/Fig3.16-' + str(i) + str(j) + '.png')


    # Figure 3.17
    fig = figure()
    ax = subplot(111, projection='3d')

    ax.scatter(iris.data[iris.target == 0,1], iris.data[iris.target == 0,3], iris.data[iris.target == 0,0], c='blue', marker='+');
    ax.scatter(iris.data[iris.target == 1,1], iris.data[iris.target == 1,3], iris.data[iris.target == 1,0], c='green', marker='*');
    ax.scatter(iris.data[iris.target == 2,1], iris.data[iris.target == 2,3], iris.data[iris.target == 2,0], c='red', marker='x');

    legend(['setosa', 'versicolor', 'virginica'])
    ax.set_xlabel("sepal width (cm)")
    ax.set_ylabel("petal width (cm)")
    ax.set_zlabel("sepal length (cm)")
    savefig('files/Fig3.17.png')


    # Figure 3.19 - Contour Plot
    clf()
    elevations = get_elevation()
    #fig = figure()
    #ax = fig.add_subplot(111, projection='3d')
    x = arange(0, 1000)
    y = arange(0, 1000)
    contour(x, y, elevations, cmap='jet', levels=arange(1500, 5500, 500))

    gca().invert_yaxis()
    axis('image')
    colorbar()
    xlabel('Arc seconds')
    ylabel('Arc seconds')

    savefig('files/Fig3.19.png')

if __name__ == '__main__':
    main()
