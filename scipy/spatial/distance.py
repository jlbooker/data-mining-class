import math


# Euclidean Distance
def euclidean(u, v):
    sum = 0
    for i in range(len(u)):
        sum += (u[i] - v[i]) ** 2  # Sigma ((ui - vi)^2)

    return math.sqrt(sum)


# Cityblock Distance
def cityblock(u, v):
    sum = 0
    for i in range(len(u)):
        sum += math.fabs(u[i] - v[i])

    #return math.fabs(sum)
    return sum


# Cosine Distance
def cosine(u, v):

    dotProduct = 0
    for i in range(len(u)):
        dotProduct += (u[i] * v[i])

    lenU = lengthOfVector(u)
    lenV = lengthOfVector(v)

    return 1 - safe_devide(dotProduct, (lenU * lenV))


# Helper method for computing the length of a vector
def lengthOfVector(u):
    length = 0
    for i in range(len(u)):
        length += (u[i]**2)

    return math.sqrt(length)


# Helper method for handling divide by zero safely
def safe_devide(a, b):
    if b == 0:
        if a == 0:
            return math.nan
        elif a > 0:
            return math.inf
        elif a < 0:
            return -math.inf
    else:
        return a / b
    pass


# Compute distance between each pair of the two inputs, using the given distance measure
def cdist(xa, xb, metric='euclidean'):

    distances = [[0 for i in range(len(xa))] for j in range(len(xb))]
    for i in range(len(xa)):
        for j in range(len(xb)):
            if metric == 'euclidean':
                distances[i][j] = euclidean(xa[i], xb[j])
            elif metric == 'cosine':
                distances[i][j] = cosine(xa[i], xb[j])
            elif metric == 'cityblock':
                distances[i][j] = cityblock(xa[i], xb[j])
            else:
                print("Similarity metric not implemented: %s" % metric)

    return distances


###### Extra Credit Distance Measures

# Minkowski Distance
def minkowski(u, v, p):
    sum = 0
    for i in range(len(u)):
        sum += math.fabs(u[i] - v[i]) ** p

    return sum ** (1/p)


# Chebyshev Distance
def chebyshev(u, v):
    maxVal = 0
    for i in range(len(u)):
        thisVal = math.fabs(u[i] - v[i])

        if(thisVal > maxVal):
            maxVal = thisVal

    return maxVal


# Correlation
def correlation(u, v):
    # Calculate mean of vector U
    sumU = 0.0
    for i in u:
        sumU += i

    meanU = sumU / len(u)

    # Calculate the mean of vector V
    sumV = 0.0
    for j in v:
        sumV += j

    meanV = sumV / len(v)

    # Calculate the length of U
    sumUSquares = 0.0
    for k in u:
        sumUSquares += (k - meanU) ** 2
    varianceU = sumUSquares


    # Calculate the length of V
    sumVSquares = 0.0
    for l in v:
        sumVSquares += (l - meanV) ** 2
    varianceV = sumVSquares


    # Covariance of u, v
    myVariance = 0.0
    for m in range(len(u)):
        myVariance += ((u[m] - meanU)*(v[m] - meanV))

    # Return cov(u,v)/length(u)*length(v)
    return 1 - (myVariance / (math.sqrt(varianceU) * math.sqrt(varianceV)))

# Hamming Distance
def hamming(u, v):
    count = 0
    for i in range(len(u)):
        if u[i] != v[i]:
            count += 1

    return count / len(u)


# Jaccard Distance
def jaccard(u, v):
    mismatches = 0
    totalNonFalse = 0

    for i in range(len(u)):
        if((u[i] is True and v[i] is False) or (u[i] is False and v[i] is True)):
            mismatches += 1
        elif (u[i] is True or v[i] is True):
            totalNonFalse += 1

    if (mismatches + totalNonFalse) == 0:
        return 0
    else:
        return mismatches / (mismatches + totalNonFalse)


# Weighted Minkowski Distance
def wminkowski(u, v, p, w):
    sum = 0
    for i in range(len(u)):
        sum += (w[i] * math.fabs(u[i] - v[i])) ** p

    return sum ** (1/p)
    pass


# Mahalanobis Distance
def mahalanobis(u, v, ci):

    uMinusV = []
    for i in range(len(u)):
        uMinusV.append(u[i] - v[i])

    distance = (uMinusV[0] * ((uMinusV[0]*ci[0][0]) + (uMinusV[1]*ci[1][0]))) + (uMinusV[1] * ((uMinusV[0]*ci[0][1]) + (uMinusV[1]*ci[1][1])))

    return math.sqrt(distance)
