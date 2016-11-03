import math

# Returns the mean of the vector a
def mean ( a ) :
    sum = 0.0

    for i in a:
        sum += i

    return float(sum) / len(a)

# Returns the variance of a
def var ( a ) :
    myMean = mean(a)

    sumOfSquares = 0.0

    for i in a:
        sumOfSquares += (i - myMean)**2

    return sumOfSquares / len(a)

# Returns the standard deviation of a
def std ( a ) :
    return math.sqrt(var(a)) # std dev is just the square root of the variance


# Returns the transpose of the given matrix
# Turns out I didn't need it, but I wrote it so I'll keep it just in case
def transpose(m):
    # Get the number of rows and columns (to simplify looping)
    numRows = len(m)
    numCols = len(m[0])

    # Transpose the matrix
    transposed = [[0 for x in range(0, numRows)] for x in range(0, numCols)]
    for x in range(numRows):
        for y in range(numCols):
            transposed[y][x] = m[x][y];

    return transposed


# Returns the covariance matrix of the given matrix
# NB: The input matrix is already transposed. Each element of the top level array holds a variable, with a list of observations for that variable
def cov (m):

    # Get the number of rows (to simplify looping)
    numRows = len(m)

    # Calculate the mean of every column vector (or every row in the transposed matrix)
    means = [0 for x in range(numRows)]
    for i in range(numRows):
        means[i] = mean(m[i])

    # For each "feature" (column) in the original data matrix, calculate its covariance against every other column

    # Initialize the covariance matrix, it's a square matrix of size numRows x numRows (since the rows are holding the columns)
    covariance = [[0.0 for x in range(0, numRows)] for x in range(0, numRows)]

    # For each variable (row), we'll calculate its variance against every other column, starting with itself and working our way up
    for x in range(numRows):
        for y in range(x, numRows):
            # Sum variance between columns x and y from i=0 to i=len(x), assuming x and y have equal number of observations
            variance = 0.0
            for i in range(len(m[x])):
                variance += (m[x][i] - means[x])*(m[y][i] - means[y])

            covariance[x][y] = variance / (len(m[x])-1)
            covariance[y][x] = variance / (len(m[x])-1)

    return covariance


# Median - Bonus Method
def median(u):
    sortedU = sorted(u)

    length = len(u)

    if length % 2 == 0:
        # List has even number of elements, so find middle two elements
        elementNumber = int(length / 2)

        # Return mean of two middle elements
        return (sortedU[elementNumber] + sortedU[elementNumber - 1]) / 2;
    else:
        return sortedU[math.ceil(length / 2) - 1]


def nanmean(u):
    numNonNan = 0
    mySum = 0.0

    for i in u:
        if(not math.isnan(i)):
            numNonNan += 1
            mySum += i

    if numNonNan == 0:
        return math.nan

    return mySum / numNonNan


def percentile(v, p):

    # Sort the list of values
    v = sorted(v)

    # Calculate the index
    index = (p / 100) * (len(v) - 1)

    # If index is a whole number, then return the value at that index
    if isinstance(index, int) and index.is_integer():
        return v[index]

    # Calculate the fractional value between two actual values
    indexFloor = math.floor(index)
    fraction = index - indexFloor

    # Shouldn't have to do this, but the 'isinstance(index, int) above doesn't catch floats like 99.0
    if fraction == 0.0:
        return v[indexFloor]

    value = v[indexFloor] + fraction * (v[indexFloor + 1] - v[indexFloor])
    return value


def histogram(v, bins = 10):

    # Sort v
    v = sorted(v)

    # Find the minimum and maximum values in v
    minValue = v[0]
    maxValue = v[len(v) - 1]

    dataRange = (maxValue - minValue)

    binSize = dataRange / bins

    # Initialize the list of bin edges
    bin_edges = []

    # The first edge is the minimum value
    binEdge = minValue

    # Add the binSize to the last binEdge until the edge's value is greater than the maxValue
    while(binEdge < maxValue):
        bin_edges.append(binEdge)
        binEdge += binSize

    # Always add the maxValue as the final edge
    bin_edges.append(maxValue)

    # Initialize the list of counts of how many values are in each bin
    hist = [0 for i in range(len(bin_edges) - 1)]

    # Count values in each bin
    # For each bin, search the data for values that fit in this bin
    for binNum in range(len(bin_edges) - 1):
        # Search data vector and count values between bin_edge[i] and bin_edge[i+1]
        for val in v:
            # If we're on the last bin, then the range is inclusive
            if binNum + 1 == len(bin_edges) - 1:
                if val >= bin_edges[binNum] and val <= bin_edges[binNum + 1]:
                    hist[binNum] += 1
            else:
                if val >= bin_edges[binNum] and val < bin_edges[binNum + 1]:
                    hist[binNum] += 1

    return hist, bin_edges

def histogram2d(x, y, bins = 10):
    # Find the min and max values of each list
    minXValue = min(x)
    maxXValue = max(x)

    minYValue = min(y)
    maxYValue = max(y)

    # Find the data rage of each data vector
    dataXRange = maxXValue - minXValue
    dataYRange = maxYValue - minYValue

    # Calculate the bin size for each dimension
    binSizeX = dataXRange / bins
    binSizeY = dataYRange / bins

    # Find the bin edges in x
    xedges = []
    binEdge = minXValue
    while binEdge < maxXValue:
        xedges.append(binEdge)
        binEdge += binSizeX

    xedges.append(maxXValue)

    # Find the bin edges in y
    yedges = []
    binEdge = minYValue
    while binEdge < maxYValue:
        yedges.append(binEdge)
        binEdge += binSizeY
    yedges.append(maxYValue)

    # Initialize a 2D list for histogram counts
    h = [[0 for a in range(len(yedges) - 1)] for b in range(len(xedges) - 1)]

    # For each value, decide which bin it's in
    for i in range(len(x)):
        # Determine which bin the X value is in
        for xedgeNum in range(len(xedges) - 1):
            if xedgeNum + 1 == len(xedges) - 1:
                if x[i] >= xedges[xedgeNum] and x[i] <= xedges[xedgeNum + 1]:
                    xIndex = xedgeNum
            else:
                if x[i] >= xedges[xedgeNum] and x[i] < xedges[xedgeNum + 1]:
                    xIndex = xedgeNum

        for yedgeNum in range(len(yedges) - 1):
            if yedgeNum + 1 == len(yedges) - 1:
                if y[i] >= yedges[yedgeNum] and y[i] <= yedges[yedgeNum + 1]:
                    yIndex = yedgeNum
            else:
                if y[i] >= yedges[yedgeNum] and y[i] < yedges[yedgeNum + 1]:
                    yIndex = yedgeNum

        h[xIndex][yIndex] += 1

    return h, xedges, yedges
