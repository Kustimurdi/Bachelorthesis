# Matrix exponential for the time development of an initial dipole
import numpy as np
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from random import *
import math
from scipy import optimize
import scipy.optimize
import cmath


N = 360
'''length of state'''
time = 3000000
W = 1
'''disorder parameter'''

bMeanDis = False
bProbDist = False
bProbDistTimeEvo = False
bStandDevTimeEvo = False
bStandDevSizeEvo = False
bProbDistSum = False
bStandDevDisEvo = False
bLocalLenDisEvo = False
bEigStateDist = True
bVarianceDisEvo = False

def calcH3MatOpenBound(N):
    '''creates h3 for a given amount of positions "N"'''
    h3Mat = np.zeros((N - 1, N - 1), dtype=np.complex64)

    #put ones on the off-diagonals
    for k in range(N - 2):
        h3Mat[k][k + 1] = -1
        h3Mat[k + 1][k] = -1

    return h3Mat


def calcH3MatPerBound(N):
    '''takes the length of the lattice: "N"
    returns the h3 hamiltonian with the respecting size '''
    h3Mat = np.zeros((N - 1, N - 1), dtype=np.complex64)

    for k in range(N - 2):
        h3Mat[k][k + 1] = -1
        h3Mat[k + 1][k] = -1

    h3Mat[N - 2][0] = -1
    h3Mat[0][N - 2] = -1
    return h3Mat


def calcHamTimeEvo(matrix, time):
    '''creates the time evolution for a given hamiltonian "matrix" and point of "time"'''
    matExp = matrix * complex(0, -1) * time
    matExp = expm(matExp)
    return matExp


def calcEigenTimeEvo(matrix, time):
    '''takes the hamiltonian matrix and the point of time: "matrix", "time" \n
    returns the time evolution of the hamiltonian using their eigenvalues and -states'''
    eigenvalues, eigenstatesMatrix = np.linalg.eigh(matrix)
    expMatrix = np.identity(len(matrix), complex)
    for i in range(len(eigenvalues)):
        expMatrix[i][i] = complex(0, -1) * time * eigenvalues[i]
    eigenstatesMatrixInv = np.linalg.inv(eigenstatesMatrix)
    matTimeEvo = np.dot(eigenstatesMatrix, expMatrix)
    matTimeEvo = np.dot(matTimeEvo, eigenstatesMatrixInv)
    return matTimeEvo


# creation of dipoles
def calcDipole(N, position, column):
    '''creates a vector with the lenght "N"-1 with a one at "position", for "column" == True it is a column vector, for "column" == False it is a row vector (dipole at position "position")'''

    if column:
        dipoleR = np.zeros((N - 1, 1), complex)
        dipoleR[position - 1][0] = 1
        return dipoleR

    else:
        dipoleL = np.zeros((1, N - 1), complex)
        dipoleL[0][position - 1] = 1
        return dipoleL


def calcProbDist(matrix):  # erster Versuch der Wahrscheinlichkeitsverteilung. funktioniert nicht so gut
    '''first try of a probability distribution function which doesn't work so well'''

    N = len(matrix) + 1
    diff = list(range(2 - N, N - 1))
    probDis = []
    for x in diff:
        if x > -1:
            dipoleA = np.zeros((N - 1, 1))
            dipoleA[N - 2][0] = 1
            dipoleB = np.zeros((1, N - 1))
            dipoleB[0][N - 2 - x] = 1
            dipoleATime = matrix.dot(dipoleA)
            skalpr = dipoleB.dot(dipoleATime)
            result = skalpr[0][0].real ** 2 + skalpr[0][0].imag ** 2
            probDis.append(result)
        else:
            dipoleA = np.zeros((1, N - 1))
            dipoleA[0][N - 2] = 1
            dipoleB = np.zeros((N - 1, 1))
            dipoleB[N - 2 + x][0] = 1
            dipoleBTime = matrix.dot(dipoleB)
            skalpr = dipoleA.dot(dipoleBTime)
            result = skalpr[0][0].real ** 2 + skalpr[0][0].imag ** 2
            probDis.append(result)
    result = []
    result.append(diff)
    result.append(probDis)
    return result


# probability distribution
def calcProbDistMiddle(timeEvoMat):  # man könnte es auch so umschreiben, dass es den hamiltonian nimmt und selber die time evo bestimmt
    '''takes the time evolution of the hamiltonian matrix: "timeEvoMat" \n
        returns a tuple of firstly the distance between an initial and a referential dipole and secondly the respecting probability distribution'''
    N = len(timeEvoMat)
    halfN = math.ceil(N / 2)

    #creates the list of distances between the inital and referential dipole
    if halfN % 2 == 0:
        diff = list(range(-halfN + 1, halfN + 1))
    else:
        diff = list(range(-halfN + 1, halfN))


    probDis = []

    #creates the time evolution of the inital dipole
    dipoleInit = np.zeros((N, 1))
    dipoleInit[halfN - 1][0] = 1
    dipoleInitTime = timeEvoMat.dot(dipoleInit)

    #calculates the overlap of the reference dipoles with the time evolved inital dipole
    dipoleRef = np.zeros((1, N))
    for x in diff:
        dipoleRef[0][halfN - x - 1] = 1
        skalProd = dipoleRef.dot(dipoleInitTime)
        result = skalProd[0][0].real ** 2 + skalProd[0][0].imag ** 2
        probDis.append(result)
        dipoleRef[0][halfN - x - 1] = 0

    ergebnis = []
    ergebnis.append(diff)
    ergebnis.append(probDis)
    return ergebnis


def plotDist(plotName, tupleOfListsOfValues, xlabel, ylabel, title, label):
    '''takes a tuple of lists with x and y values: "tupleOfListsOfValues
        (is not used)'''
    plotName.plot(tupleOfListsOfValues[0], tupleOfListsOfValues[1], label=label)
    plotName.set_xlabel(xlabel)
    plotName.set_ylabel(ylabel)
    plotName.set_title(title)


def plotDistEnh(plotName, tupleOfListsOfValues, xlabel, ylabel, title):
    '''takes a tuple of lists with x and y values: "tupleOfListsOfValues \n
        takes three strings for the x label, the y label and the title: "xlabel", "ylabel", "title"
        returns the respecting plot'''
    plotName.plot(tupleOfListsOfValues[0], tupleOfListsOfValues[1])
    plotName.set_xlabel(xlabel)
    plotName.set_ylabel(ylabel)
    plotName.set_title(title)


def scatterDistEnh(plotName, tupleOfListsOfValues, xlabel, ylabel, title):
    '''takes a tuple of lists with x and y values: "tupleOfListsOfValues \n
        takes three strings for the x label, the y label and the title: "xlabel", "ylabel", "title"
        returns the respecting plot'''
    plotName.plot(tupleOfListsOfValues[0], tupleOfListsOfValues[1], 'o')
    plotName.set_xlabel(xlabel)
    plotName.set_ylabel(ylabel)
    plotName.set_title(title)


# adding the disorder
def calcDisorder(N, disorderParameter):  # kreiert die additionale Matrix der Störung
    '''takes an amount of positions "N" and a disorder parameter "disorderParameter" \n
        returns the respecting disorder matrix'''
    W = disorderParameter

    #goes through the positions, appoints to them random values between [-W,W]
    hDis = []
    for i in range(N):
        h = uniform(-W, W)
        hDis.append(h)

    #creates a matrix and places the difference of the values of two adjacent positions on the respecting position of the diagonal of the matrix
    disMat = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        disMat[i][i] = hDis[i] - hDis[i + 1]
    return disMat


def calcStandDev(probDistTuple):
    '''takes the tuple of the calcProbDistMiddle method "probDis" (the probability distribution for a given time evolution of a hamiltonian matrix) \n
        returns the standard deviation of the distribution'''
    lenthOfState = len(probDistTuple[1])
    valueOne = 0
    valueTwo = 0

    #computes the two values in the square root of the equation for the deviation
    for i in range(lenthOfState):
        valueOne += probDistTuple[1][i] * (i + 1) ** 2
        valueTwo += probDistTuple[1][i] * (i + 1)
    sigma = math.sqrt(valueOne - valueTwo ** 2)
    return sigma


def calcLocalLen(matrix, W):
    '''takes the hamiltonian matrix without the disorder and the disorder parameter: "matrix", "W" \n
        returns the localization length'''
    N = len(matrix) + 1
    disMat = calcDisorder(N, W)
    hamiltonian = matrix + disMat
    hamTimeEvo = calcHamTimeEvo(hamiltonian, 100000)
    probDis = calcProbDistMiddle(hamTimeEvo) #mal schauen ob die Länge der Simulation eine passende Größenordnung für die Zeit hat, um den zeitl. Limes darzustellen, ohne Probleme mit dem Rand der Simulation zu erzeugen (falls es noch welche gibt)
    localLen = calcStandDev(probDis)
    return localLen


def calcStandDevTimeEvo(startTime, endTime, incrementTime, h3matrix, W, numberCalc):
    '''takes a start and ending time and an increment for the time: "startTime", "endTime", "incrementTime"\n
        takes a hamiltonian matrix and the disorder parameter: "h3matrix", "W" \n
        takes the number of calculation steps for averaging the standard deviation \n
        returns a tuple of the time steps and the respecting standard deviations'''
    timeList = np.arange(startTime, endTime + incrementTime, incrementTime)
    matrix = calcDisorder(len(h3matrix) + 1, W) + h3matrix
    hamTimeEvoInitial = calcHamTimeEvo(matrix, startTime)
    hamTimeStep = calcHamTimeEvo(matrix, incrementTime)

    startSigmaList = []

    for i in range(len(timeList)):
        # probability distributions for the matrices
        probDist = calcProbDistMiddle(hamTimeEvoInitial)
        # standard deviations for the probability distributions
        sigma = calcStandDev(probDist)
        startSigmaList.append(sigma)
        hamTimeEvoInitial = np.matmul(hamTimeEvoInitial,hamTimeStep)

    sigmaSumArray = np.asarray(startSigmaList)

    for j in range(numberCalc - 1):
        matrix = calcDisorder(len(h3matrix) + 1, W) + h3matrix
        hamTimeEvoInitial = calcHamTimeEvo(matrix, startTime)
        hamTimeStep = calcHamTimeEvo(matrix, incrementTime)
        sigmaList = []
        for i in range(len(timeList)):
            # probability distributions for the matrices
            probDist = calcProbDistMiddle(hamTimeEvoInitial)
            # standard deviations for the probability distributions
            sigma = calcStandDev(probDist)
            sigmaList.append(sigma)
            hamTimeEvoInitial = np.matmul(hamTimeEvoInitial, hamTimeStep)

        sigmaSumArray += np.asarray(sigmaList)
    sigmaSumArray = sigmaSumArray * (1 / numberCalc)

    result = []
    result.append(timeList)
    result.append(sigmaSumArray)
    return result


def calcProbSumList(startTime, endTime, incrementTime, matrix):
    '''exists to test the crediblity of the probability distribution \n
        takes a start and ending time and an increment for the time: "startTime", "endTime", "incrementTime"\n
        takes a hamiltonian matrix: "matrix" \n
        returns a tuple of the time steps and the sums of the respecting probability distributions \n
        (the list of the sums should up to a point of time only contain ones, for big enough time values the boundaries of the chain should be reached and values apart from one could appear'''
    timeList = np.arange(startTime, endTime + incrementTime, incrementTime)
    probSumList = []
    hamTimeEvoInitial = calcHamTimeEvo(matrix, startTime)
    hamTimeStep = calcHamTimeEvo(matrix, incrementTime)

    for i in range(len(timeList)):
        #probability distributions for the matrices
        probDist = calcProbDistMiddle(hamTimeEvoInitial)
        #sum of the probability distributions
        probSum = sum(probDist[1])
        probSumList.append(probSum)
        hamTimeEvoInitial = np.matmul(hamTimeEvoInitial, hamTimeStep)

    result = []
    result.append(timeList)
    result.append(probSumList)
    return result


def calcStandDevDisEvo(startDis, endDis, incrementDis, matrix, time, numberCalc):
    '''takes a start and end value and an increment for the disorder parameter: "startDis", "endDis", "incrementDis" \n
        takes the time of the state and the hamiltonian matrix without the disorder: "time, "matrix" \n
        return a tuple of the list of disorders and the list of the respecting standard deviation at time "time"'''
    N = len(matrix) + 1
    wList = np.arange(startDis, endDis + incrementDis, incrementDis)
    startSigmaDisList = []

    for i in wList:
        #hamiltonian matrix with disorder
        disMat = calcDisorder(N,i) + matrix
        disMatExp = calcHamTimeEvo(disMat, time)
        #probability distribution of the hamiltonian
        probDist = calcProbDistMiddle(disMatExp)
        #standard deviation of the hamiltonian
        sigmaDis = calcStandDev(probDist)
        startSigmaDisList.append(sigmaDis)

    sigmaDisSumArray = np.asarray(startSigmaDisList)

    for j in range(numberCalc - 1):
        sigmaDisList = []
        for i in wList:
            # hamiltonian matrix with disorder
            disMat = calcDisorder(N, i) + matrix
            disMatExp = calcHamTimeEvo(disMat, time)
            # probability distribution of the hamiltonian
            probDist = calcProbDistMiddle(disMatExp)
            # standard deviation of the hamiltonian
            sigmaDis = calcStandDev(probDist)
            sigmaDisList.append(sigmaDis)
        sigmaDisSumArray += np.asarray(sigmaDisList)

    sigmaDisSumArray = sigmaDisSumArray * (1 / numberCalc)

    result = []
    result.append(wList)
    result.append(sigmaDisSumArray)
    return result


def calcLocalLenDisEvo(startDis, endDis, incrementDis, matrix, numberCalc):
    '''takes a start and end value and an increment for the disorder parameter: "startDis", "endDis", "incrementDis" \n
        takes the hamiltonian matrix without the disorder: "matrix" \n
        takes the number of calculation steps for averaging the localization length \n
        return a tuple of the list of disorders and the list of the respecting localization length of the hamiltonian "matrix"'''
    wList = np.arange(startDis, endDis + incrementDis, incrementDis)
    startLocalLenDisList = []

    for i in wList:
        startLocalLenDisList.append(calcLocalLen(matrix,i))

    localLenSumArray = np.asarray(startLocalLenDisList)

    for j in range(numberCalc - 1):
        localLenDisList = []

        for k in wList:
            localLenDisList.append(calcLocalLen(matrix, k))

        localLenSumArray += np.asarray(localLenDisList)

    localLenSumArray = localLenSumArray * (1 / numberCalc)

    result = []
    result.append(wList)
    result.append(localLenSumArray)
    return result


def calcEigenstates(matrix):
    '''takes a hamiltonian matrix: "matrix" \n
        returns a tuple of a list of the respecting eigenvalues and a matrix with the eigenvectors as the columns in the same order \n
        (it removes the zero eigenvalue and the eigenvector which are created by the np.linalg.eigh() function if there is one)'''
    eigenvalues, eigenvectorMatrix = np.linalg.eigh(matrix)
    for i in range(len(eigenvalues)):
        if eigenvalues[i] == 0:
            eigVecMatNew = []
            eigValNew = np.delete(eigenvalues, i)
            for h in range(len(eigenvectorMatrix)):
                eigVecRow = np.delete(eigenvectorMatrix[h], i)
                eigVecMatNew.append(eigVecRow)
            result = []
            result.append(eigValNew)
            result.append(eigVecMatNew)
            return result
    result = []
    result.append(eigenvalues)
    result.append(eigenvectorMatrix)
    return result


def calcEigenstatesAverage(N, W, numberCalc):
    '''funktioniert nicht'''
    h3 = calcH3MatPerBound(N + 1)
    h3Dis = calcDisorder(N + 1, W)
    matrix = h3 + h3Dis
    eigenstates = np.linalg.eigh(matrix)
    eigenvaluesSum = np.asarray(eigenstates[0])
    eigenstatesMatrix = eigenstates[1]

    for i in range(numberCalc - 1):
        h3Dis = calcDisorder(N + 1, W)
        matrix = h3 + h3Dis
        eigenstates = np.linalg.eigh(matrix)
        eigenvaluesSum += np.asarray(eigenstates[0])
        for j in range(len(eigenstatesMatrix)):
            for k in range(len(eigenstatesMatrix[0])):
                eigenstatesMatrix[j][k] = eigenstatesMatrix[j][k] + eigenstates[1][j][k]
    eigenvaluesSum = eigenvaluesSum * (1/numberCalc)
    eigenstatesMatrix = eigenstatesMatrix * (1/numberCalc)
    result = []
    result.append(eigenvaluesSum)
    result.append(eigenstatesMatrix)
    return result



def calcEigVecDist(eigenvector):
    '''takes an eigenvector
    returns a tuple with the positions n and the distribution of the eigenvector'''
    n = list(range(1, len(eigenvector) + 1))
    distList = []
    for i in n:
        dipoleRI = calcDipole(len(eigenvector)+1, i, False)
        skalProd = dipoleRI.dot(eigenvector)
        absVal = skalProd.real ** 2 + skalProd.imag ** 2
        distList.append(absVal)
    realDistList = []
    for i in range(len(distList)):
        realDistList.append(distList[i].item(0))
    result = []
    result.append(n)
    result.append(realDistList)
    return result


def returnEigVec(tuple, index):
    '''takes the tuple of the lists of eigenvalues und eigenvectors and the index of the eigenvector
        returns the respecting eigenvector'''
    return tuple[1][:, index - 1]


def calcVarianceLongTime(matrix, W):
    '''takes a hamiltonian matrix and the disorder parameter W
        returns the long time variance'''
    N = len(matrix)
    disMat = calcDisorder(N + 1, W)
    hamiltonian = matrix + disMat
    eigenvalues, eigenstatesList = np.linalg.eigh(hamiltonian)
    halfN = math.ceil(N / 2)


    # creates the list of distances between the inital and referential dipole
    if N % 2 == 0:
        diff = list(range(-halfN + 1, halfN + 1))
    else:
        diff = list(range(-halfN + 1, halfN))

    Var = 0
    dipoleZeroRow = calcDipole(N + 1, halfN, False)
    eigenstatesNormed = []

    for i in range(N):
        eigenstateColumn = np.zeros((N, 1), complex)
        for j in range(N):
            eigenstateColumn[j][0] = eigenstatesList[j][i]
            eigenstatesNormed.append(eigenstateColumn)

    for n in diff:
        eigenstateSum = 0
        dipoleNRow = calcDipole(N + 1, n + halfN, False)
        for l in range(N):
            eigenstateSum += abs((dipoleZeroRow.dot(eigenstatesNormed[l]))[0][0])**2 * abs((dipoleNRow.dot(eigenstatesNormed[l]))[0][0])**2
        Var += n**2 * eigenstateSum

    return Var


def calcVarianceDisEvo(startDis, endDis, incrementDis, matrix, numberCalc):
    '''takes a start and end value and an increment for the disorder parameter: "startDis", "endDis", "incrementDis" \n
        takes the hamiltonian matrix without the disorder: "matrix" \n
        takes the number of calculation steps for averaging the variance \n
        return a tuple of the list of disorders and the list of the respecting long time limes of the variance of the hamiltonian "matrix" using its eigenstates'''
    wList = np.arange(startDis, endDis + incrementDis, incrementDis)
    startVarianceList = []
    for i in wList:
        startVarianceList.append(calcVarianceLongTime(matrix, i))

    varianceSumArray = np.asarray(startVarianceList)

    for j in range(numberCalc - 1):
        varianceDisList = []

        for i in wList:
            varianceDisList.append(calcVarianceLongTime(matrix,i))

        varianceSumArray += np.asarray(varianceDisList)

    varianceSumArray = varianceSumArray * (1 / numberCalc)


    result = []
    result.append(wList)
    result.append(varianceSumArray)
    return result


def calcLogYValue(tuple):
    logYValueList = []
    for i in tuple[1]:
        logYValueList.append(math.log(i, math.exp(1)))
    result = []
    result.append(tuple[0])
    result.append(logYValueList)
    return result


def powerLaw(x, a, k, c):
    return a*np.power(x, k) + c

def expFkt(x, a, p, l):
    return a * np.exp(-(x-p)/l)


def returnPeakValues(eigVecDist, halfWidth):
    peak = max(eigVecDist[1])
    peakPos = eigVecDist[1].index(peak)
    peakMiddleDiff = peakPos - round(len(eigVecDist[1])/2)
    xList = list(range(peakPos + 1 - halfWidth, peakPos + 2 + halfWidth))
    peakMiddleList = eigVecDist[1][peakMiddleDiff:] + eigVecDist[1][:peakMiddleDiff]
    newPeakPos = peakMiddleList.index(peak)
    yList = []
    for i in range(len(xList)):
        yList.append(peakMiddleList[newPeakPos - halfWidth + i])
    result = []
    result.append(np.asarray(xList))
    result.append(np.asarray(yList))
    return result


def calcEigPeakAverage(N, W, numberCalc, index, halfWidth):
    h3 = calcH3MatPerBound(N + 1)
    h3Dis = calcDisorder(N + 1, W)
    matrix = h3 + h3Dis
    eigenvaluesList = []
    eigenstates = np.linalg.eigh(matrix)
    indexEigenvector = returnEigVec(eigenstates, index)
    eigenvaluesList.append(eigenstates[0][index])
    indexEigvecDist = calcEigVecDist(indexEigenvector)
    peakSum = np.asarray(returnPeakValues(indexEigvecDist[1], halfWidth)

    for i in range(numberCalc):
        h3Dis = calcDisorder(N + 1, W)
        matrix = h3 + h3Dis
        eigenstates = np.linalg.eigh(matrix)
        indexEigenvector = returnEigVec(eigenstates, index)
        eigenvaluesList.append(eigenstates[0][index])
        indexEigvecDist = calcEigVecDist(indexEigenvector)
        peakSum += np.asarray(returnPeakValues(indexEigvecDist[1], halfWidth))

    peakSum = peakSum * (1/numberCalc)
    result = []
    result.append(eigenvaluesList)
    result.append(peakSum)
    return result


def calcStandDevSizeEvo(startSize, endSize, incrementSize, time, W, numberCalc):
    sizeList =  np.arange(startSize, endSize + incrementSize, incrementSize)
    startSigmaList = []
    for i in sizeList:
        h3Matrix = calcH3MatPerBound(i + 1)
        h3DisMatrix = calcDisorder(i + 1, W)
        matrix = h3Matrix + h3DisMatrix
        h3Time = calcHamTimeEvo(matrix, time)
        probDist = calcProbDistMiddle(h3Time)
        sigma = calcStandDev(probDist)
        startSigmaList.append(sigma)

    sigmaSumArray = np.asarray(startSigmaList)

    for j in range(numberCalc - 1):
        sigmaList = []
        for i in sizeList:
            h3Matrix = calcH3MatPerBound(i + 1)
            h3DisMatrix = calcDisorder(i + 1, W)
            matrix = h3Matrix + h3DisMatrix
            h3Time = calcHamTimeEvo(matrix, time)
            probDist = calcProbDistMiddle(h3Time)
            sigma = calcStandDev(probDist)
            sigmaList.append(sigma)
        sigmaSumArray += np.asarray(sigmaList)

    sigmaSumArray = sigmaSumArray * (1 / numberCalc)

    result = []
    result.append(sizeList)
    result.append(sigmaSumArray)
    return result


h3Mat = calcH3MatPerBound(N)  # Hamilton matrix
h3Exp = calcHamTimeEvo(h3Mat, time)  # matrix exponential vom Hamilton für wahrscheinlichkeitsverteilung

disMat = calcDisorder(N, W)  # Störungsmatrix

h3DisMat = h3Mat + disMat  # kombinierte matrix von hamiltonian und störung
h3DisExp = calcHamTimeEvo(h3DisMat, time)  # matrixexponential von hamiltonian mit störung

#VarDis = calcVarianceLongTime(h3DisMat, 3) #Varianz über Eigenzustände von der disorder matrx
#VarH3 = calcVarianceLongTime(h3Mat, 0) #Varianz über Eigenzustände von h3

if bProbDist:
    h3ProbDist = calcProbDistMiddle(h3Exp)  # wahrscheinlichkeitsverteilung für h3 (list von 2 listen)
    h3DisProbDist = calcProbDistMiddle(h3DisExp)  # wahrscheinlichkeitsverteilung für h3 mit störung

    #plot for probability distribution
    figProbDistClean = plt.figure(figsize=(12,6))
    subProbDistClean = figProbDistClean.add_subplot(2,2,1)
    plotDistEnh(subProbDistClean,h3ProbDist,"n - m","p(x)","probability distribution \n N = " + str(N) + ", t = " + str(time))
    #plt.savefig('probabilityDistribution.png', bbox_inches='tight', dpi=150)
    figProbDistDis = plt.figure()
    subProbDistDis = figProbDistDis.add_subplot(2,2,1)
    plotDistEnh(subProbDistDis,h3DisProbDist,"n - m","p(x)","probability distribution \n with disorder \n N = " + str(N) + ", W = " + str(W) + ", t = " + str(time))


if bProbDistTimeEvo:
    tPeriod = 4
    tIncrement = 0.1
    tList = np.arange(0, tPeriod + 1, tIncrement)
    startMat = calcHamTimeEvo(h3DisMat, 0)
    timeMat = calcHamTimeEvo(h3DisMat, tIncrement)
    zList = []
    for i in tList:
        probDist = calcProbDistMiddle(startMat)
        zList.append(probDist[1])
        startMat = np.matmul(startMat, timeMat)
        xValues = probDist[0]
    z = np.asarray(zList)
    x, y = np.meshgrid(xValues, tList)
    figColour = plt.figure()
    plt.contourf(x, y, z)
    plt.xlabel('n - m')
    plt.ylabel("time")
    #plt.title('time evolution of \n the probability distribution')
    plt.colorbar()
    #plt.legend("W = " + str(W), loc="upper left")
    figColour.savefig("probDistColourWithout_structure.pdf", bbox_inches='tight', dpi=150)
    plt.show()


if bStandDevTimeEvo:
    h3Sigma = calcStandDevTimeEvo(0, 60, 2, h3Mat, 3000, 1) #list of standard deviations for the h3 matrix
    h3DisSigmaW2 = calcStandDevTimeEvo(0, 300, 2, h3Mat, 1, 5) #list of standard deviations for the h3 matrix with the random disorder
    h3DisSigmaW1 = calcStandDevTimeEvo(0, 300, 2, h3Mat, 0.5, 5)

    figSigma = plt.figure()
    plt.plot(h3Sigma[0], h3Sigma[1])
    plt.xlabel("time")
    plt.ylabel(r"$\sigma$")
    #sub1Sigma = figSigma.add_subplot(2, 2, 1)
    #plotDistEnh(sub1Sigma, h3Sigma, "time", "sigma", "time evolution \n of the standard deviation \n N = " + str(N))
    #sub2Sigma = figSigma.add_subplot(2, 2, 2)
    #plotDistEnh(sub2Sigma,h3DisSigma, "time", "sigma","time evolution \n of the standard deviation \n with disorder \n N = " + str(N))
    #plotDist(sub1Sigma, h3Sigma, "time", r"$\sigma$", "time evolution \n of the standard deviation \n for N =" + str(N), "W = 0")
    #sub1Sigma.plot(h3DisSigmaW2[0], h3DisSigmaW2[1], label='W = 1')
    #sub1Sigma.plot(h3DisSigmaW1[0], h3DisSigmaW1[1], label='W = 0.2')
    #sub1Sigma.legend(loc='upper left')
    #figSigma.savefig('standDevTimeEvo.png', bbox_inches='tight', dpi=150)
    plt.savefig("standDevTimeEvoInfinite_structure.pdf", bbox_inches='tight', dpi=150)
    plt.show()


if bStandDevSizeEvo:
    h3SigmaSizeEvo = calcStandDevSizeEvo(200, 400, 1, 30000, 0, 5)
    figSigmaSizeEvo = plt.figure()
    #sub1SigmaSizeEvo = figSigmaSizeEvo.add_subplot(2,2,1)
    #plotDistEnh(sub1SigmaSizeEvo, h3SigmaSizeEvo, "size of the system", r"$\sigma$", "size evolution \n of the standard deviation \n")
    plt.plot(h3SigmaSizeEvo[0], h3SigmaSizeEvo[1])
    plt.xlabel("N")
    plt.ylabel(r"$\sigma$")
    plt.savefig("standDevSizeEvoWithout2_structure.pdf", bbox_inches='tight', dpi=150)
    plt.show()


if bProbDistSum:
    h3Sum = calcProbSumList(0, 70, 5, h3Mat)
    h3DisSum = calcProbSumList(0, 70, 5, h3DisMat)

    #plot of the time evolution of the sum of the probability distribution
    figSum = plt.figure()
    sub1Sum = figSum.add_subplot(2, 2, 1)
    plotDistEnh(sub1Sum,h3Sum, "time", r"$\Sigma_{n - m}$ p(n - m)","sum of the probability distribution\n N = " + str(N))
    sub2Sum = figSum.add_subplot(2, 2, 2)
    plotDistEnh(sub2Sum,h3DisSum, "time", r"$\Sigma_{n - m}$ p(n - m)", "sum of the probability distribution \n with disorder \n N = " + str(N) + " W = " + str(W))


if bStandDevDisEvo:
    h3SigmaDisEvo = calcStandDevDisEvo(0,0.5,0.01,h3Mat,5000, 10)
    #h3SigmaDisEvoTime = calcStandDevDisEvo(0,1,0.01,h3Mat,10000, 10)

    figSigmaDisEvo = plt.figure()
    sub1SigmaDisEvo = figSigmaDisEvo.add_subplot(2,2,1)
    plotDistEnh(sub1SigmaDisEvo, h3SigmaDisEvo, "W", r"$\sigma$", "disorder evolution \n of the standard deviation \n N = " + str(N) + " time = " + str(5000))
    sub2SigmaDisEvo = figSigmaDisEvo.add_subplot(2, 2, 2)
    plotDistEnh(sub2SigmaDisEvo, h3SigmaDisEvo, "W", r"$\sigma$", "disorder evolution \n of the standard deviation \n with logarithmic axis")
    sub2SigmaDisEvo.set_xscale('log')
    sub2SigmaDisEvo.set_yscale('log')


if bLocalLenDisEvo:
    h3LocalLenDisEvo = calcLocalLenDisEvo(0,0.5,0.01,h3Mat,10)

    figLocalLenDisEvo = plt.figure()
    sub1LocalLenDisEvo = figLocalLenDisEvo.add_subplot(2, 2, 1)
    plotDistEnh(sub1LocalLenDisEvo, h3LocalLenDisEvo, "W", r"$\zeta_{loc}$", "disorder evolution \n of the localization length \n N = " + str(N))

    h3VarianceList = []
    for i in range(len(h3LocalLenDisEvo[1])):
        varianceValue = h3LocalLenDisEvo[1][i]**2
        h3VarianceList.append(varianceValue)

    h3VarianceDisEvo = []
    h3VarianceDisEvo.append(h3LocalLenDisEvo[0])
    h3VarianceDisEvo.append(h3VarianceList)

    sub2LocalLenDisEvo = figLocalLenDisEvo.add_subplot(2, 2, 2)
    plotDistEnh(sub2LocalLenDisEvo, h3VarianceDisEvo, "W", r"$\zeta_{loc}^2$", "disorder evolution \n of the variance \n N = " + str(N))

    #fitParams, pcov = scipy.optimize.curve_fit(powerLaw, h3LocalLenDisEvo[0], h3LocalLenDisEvo[1])
    #yFit = powerLaw(h3LocalLenDisEvo[0], fitParams[0], fitParams[1], fitParams[2])
    #print(fitParams)
    #print(yFit)

    sub3LocalLenDisEvo = figLocalLenDisEvo.add_subplot(2, 2, 3)
    plotDistEnh(sub3LocalLenDisEvo, h3LocalLenDisEvo, "W", r"$\log(\zeta_{loc})$", "natural logarithm \n of the localization length")
    #sub3LocalLenDisEvo.plot(h3LocalLenDisEvo[0], yFit)
    sub3LocalLenDisEvo.set_yscale('log')
    sub3LocalLenDisEvo.set_xscale('log')


if bEigStateDist:
    h3DisEigStates = calcEigenstates(h3DisMat)
    h3DisEigStates2 = calcEigenstates(h3DisMat)
    h3DisEigVec1Dist = calcEigVecDist(h3DisEigStates[1][:,0])
    h3DisEigVec2Dist = calcEigVecDist(h3DisEigStates2[1][:,0])
    #print(h3DisEigVec1Dist[0])
    #print(h3DisEigVec1Dist[1])
    peakValues = returnPeakValues(h3DisEigVec1Dist, 20)
    #print(peakValues[0])
    #print(peakValues[1])
    fitParams, pcov = scipy.optimize.curve_fit(expFkt, peakValues[0], peakValues[1])
    print(fitParams)
    h3DisEigStatesFit_yValues = expFkt(np.asarray(peakValues[0]), *fitParams)
    #print(h3DisEigStatesFit_yValues)
    print(len(peakValues[0]))
    print(len(h3DisEigStatesFit_yValues))

    peakAverage = calcEigPeakAverage(N, W, 5, 0, 20)
    xList = list(range(len(peakAverage[1])))

    figEigStateDist = plt.figure()
    sub1EigStateDist = figEigStateDist.add_subplot(221)
    plotDistEnh(sub1EigStateDist, h3DisEigVec1Dist, "n", "|phi(n)|^2", "distribution of the eigenvector \n W = " + str(W) + "\n E = " + str(h3DisEigStates[0][0]))
    sub2EigStateDist = figEigStateDist.add_subplot(222)
    plotDistEnh(sub2EigStateDist, h3DisEigVec2Dist, "n", "|phi(n)|^2", "distribution of the eigenvector \n W = " + str(W) + "\n E = " + str(h3DisEigStates[0][1]))
    sub3EigStateDist = figEigStateDist.add_subplot(223)
    sub3EigStateDist.set_yscale('log')
    plotDistEnh(sub3EigStateDist, h3DisEigVec1Dist, "n", "|phi(n)|^2", "log of the above")
    sub4EigStateDist = figEigStateDist.add_subplot(224)
    sub4EigStateDist.set_yscale('log')
    plotDistEnh(sub4EigStateDist, h3DisEigVec2Dist, "n", "|phi(n)|^2", "log of the above")
    peakFig = plt.figure()
    plt.plot(peakValues[0], peakValues[1])
    plt.yscale('log')
    #fitFig = plt.figure()
    plt.scatter(peakValues[0], h3DisEigStatesFit_yValues)
    plt.scatter(xList, peakAverage[1])


if bVarianceDisEvo:
    h3VariancePlotList = calcVarianceDisEvo(0, 3, 0.1, h3Mat, 100)

    figVarianceDisEvo = plt.figure()
    sub1VarianceDisEvo = figVarianceDisEvo.add_subplot(2, 2, 1)
    #sub1VarianceDisEvo.set_yscale('log')
    plotDistEnh(sub1VarianceDisEvo, h3VariancePlotList, "W", r"$\zeta_{loc}^2$","disorder evolution \n of the time limes variance \n N = " + str(N))


plt.show()

#distribution of the eigenvector \n of H3 with W = " + str(W) + "\n with the eigenvalue \n E = " + str(h3DisEigStates[0][0])


# test zum überprüfen, ob die position der dipole eine rolle spielt
dipoleR3 = calcDipole(N, 3, True)
dipoleR3Time = h3Exp.dot(dipoleR3)
dipoleL2 = calcDipole(N, 2, False)
dipoleR2 = calcDipole(N, 2, True)
dipoleR2Time = h3Exp.dot(dipoleR2)
dipoleL1 = calcDipole(N, 1, False)
versuch1 = dipoleL2.dot(dipoleR3Time)
versuch2 = dipoleL1.dot(dipoleR2Time)


#test1 = calcEigenstates(h3DisMat)
#test2 = calcEigenstatesAverage(100, 1, 30)
#test3 = calcEigenstatesAverage(100, 1, 30)
#print(test1)
#print(test2)
#print(test3)
