# Matrix exponential for the time development of an initial dipole
import numpy as np
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt
from random import *
import math


N = 100
'''length of state'''
time = 5
W = 1.5
'''disorder paramter'''


def calcH3MatOpenBound(N):
    '''creates h3 for a given amount of positions "N"'''
    h3Mat = np.zeros((N - 1, N - 1), dtype=np.complex)

    #put ones on the off-diagonals
    for k in range(N - 2):
        h3Mat[k][k + 1] = -1
        h3Mat[k + 1][k] = -1

    return h3Mat


def calcH3MatPerBound(N):
    '''unnecessary function which doesn't work'''
    h3Mat = np.zeros((N - 1, N - 1), dtype=np.complex)

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


# creation of dipoles
def calcDipole(N, position, column):
    '''creates a vector with the lenght "N"-1 with a one at "position", for "column" == True it is a column vector, for "column" == False it is a row vector (dipole at position "position")'''

    if column:
        dipoleR = np.zeros((N - 1, 1))
        dipoleR[position - 1][0] = 1
        return dipoleR

    else:
        dipoleL = np.zeros((1, N - 1))
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
def calcProbDistMiddle(matrix):  # bestimmt die wahrscheinlichkeitsverteilung für eine Zeitentwicklung des Dipols -> matrix == matrix exponential des Hamiltonian
    '''takes the time evolution of a hamiltonian matrix \n
     returns a tupel of firstly the distance between an initial and a referential dipole and secondly the respecting probability distribution'''

    N = len(matrix)
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
    dipoleInitTime = matrix.dot(dipoleInit)

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


def plotDist(tupleOfListsOfValues, xlabel, ylabel):
    '''takes a tuple of lists with x and y values: "tupleOfListsOfValues'''
    plt.plot(tupleOfListsOfValues[0], tupleOfListsOfValues[1], "r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plotDistEnh(plotName, tupleOfListsOfValues, xlabel, ylabel, title):
    '''takes a tuple of lists with x and y values: "tupleOfListsOfValues'''
    plotName.plot(tupleOfListsOfValues[0], tupleOfListsOfValues[1], "r")
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


def calcStandDev(probDist):
    '''takes the tuple of the calcProbDistMiddle method "probDis" (the probability distribution for a given time evolution of a hamiltonian matrix) \n
    returns the standard deviation of the distribution'''
    lenthOfState = len(probDist[1])
    valueOne = 0
    valueTwo = 0

    #computes the two values in the square root of the equation for the deviation
    for i in range(lenthOfState):
        valueOne += probDist[1][i] * (i + 1) ** 2
        valueTwo += probDist[1][i] * (i + 1)
    sigma = math.sqrt(valueOne - valueTwo ** 2)
    return sigma


def calcStandDevList(startTime, endTime, incrementTime, matrix):
    '''takes a start and ending time, the increment for the time and a hamiltonian matrix \n
    returns a tuple of the time steps and the respecting standard deviations'''
    timeList = range(startTime, endTime + 1, incrementTime)

    #creates a list of time evolved hamiltonian matrices
    matExpList = []
    for i in timeList:
        matExpList.append(calcHamTimeEvo(matrix, i))

    #creates a list of probability distributions for the matrices
    probDisList = []
    for i in matExpList:
        probDisList.append(calcProbDistMiddle(i))

    #creates a list of standard deviations for the probability distributions
    sigmaList = []
    for i in probDisList:
        sigmaList.append(calcStandDev(i))

    result = []
    result.append(timeList)
    result.append(sigmaList)
    return result


def calcProbSumList(startTime, endTime, incrementTime, matrix):
    '''exists to test the crediblity of the probability distribution \n
    takes a start and ending time, the increment for the time and a hamiltonian matrix \n
    returns a tuple of the time steps and the sums of the respecting probability distributions \n
    (the list of the sums should up to a point of time only contain ones, for big enough time values the boundaries of the chain should be reached and values apart from one could appear'''
    timeList = range(startTime, endTime + 1, incrementTime)

    #creates a list of time evolved hamiltonian matrices
    matExpList = []
    for i in timeList:
        matExpList.append(calcHamTimeEvo(matrix, i))

    # creates a list of probability distributions for the matrices
    probDisList = []
    for i in matExpList:
        probDisList.append(calcProbDistMiddle(i))

    #creates a list of the sums of the probability distributions
    probSumList = []
    for i in probDisList:
        k = 0
        for j in i[1]:
            k += j
        probSumList.append(k)

    result = []
    result.append(timeList)
    result.append(probSumList)
    return result

def calcStandDevDisEvo(startDis, endDis, incrementDis, time, matrix):
    '''takes a start and end value and an increment for the disorder parameter: "startDis", "endDis", "incrementDis" \n
    takes the time of the state and the hamintonian matrix without the disorder: "time, "matrix"
    return a tuple of the list of disorders and the list of the respecting standard deviation at time "time"'''
    N = len(matrix) + 1
    wList = range(startDis, endDis + 1, incrementDis)

    #creates a list of hamiltonian matrices with disorder
    matDisList = []
    for i in wList:
        disMat = calcDisorder(N,i) + matrixTimeEvo
        matDisList.append(disMat)

    #creates a list of the time evolutions of the hamiltonian matrices
    matDisTimeEvoList = []
    for i in matDisList:
        matDisTimeEvoList.append(calcHamTimeEvo(i, time))

    #creates a list of the probability distributions of the hamiltonians
    probDistList = []
    for i in matDisTimeEvoList:
        probDistList.append(calcProbDistMiddle(i))

    #creates a list of the standard deviations of the hamitonians
    sigmaDisList = []
    for i in probDistList:
        sigmaDisList.append(calcStandDev(i))

    result = []
    result.append(wList)
    result.append(sigmaDisList)
    return result


h3Mat = calcH3MatOpenBound(N)  # Hamilton matrix
h3Exp = calcHamTimeEvo(h3Mat, time)  # matrix exponential vom Hamilton für wahrscheinlichkeitsverteilung
h3ProbDist = calcProbDistMiddle(h3Exp)  # wahrscheinlichkeitsverteilung für h3 (list von 2 listen)
disMat = calcDisorder(N, W)  # Störungsmatrix
h3DisMat = h3Mat + disMat  # kombinierte matrix von hamiltonian und störung
h3DisExp = calcHamTimeEvo(h3DisMat, time)  # matrixexponential von hamiltonian mit störung
h3DisProbDist = calcProbDistMiddle(h3DisExp)  # wahrscheinlichkeitsverteilung für h3 mit störung

#plot for probability distribution
figProbDist = plt.figure()
sub1ProbDist = figProbDist.add_subplot(2,2,1)
plotDistEnh(sub1ProbDist,h3ProbDist,"n - m","p(x)","probability distribution \n N = " + str(N))
h3ProbDistText = ''''''
sub2ProbDist = figProbDist.add_subplot(2,2,2)
plotDistEnh(sub2ProbDist,h3DisProbDist,"n - m","p(x)","probability distribution \n with disorder \n N = " + str(N) + " W = " + str(W))
#plt.show()

sigmaH3 = calcStandDevList(0, 30, 1, h3Mat) #list of standard deviations for the h3 matrix
sigmaH3Dis = calcStandDevList(0, 30, 1, h3DisMat) #list of standard deviations for the h3 matrix with the random disorder

#plot for the time evolution of the standard deviation
figSigma = plt.figure()
sub1Sigma = figSigma.add_subplot(2, 2, 1)
#plotDist(sigmaH3, "time", "sigma")
plotDistEnh(sub1Sigma, sigmaH3, "time", "sigma", "time evolution \n of the standard deviation \n N = " + str(N))
sub2Sigma = figSigma.add_subplot(2, 2, 2)
plotDistEnh(sub2Sigma,sigmaH3Dis, "time", "sigma","time evolution \n of the standard deviation \n with disorder \n N = " + str(N) + " W = " + str(W))
#plt.show()

sumH3 = calcProbSumList(0, 100, 1, h3Mat)
sumH3Dis = calcProbSumList(0, 100, 1, h3DisMat)

#plot of the time evolution of the sum of the probability distribution
figSum = plt.figure()
sub1Sum = figSum.add_subplot(2, 2, 1)
plotDistEnh(sub1Sum,sumH3, "time", "sum","sum of the probability distribution\n N = " + str(N))
sub2Sum = figSum.add_subplot(2, 2, 2)
plotDistEnh(sub2Sum,sumH3Dis, "time", "sum", "sum of the probability distribution \n with disorder \n N = " + str(N) + " W = " + str(W))
plt.show()



k = 0
for i in h3ProbDist[1]:
    k += i
# print(h3ProbDist[1])
#print(k)  # summe der wahrscheinlichkeiten für h3
l = 0
for m in h3DisProbDist[1]:
    l += m
#print(l)  # summe der wahrscheinlichkeiten für h3 mit störung

# test zum überprüfen, ob die position der dipole eine rolle spielt
dipoleR3 = calcDipole(N, 3, True)
dipoleR3Time = h3Exp.dot(dipoleR3)
dipoleL2 = calcDipole(N, 2, False)
dipoleR2 = calcDipole(N, 2, True)
dipoleR2Time = h3Exp.dot(dipoleR2)
dipoleL1 = calcDipole(N, 1, False)
versuch1 = dipoleL2.dot(dipoleR3Time)
versuch2 = dipoleL1.dot(dipoleR2Time)
