# Matrix exponential for the time development of an initial dipole
import numpy as np
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt
from random import *
import math

'''length of state'''
N = 200  # length of state
time = 10
W = 1.5

'''creates h3 for a given amount of positions N jo boi'''
def calcH3MatOpenBound(N):
    h3Mat = np.zeros((N - 1, N - 1), dtype=np.complex)
    for k in range(N - 2):  # the off-diagonals
        h3Mat[k][k + 1] = -1
        h3Mat[k + 1][k] = -1
    return h3Mat

def calcH3MatPerBound(N):  # ist unnötig bzw. funktioniert icht
    h3Mat = np.zeros((N - 1, N - 1), dtype=np.complex)
    for k in range(N - 2):
        h3Mat[k][k + 1] = -1
        h3Mat[k + 1][k] = -1
    h3Mat[N - 2][0] = -1
    h3Mat[0][N - 2] = -1
    return h3Mat


def calcHamTimeEvo(matrix, time):
    matExp = matrix * complex(0, -1) * time
    matExp = expm(matExp)
    return matExp


# creation of dipoles
def calcDipole(N, position, column):  # column vector for column == true and row vector for column == false
    if column:
        dipoleR = np.zeros((N - 1, 1))
        dipoleR[position - 1][0] = 1
        return dipoleR
    else:
        dipoleL = np.zeros((1, N - 1))
        dipoleL[0][position - 1] = 1
        return dipoleL


def calcProbDist(matrix):  # erster Versuch der Wahrscheinlichkeitsverteilung. funktioniert nicht so gut
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
def calcProbDistMiddle(
        matrix):  # bestimmt die wahrscheinlichkeitsverteilung für eine Zeitentwicklung des Dipols -> matrix == matrix exponential des Hamiltonian
    N = len(matrix)
    halfN = math.ceil(N / 2)
    if halfN % 2 == 0:
        diff = list(range(-halfN + 1, halfN + 1))
    else:
        diff = list(range(-halfN + 1, halfN))
    probDis = []
    dipoleInit = np.zeros((N, 1))
    dipoleInit[halfN - 1][0] = 1
    dipoleInitTime = matrix.dot(dipoleInit)
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


def plotDist(listWithListsWithValues, xlabel, ylabel):
    plt.plot(listWithListsWithValues[0], listWithListsWithValues[1], "r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# adding the disorder
def calcDisorder(N, disorderParameter):  # kreiert die additionale Matrix der Störung
    W = disorderParameter
    hDis = []
    for i in range(N):
        h = uniform(-W, W)
        hDis.append(h)
    disMat = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        disMat[i][i] = hDis[i] - hDis[i + 1]
    return disMat


def calcStandDev(probDis):
    lenthOfState = len(probDis[1])
    valueOne = 0
    valueTwo = 0
    for i in range(lenthOfState):
        valueOne += probDis[1][i] * (i + 1) ** 2
        valueTwo += probDis[1][i] * (i + 1)
    sigma = math.sqrt(valueOne - valueTwo ** 2)
    return sigma


def calcStandDevList(startTime, endTime, incrementTime, matrix):
    timeList = range(startTime, endTime + 1, incrementTime)
    matExpList = []
    for i in timeList:
        matExpList.append(calcHamTimeEvo(matrix, i))
    probDisList = []
    for i in matExpList:
        probDisList.append(calcProbDistMiddle(i))
    sigmaList = []
    for i in probDisList:
        sigmaList.append(calcStandDev(i))
    result = []
    result.append(timeList)
    result.append(sigmaList)
    return result


def calcProbSumList(startTime, endTime, incrementTime, matrix):
    timeList = range(startTime, endTime + 1, incrementTime)
    matExpList = []
    for i in timeList:
        matExpList.append(calcHamTimeEvo(matrix, i))
    probDisList = []
    for i in matExpList:
        probDisList.append(calcProbDistMiddle(i))
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


h3Mat = calcH3MatOpenBound(N)  # Hamilton matrix
h3Exp = calcHamTimeEvo(h3Mat, time)  # matrix exponential vom Hamilton für wahrscheinlichkeitsverteilung
h3ProbDist = calcProbDistMiddle(h3Exp)  # wahrscheinlichkeitsverteilung für h3 (list von 2 listen)
disMat = calcDisorder(N, W)  # Störungsmatrix
h3DisMat = h3Mat + disMat  # kombinierte matrix von hamiltonian und störung
h3DisExp = calcHamTimeEvo(h3DisMat, time)  # matrixexponential von hamiltonian mit störung
h3DisProbDist = calcProbDistMiddle(h3DisExp)  # wahrscheinlichkeitsverteilung für h3 mit störung

k = 0
for i in h3ProbDist[1]:
    k += i
# print(h3ProbDist[1])
print(k)  # summe der wahrscheinlichkeiten für h3
l = 0
for m in h3DisProbDist[1]:
    l += m
print(l)  # summe der wahrscheinlichkeiten für h3 mit störung

fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# plotDist(h3ProbDist,"n - m","p(x)")
# ax2 = fig.add_subplot(2,2,2)
# plotDist(h3DisProbDist,"n - m","p(x)")


sigmaH3 = calcStandDevList(0, 30, 1, h3Mat)
sigmaH3Dis = calcStandDevList(0, 30, 1, h3DisMat)

ax3 = fig.add_subplot(2, 2, 3)
plotDist(sigmaH3, "time", "sigma")
ax4 = fig.add_subplot(2, 2, 4)
plotDist(sigmaH3Dis, "time", "sigma")

sumH3 = calcProbSumList(0, 30, 1, h3Mat)
sumH3Dis = calcProbSumList(0, 30, 1, h3DisMat)

ax5 = fig.add_subplot(2, 2, 1)
plotDist(sumH3, "time", "sum")
ax6 = fig.add_subplot(2, 2, 2)
plotDist(sumH3Dis, "time", "sum")

plt.show()

# test zum überprüfen, ob die position der dipole eine rolle spielt
dipoleR3 = calcDipole(N, 3, True)
dipoleR3Time = h3Exp.dot(dipoleR3)
dipoleL2 = calcDipole(N, 2, False)
dipoleR2 = calcDipole(N, 2, True)
dipoleR2Time = h3Exp.dot(dipoleR2)
dipoleL1 = calcDipole(N, 1, False)
versuch1 = dipoleL2.dot(dipoleR3Time)
versuch2 = dipoleL1.dot(dipoleR2Time)
