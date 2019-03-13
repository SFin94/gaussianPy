import sys
import numpy as np
import matplotlib.pyplot as plt


def parseInfo(fileName, freqGoal):
    energyOpt = []
    convStepCount = 0
    freqCount = 0
    with open(fileName,'r') as input:
        for el in input:
            if 'SCF Done' in el:
                print(el)
                energyOpt.append(float(el.split('=')[1].split()[0]))

            if 'Converged?' in el:
                convStepCount += 1
                [print(input.__next__()) for x in range(4)]

            if (('Frequencies' in el) & (freqCount < freqGoal)):
                print(el)
                freqCount += 1
            if (('Zero-point' in el)|('Thermal correction' in el)):
                print(el)

        print('Final Energy: ' + str(energyOpt[-1]))
        print('Steps taken to optimise: ' + str(convStepCount))
    return(energyOpt)


def plotTraj(energyTraj):

    plt.figure(figsize=(8,7))
    stepsTaken = list(range(0, len(energyTraj)))
    plt.plot(stepsTaken, energyTraj)
    plt.xlabel('Step number')
    plt.ylabel('Energy (h)')
    plt.show()


if __name__ == '__main__':

    numFreqs = int(sys.argv[2])
    energyTraj = parseInfo(str(sys.argv[1]), numFreqs)
    plotTraj(energyTraj)


