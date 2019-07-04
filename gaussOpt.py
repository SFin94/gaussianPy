import sys
import numpy as np
import matplotlib.pyplot as plt


'''Script to print out the convergence information, opt trajectory and final energy for a job. Can optionally plot the optimisation trajectory if third argument included
    Usage: [gaussOpt.py] fileName freqeuncyNumber (t)

    Where:
        filename: name of the log file (str; with .log extension)
        freqeuncyNumber: number of lines (int) of freqeuncies to print out
        t: Optional flag to include at the end if plot of optimisation trajectory wanted
'''

def parseInfo(fileName, freqGoal=1):
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

    if len(sys.argv) > 2:
        energyTraj = parseInfo(str(sys.argv[1]), int(sys.argv[2]))
    else: energyTraj = parseInfo(str(sys.argv[1]))

    if len(sys.argv) > 3:
        if sys.argv[3] == 't':
            plotTraj(energyTraj)


