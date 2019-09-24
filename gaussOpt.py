import sys
import numpy as np
import argparse
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
    for el in fileName:
        if 'SCF Done' in el:
            energyOpt.append(float(el.split('=')[1].split()[0]))

        if 'Converged?' in el:
            convStepCount += 1
            [print(fileName.__next__()) for x in range(4)]

        if (('Frequencies' in el) & (freqCount < freqGoal)):
            print(el)
            freqCount += 1
        if (('Zero-point' in el)|('Thermal correction' in el)):
            print(el)

    print('Final Energy: ' + str(energyOpt[-1]))
    print('Steps taken to optimise: ' + str(convStepCount))
    return(energyOpt)


def plotTraj(energyTraj):


    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Set figure and plot param(s) vs energy
    fig, ax = plt.subplots(figsize=(8,7))

    # Remove plot frame lines
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    stepsTaken = list(range(0, len(energyTraj)))
    ax.plot(stepsTaken, energyTraj, color='#0C739C', alpha=0.6)
    ax.set_xlabel('Step number')
    ax.set_ylabel('Energy (h)')
    plt.show()


if __name__ == '__main__':

    usage = "usage: %(prog)s [fileName] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    # Set parser arguments and process
    parser.add_argument("fileName", nargs=1, type=argparse.FileType('r+'),  help="Gaussian optimisatin log file name")
    parser.add_argument("-f", "--freqs", dest='freqGoal', type=int, default=0, help="Number of real (i.e. Low frequencies omitted) vibrational modes to display")
    parser.add_argument("-t", "--traj", dest='traj', type=bool, default=False,
                        help="Boolean flag which prints optimisation trajectory if True.")
    args = parser.parse_args()

    energyTraj = parseInfo(args.fileName[0], args.freqGoal)
    if args.traj == True:
        plotTraj(energyTraj)


