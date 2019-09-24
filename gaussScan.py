import matplotlib.pyplot as plt
import matplotlib.lines as mlin
import pandas as pd
import numpy as np
import gaussGeom as geom
import argparse

'''
    A python script which plots the PES from the results of a Gaussian relaxed scan.

    Usage: python gaussScan.py scan.log (trackedParam.txt)
    Where scan.log is the .log file of the relaxed scan result and trackedParam.txt is an optional input file.
     Input:
        scan.log args[0]: Str - File name of the log file of the scan results
        trackedParam.txt args[1]: (optional) Str - input file containing any additonal parameters to be tracked during the scan (see function below for format of file)
    Functions:
        findParam: Locates from the log file the scan parameters (atom indexes, number of steps, step size)
        trackParams: Parses in any additional parameters to be tracked across the scan
        scan: Calculates the relative energy and parameter values for each optimised point in the scan from the .log file
        scanPlot: Plots the scan parameter against the energy

'''

def findParam(inputFile):

    '''Function which extracts the scan parameter from a log files modRedundant input section

    Parameters:
     inputFile: Str - name of the input log file

    Returns:
     scanParam: Dict {'paramKey': str, 'atomInd': list of ints, 'nSteps': int, 'stepSize': float} - contains a parameter key of the modRed parameter type and atomIDs in log file of the scan parameter, atom indexes (python 0 index) of the scan parameter, the number of scan steps and the step size.
    '''

    # Sets up empty list for the modRedundant input, opens and reads file, extracting the modRedundant input
    modRedundant = []
    with open(inputFile, 'r') as logFile:
        for el in logFile:
            if 'The following ModRedundant input section has been read:' in el:
                el = logFile.__next__()
                # Extracts the ModRedundant section
                while el.strip() != '':
                    modRedundant.append(el.strip().split())
                    el = logFile.__next__()
                break

    # Types dictionary of the corresponding number of atom IDs required for each one
    types = {'X': 1, 'B': 2, 'A': 3, 'D': 4}

    # Iterates over the modRedundant inputs, finds the scan parameter and saves the input
    for mR in modRedundant:
        # Identifies number of atom IDs to expect and tests the action input for the scan parameter (assuming only one here, could have more)
        numAtoms = types[mR[0]]
        if mR[numAtoms+1] == 'S':
            scanParam = {'paramKey': mR[0], 'atomInd': [], 'nSteps': int(mR[-2]), 'stepSize': float(mR[-1])}
            # NB: Have to deduct one from atom ind for python 0 indexing, real indexes stored in paramKey
            for atomInd in mR[1:numAtoms+1]:
                scanParam['atomInd'].append(int(atomInd) - 1)
                scanParam['paramKey'] += (' ' + atomInd)
    try:
        return(scanParam)
    except NameError:
        print('No scan parameter located')
        raise


def parseTrackedParams(inputFile):

    '''Function which parses any additional parameters to be tracked from an input file

        Input:
         inputfile: str - name of input .txt file which contains any additional parameters to be tracked across the scan

         Format of input file:
             paramName (atomTypes) atomInd1 atomInd2 [atomInd3 atomInd4]
             E.g. OPSC 3 1 2 7

        Returns:
         trackedParams: Dict of {str: list of ints; paramName: [atomIndexes]} - atom indexes for any additional parameters to track across the scan
    '''

    # Initialise empty dict for params
    trackedParams = {}
    # Parse in file and seperate the indexes from the parameter ID and save as an entry to the dict
    with open(inputFile, 'r') as input:
        for el in input:
            param = el.strip().split(' ')
            indexes = [int(ind) for ind in param[1:]]
            trackedParams[param[0]] = indexes
    return(trackedParams)


def scan(scanParam, scanPoints, scanResults, inputFile, trackedInput):

    '''Function which parses the log file for the energy results from the scan for the scanned parameter and any other set tracked parameters

        Input:
         scanParam: list of ints - atom indexes (0 index) of the parameter being scanned
         scanPoints: int - number of scan points
         inputFile: str - name of log file (with .log extension) containing the scan results
         trackParams: (optional) Dict of str: list of ints (paramName: [atomIndexes]) - atom indexes for any additional parameters to track across the scan

        Returns:
         scanResults: numpy array (dim: scanPoints x 4 (+ number of tracked parameters)) - array holding the scan parameter value, energy (h), relative energy (kJ/mol) and the value of any tracked parameters for each optimized scan point
         optimised: List of bools - bool for each scanpoint showing if geometry is optimised (True) or not (False)
     '''
    # Set up empty dict to save new results in for each step
    stepResult = {col: None for col in scanResults.columns}

    # For each step in the scan, pulls optimised geometry and calculates parameter values, energy and whether the structure is optimised
    for scanStep in range(scanPoints + 1):
        stepGeom = geom.geomPulllog(inputFile, optStep = (scanStep+1))[0]
        stepResult[scanResults.columns[0]] = geom.paramGeom([scanParam], stepGeom)[0]
        for tParam, paramVal in trackedParams.items():
            stepResult[tParam] = geom.paramGeom([paramVal], stepGeom)[0]
        stepResult['E (h)'], stepResult['Optimised'] = geom.energyPull(inputFile, optStep = (scanStep+1))

        # Append step results to dataframe
        scanResults = scanResults.append(stepResult, ignore_index=True)

    return(scanResults)


def scanPlot(scanResults, paramKey, save=None):

    '''Function which plots the scan results (scan parameter vs. energy)

        Input:
        scanResults: numpy array (dim: scanPoints x 3 (+ number of tracked parameters)) - array holding the scan parameter value, energy (h), relative energy (kJ/mol) and the value of any tracked parameters for each optimized scan point
         atomID: List of ints - atom IDs of the scan parameter
         save: (optional) string - filename for saved image of the scan plot
    '''

    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Set figure and plot param(s) vs energy
    fig, ax = plt.subplots(figsize=(7,6))

    # Remove plot frame lines
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set colour list so that unoptimised points are coloured differently
    colList = []
    colours = ['#E71E47', '#0C739C']
    [colList.append(colours[opt]) for opt in scanResults['Optimised']]

    # Plot points and connecting lines
    ax.scatter(scanResults[paramKey], scanResults['Rel E (kJ/mol)'], color=colList, marker='o', s=50, alpha=0.6)
    ax.plot(scanResults[paramKey], scanResults['Rel E (kJ/mol)'], marker=None, alpha=0.3, color='#0C739C')

    # Build xlabel from parameter info
    labelTypes = {'B': 'R({0[0]}, {0[1]}) ($\AA$)' , 'A': 'A({0[0]}, {0[1]}, {0[2]}) ($^\circ$)', 'D': 'D({0[0]}, {0[1]}, {0[2]}, {0[3]}) ($^\circ$)'}
    xLabel = labelTypes[paramKey.split()[0]]
    ax.set_xlabel(xLabel.format(paramKey.split()[1:]), fontsize=13)
    ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.8)

    ax.legend(handles=[mlin.Line2D([], [], color='#E71E47', label='Unoptimised', marker='o', alpha=0.6, linestyle=' '), mlin.Line2D([], [], color='#0C739C', label='Optimised', marker='o', alpha=0.6, linestyle=' ')], frameon=False, handletextpad=0.1)

    if save != None:
        plt.savefig(save + '.png')

    return(fig, ax)


def writeFile(paramKey, scanSteps, scanResults):

    '''Function which writes the scan results to a csv file

        Inputs:
         paramKey: str - consists of modred parameter type code and the atoms IDs (real ones from log)
         scanSteps: int - number of steps in the scan
         scanResults: numpy array (dim: scanPoints x 3 (+ number of tracked parameters)) - array holding the scan parameter value, energy (h), relative energy (kJ/mol) and the value of any tracked parameters for each optimized scan point
     '''

    # Constructs file name, opens file and writes header lines
    filename = 'ps'
    for pK in paramKey.split():
        filename += pK
    with open(filename + '.csv', 'w+') as output:
        print('Results for relaxed scan of ' + paramKey, file=output)
        print('Scan parameter,E (h),Relative E (kJ/mol)', file=output)

        # Write scan results to file
        for sStep in range(scanSteps+1):
            print('{0[0]},{0[1]},{0[2]}'.format(scanResults[sStep,:]), file=output)


if __name__ == '__main__':

    '''Parse in the input log files of the scan calculations and any additional input file containing
        tracked parameters.
    '''

    usage = "usage: %(prog)s [inputFile(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("inputFiles", nargs='*', type=str, help="The resulting .log files with the scan in")
    parser.add_argument("-t", "--tparams", dest="trackedInput", nargs=1, type=str)
    parser.add_argument("-r", "--read", dest="read", action='store_true')
    args = parser.parse_args()


    if args.read == True:

        # Parse in csv file of scan results
        scanResults = pd.read_csv(args.inputFiles[0])
        paramKey = scanResults.columns[0]

        # Plots results
        fig, ax = scanPlot(scanResults, paramKey)
        plt.show()

    else:

        # Sets original scan information from first file and column names for results
        scanInfo = findParam(args.inputFiles[0])
        colNames = [scanInfo['paramKey'], 'E (h)', 'Optimised']

        # Process tracked parameters if any
        if args.trackedInput != None:
            trackedParams = parseTrackedParams(args.trackedInput[0])
            colNames += (list(trackedParams))
        else:
            trackedParams = {}

        # Set up results dataframe and scan information from first input file
        scanResults = pd.DataFrame(columns=colNames)
        scanResults = scan(scanInfo['atomInd'], scanInfo['nSteps'], scanResults, args.inputFiles[0], trackedParams)

        # For multiple scan files, check that the parameter being scanned is the same and add scan results
        if len(args.inputFiles) > 1:
            for inputFile in args.inputFiles[1:]:
                extraSteps = findParam(inputFile)
                if extraSteps['paramKey'] != scanInfo['paramKey']:
                    raise Exception('Parameter being scanned is not the same across the files')
                else:
                    scanResults = scan(extraSteps['atomInd'], extraSteps['nSteps'], scanResults, inputFile, trackedParams)

        # Process energy value in scan results for relative E in kJ/mol
        minE = scanResults['E (h)'].min()
        scanResults['Rel E (kJ/mol)'] = (scanResults['E (h)'] - minE)*2625.5
        scanResults = scanResults.sort_values(scanInfo['paramKey'])

        # Plots results and saves to csv file
        fig, ax = scanPlot(scanResults, scanInfo['paramKey'])
        plt.show()

        # Constructs file name, opens file and writes header lines
        filename = 'ps'
        for pK in scanInfo['paramKey'].split():
            filename += pK
        scanResults.to_csv(filename + '.csv', index=False)
