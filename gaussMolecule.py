import numpy as np
import sys


class Molecule():

    '''
    Class attributes
        numAtoms: Int - The number of atoms in the molecule
        atomIDs: List of str - atom IDs of the atoms in the molecule
        atomCoords: Numpy array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom
        optimised: Bool - flag of whether the molecule is optimised or not [Default: None if not known]
        energy: Flost - Energy of molecule in hartree
        energy_kj: Float - Energy of molecule in kJ/mol

    Class methods
        countAtoms
        analyseLog
    '''

    def __init__(self, inputFile, optStep=1, mp2=False):

        fileType = inputFile.split('.')[1]
        self.Optimised = None

        if fileType == 'log':

            self.numAtoms = self.countAtoms(inputFile)
            self.atomCoords, self.atomIDs, self.energy = self.analyseLog(inputFile, optStep, mp2)
            self.energy_kj = self.energy*2625.5


        if fileType == 'xyz':

#            self.numAtoms, self.atomCoords, self.atomIDs =
            self.energy = None
            self.energy_kj = None



    def countAtoms(self, inputFile):

        '''Function to count the number of atoms in a molecule from a gaussian log file

        Parameters:
         inputFile: Str - name of the input log file

        Returns:
         numAtoms: Int - The number of atoms in the system
        '''

        # Opens file and searches for line which contains the number of atoms in the system
        with open(inputFile, 'r') as logFile:
            for el in logFile:
                if 'NAtoms' in el:
                    numAtoms = int(el.split()[1])

        return(numAtoms)


    def analyseLog(self, inputFile, optStep, mp2):

        '''Class method which analyses the log file and pulls out the geometry, atom IDs and energy

        Parameters:
         inputFile: Str - name of the input log file

        Returns:
         atomCoords: Numpy array (dim: self.numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom
         atomIDs: List of str - atom IDs of the atoms in the molecule
         optimised:
         energy:
        '''

        # Number of 'optimized steps' encountered through file
        optCount = 0
        atomIDs = []
        try:
            atomCoords = np.zeros((self.numAtoms, 3))
        except:
            print('Number of atoms not set from log file')


        # Open and read input file
        with open(inputFile, 'r') as logFile:
            for el in logFile:

                # Sets atomIDs from initialising list of input structure
                if 'Charge = ' in el:
                    el = logFile.__next__()
                    if ('No Z-Matrix' in el) or ('Redundant internal coordinates' in el):
                        el = logFile.__next__()
                    for atom in range(self.numAtoms):
                        atomIDs.append(el.split()[0][0])
                        el = logFile.__next__()


                # NB: SCF Done and standard orientation output precede the corresponding optimisation section
                if 'SCF Done:' in el:
                    molEnergy = float(el.split('=')[1].split()[0])
                # MP2 energy printed out seperately - has to be processed to float form
                if mp2 == True:
                    if 'EUMP2' in el:
                        mp2Raw = el.split('=')[2].strip()
                        molEnergy = float(mp2Raw.split('D')[0])*np.power(10, float(mp2Raw.split('D')[1]))

                if 'Standard orientation:' in el:
                    # Skip the header section of the standard orientation block
                    [logFile.__next__() for x in range(0,4)]
                    # Read in the atomic coordinates, atom ID will be row index
                    for ind in range(self.numAtoms):
                        el = logFile.__next__()
                        for jind in range(3):
                            atomCoords[ind, jind] = float(el.split()[jind+3])

                # Increments optCount if 'Optimized' met, breaks loop if target opt step is reached
                if 'Optimized Parameters' in el:
                    optCount += 1
                    if 'Non-Optimized' in el:
                        self.optimised = False
                    else:
                        self.optimised = True
                if (optCount == optStep):
                    break

        return(atomCoords, atomIDs, molEnergy)


        def geomPullxyz(inputFile):

            '''Function which extracts the optimised geometry of a molecule from an .xyz file.

                Parameters:
                 inputFile: Str - name of the input log file

                Returns:
                 molCoords: Numpy array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom
                '''

            # Open and read input file
            with open(inputFile, 'r') as xyzFile:

                for el in xyzFile:

                    # Set atom number from first line of xyz file
                    numAtoms = int(el.strip())
                    [xyzFile.__next__() for i in range(1)]

                    molCoords = np.zeros((numAtoms, 3))
                    atomIDs = []

                    # Read in the atomic coordinates, atom ID will be row index
                    for ind in range(numAtoms):
                        el = xyzFile.__next__()
                        atomIDs.append(str(el.split()[0]))
                        for jind in range(1, 3):
                            molCoords[ind, jind] = float(el.split()[jind])

            return(molCoords, atomIDs)
#
#
#def geomPushxyz(outputFile):
#
#    '''Function which outputs the extracted geometry to an .xyz file.
#
#        Parameters:
#         outputFile: Str - name of the output xyz file
#    '''
#
#    # Open output file, print header lines then atom indexes and cartesian coordinates to file
#    with open(outputFile + '.xyz', 'w+') as output:
#        print(numAtoms, file=output)
#        print('Structure of {} from {}'.format(fileName, inputFile), file=output)
#        for atomInd, atom in enumerate(atomID):
#            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom, coordinates[atomInd]), file=output)
#
#
#def atomIdentify(inputFile, numAtoms=None):
#
#    '''Function which extracts the atom IDs from a gaussian log file.
#
#        Parameters:
#         inputFile: Str - name of the input log file
#         numAtoms: Int - The number of atoms in the system
#
#         Returns:
#          atomIDs: List of str - atom IDs
#        '''
#
#    atomIDs = []
#    with open(inputFile, 'r') as logFile:
#        for el in logFile:
#            if 'Charge = ' in el:
#                el = logFile.__next__()
#                if ('No Z-Matrix' in el) or ('Redundant internal coordinates' in el):
#                    el = logFile.__next__()
#                for atom in range(numAtoms):
#                    atomIDs.append(el.split()[0][0])
#                    el = logFile.__next__()
#                break
#    return(atomIDs)
#






