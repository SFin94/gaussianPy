import numpy as np
import sys


''' A module which contains functions to extract geometries from Gaussian log files or xyz file, extract energies from Gaussian log files, and calculate geometric parameters
'''

def countAtoms(inputFile)

'''Function to count the number of atoms in a molecule from a gaussian log file

    Inputs:
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


def geomPulllog(inputFile, numAtoms=None, optStep=1):

    '''Function which extracts the optimised geometry of a molecule in the standard orientation from a Guassian .log file. NB: The standard orientation output is at the start of the next optimisation cycle, before 'Optimized' would be met.

    Parameters:
     inputFile: Str - name of the input log file
     numAtoms: Int - The number of atoms in the system
     optStep: Int - Optional argument, the optimised structure number wanted from the file (intermediate structure may be desired from a scan calculation. Default value = 1.

     Returns:
      molCoords: Numpy array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom
      optimised: Bool - False if not optimised and true if optimised geometry
    '''
    # If number of atoms is not inputted then searches file for the number
    if numAtoms == None:
        numAtoms = countAtoms(inputFile)

    # Number of 'optimized steps' encountered through file
    optCount = 0
    # Set up array for coordinates
    molCoords = np.zeros((numAtoms, 3))

    # Open and read input file
    with open(inputFile, 'r') as logFile:
        for el in logFile:
            # Standard orientation output precedes the corresponding optimisation section
            if 'Standard orientation:' in el:
                # Skip the header section of the standard orientation block
                [logFile.__next__() for x in range(0,4)]
                # Read in the atomic coordinates, atom ID will be row index
                for ind in range(numAtoms):
                    el = logFile.__next__()
                    for jind in range(3):
                        molCoords[ind, jind] = float(el.split()[jind+3])

            # Increments optCount if 'Optimized' met, breaks loop if target opt step is reached
            if 'Optimized Parameters' in el:
                optCount += 1
                optimised = True
                if 'Non-Optimized' in el:
                    optimised = False
            if (optCount == optStep):
                return(molCoords, optimised)
                break
    return(molCoords, optimised)


def energyPull(inputFile, optStep=1):

    '''Function which extracts the optimised energy of a molecule from a Guassian .log file. NB: The SCF Done output is at the start of the next optimisation cycle, before 'Optimized' would be met.

        Parameters:
        inputFile: Str - name of the input log file
        optStep: Int - Optional argument, the optimised structure number wanted from the file (intermediate structure may be desired from a scan calculation. Default value = 1.

        Returns:
        molEnergy: float -  SCF Done energy
        optimised: Bool - False if not optimised and true if optimised geometry
        '''

    # Number of 'optimized steps' encountered through file
    optCount = 0
    # Open and read input file
    with open(inputFile, 'r') as logFile:
        for el in logFile:
            # SCF Done output precedes the correspdoning optimisation section
            if 'SCF Done:' in el:
                molEnergy = float(el.split('=')[1].split()[0])

            # Increments optCount if 'Optimized' met, breaks loop if target opt step is reached
            if 'Optimized Parameters' in el:
                optCount += 1
                optimised = True
                if 'Non-Optimized' in el:
                    optimised = False

            if (optCount == optStep):
                return(molEnergy, optimised)
                break
    return(molEnergy, optimised)


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


def atomIdentify(inputFile, numAtoms=None):

    '''Function which extracts the atom IDs from a gaussian log file.

        Parameters:
         inputFile: Str - name of the input log file
         numAtoms: Int - The number of atoms in the system

         Returns:
          atomIDs: List of str - atom IDs
        '''
    # If number of atoms is not inputted then searches file for the number
    if numAtoms == None:
        numAtoms = countAtoms(inputFile)

    atomIDs = []
    with open(inputFile, 'r') as logFile:
        for el in logFile:
            if 'Charge = ' in el:
                el = logFile.__next__()
                if ('No Z-Matrix' in el) or ('Redundant internal coordinates' in el):
                    el = logFile.__next__()
                for atom in range(numAtoms):
                    atomIDs.append(el.split()[0][0])
                    el = logFile.__next__()
                break
    return(atomIDs)


def paramGeom(paramInd, geometry):

    paramVal = []
    for pI in paramInd:
        if len(pI) == 2:
            paramVal.append(atomDist(geometry[pI[0]], geometry[pI[1]]))
        elif len(pI) == 3:
            paramVal.append(atomAngle(geometry[pI[0]], geometry[pI[1]], geometry[pI[2]]))
        else:
            paramVal.append(atomDihedral(geometry[pI[0]], geometry[pI[1]], geometry[pI[2]], geometry[pI[3]]))
    return(paramVal)


def atomDist(atomOne, atomTwo):

    ''' Simple function whih calculates the distance between two atoms
        Parameters:
        atomOne - np array; x, y, z coordinates of atom one
        atomTwo - np array; x, y, z coordinates of atom two

        Returns:
        dist - float; distance between the two atoms
        '''
    # Calculates the bond vector between the two atoms
    bVec = atomOne - atomTwo
    # Calculates the inner product of the vectors (magnitude)
    dist = np.sqrt(np.dot(bVec, bVec))
    return dist


def atomAngle(atomOne, atomTwo, atomThree):

    ''' Simple function which calculates the angle between three atoms, middle atom is atomTwo
        Parameters:
        atomOne - np array; x, y, z coordinates of atom one
        atomTwo - np array; x, y, z coordinates of atom two
        atomThree - np array; x, y, z coordinates of atom three

        Returns:
        angle - float; angle between the two vectors: (atomTwo, atomOne) and (atomTwo, atomThree)
        '''
    # Calculate the two bond vectors
    bOneVec = atomOne - atomTwo
    bTwoVec = atomThree - atomTwo

    # Calculate the inner products of the two bonds with themselves and each other
    bOne = np.sqrt(np.dot(bOneVec, bOneVec))
    bTwo = np.sqrt(np.dot(bTwoVec, bTwoVec))
    angle = np.dot(bOneVec, bTwoVec)/(bOne*bTwo)

    # Return the angle between the bonds in degrees
    return np.arccos(angle)*(180/np.pi)


def atomDihedral(atomOne, atomTwo, atomThree, atomFour):

    ''' Simple function to calculate the dihedral angle between four atoms
    Parameters:
     atomOne - np array; x, y, z coordinates of atom one
     atomTwo - np array; x, y, z coordinates of atom two
     atomThree - np array; x, y, z coordinates of atom three
     atomFour - np array; x, y, z coordinates of atom four

    Returns:
     dihedral - float; dihedral angle between the planes: (atomOne, Two, Three) and (atomTwo, Three, Four)
    '''

    bOneVec = atomTwo - atomOne
    bTwoVec = atomThree - atomTwo
    bThreeVec = atomFour - atomThree

    # Calculate the norms to the planes
    nOne = np.cross(bOneVec, bTwoVec)
    nTwo = np.cross(bTwoVec, bThreeVec)

    # Normalise the two vectors
    nOne /= np.linalg.norm(nOne)
    nTwo /= np.linalg.norm(nTwo)
    bTwoVec /= np.linalg.norm(bTwoVec)

    # Find third vector to create orthonormal frame
    mOne = np.cross(nOne, bTwoVec)

    # Evaluate n2 w.r.t the orthonormal basis
    x = np.dot(nTwo, nOne)
    y = np.dot(nTwo, mOne)

    return(np.arctan2(-y, x)*(180/np.pi))


#if __name__ == '__main__':

#    input = (str(sys.argv[1]))
#    molecule = geomPull(input, 27)
#    print(molecule)
