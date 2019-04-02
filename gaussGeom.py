import numpy as np
import os
import sys


''' A module which contains functions to extract the geometry from a Gaussian log file and can calculate geomertic parameters
'''



def geomPull(inputFile, numAtoms, optStep=1):

    '''Function which extracts the optimised geometry of a molecule in the standard orientation from a Guassian .log file. NB: The standard orientation output is at the start of the next optimisation (?) cycle, before 'Optimization' would be met.

    Parameters:
     inputFile: Str - name of the input log file
     numAtoms: Int - The number of atoms in the system
     oprtStep: Int - Optional argument, the optimised structure number wanted from the file (intermediate structure may be desired from a scan calculation. Default value = 1.

     Returns:
      molCoords: Numpy array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom
    '''

    # Number of 'optimized steps' encountered through file
    optCount = 0
    # Set up array for coordinates
    molCoords = np.zeros((numAtoms, 3))

    # Open and read input file
    with open(inputFile, 'r') as logFile:
        for el in logFile:
            # Standard orientation output precedes the corresponding optimisation section
            if ('Standard orientation:' in el):
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
            if (optCount == optStep):
                return(molCoords)
    return(molCoords)


def atomIdentify(inputFile, numAtoms):

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

    bOneVec = atomOne - atomTwo
    bTwoVec = atomThree - atomTwo
    bThreeVec = atomFour - atomThree

    # Calculate the norms to the planes
    nOne = np.cross(bOneVec, bTwoVec)
    nTwo = np.cross(bThreeVec, bTwoVec)

    # Normalise the two vectors
    nOne /= np.linalg.norm(nOne)
    nTwo /= np.linalg.norm(nTwo)
    bTwoVec /= np.linalg.norm(bTwoVec)

    # Find third vector to create orthonormal frame
    mOne = np.cross(nOne, bTwoVec)

    # Evaluate n2 w.r.t the orthonormal basis
    x = np.dot(nTwo, nOne)
    y = np.dot(nTwo, mOne)

    return np.arctan2(y, x)*(180/np.pi)


#if __name__ == '__main__':

#    input = (str(sys.argv[1]))
#    molecule = geomPull(input, 27)
#    print(molecule)
