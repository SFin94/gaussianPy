import sys
import numpy as np
import gaussGeom as gg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class waterPosition:

    # Set geometry constants for TIP3P water molecule
    bondOH = 0.9572
    angleHOH = np.radians(104.52)
    angles = [-(np.pi-angleHOH)/2, -(np.pi+angleHOH)/2]

    # Attributes: water positions: OW; HWOne; HWTwo and dummy positions dOne; dTwo

    def __init__(self, geometry, ids, target, targMolInd, neighbourInds):

        '''
        Initilises class and sets the object attributes: geometry, atomIds, neighbours (neighbour xyz positions)

        Parameters:
         geometry - np.array (dim: nAtom x 3): x, y, z coordinates for each atom in molecule
         ids - list (len: nAtoms): Atomic symbol for each atom in molecule
         target - str: 'acc' or 'don' which decides set up route of the water
         targMol - int: atom index of the acceptor or donor atom of the molecule
         neighbourInds - int (len: no. of neighbours): atom indexes of the covalently bonded neighbours to the acceptor/donor atom
        '''

        # Set IDs and geometries as object attributes, add neighbours?
        self.atomIds = ids
        self.geometry = geometry
        self.targMol = geometry[targMolInd-1]

        # Set the ID string for a neighbour atom and the donor/acceptor target atom
        self.targMolID = self.atomIds[targMolInd-1] + str(targMolInd)
        self.neighbourID = self.atomIds[int(neighbourInds[0])-1] + neighbourInds[0]

        # Set neighbours list with xyz coords of colvalently bonded neighbours
        self.neighbours = []
        for nB in neighbourInds:
            self.neighbours.append(self.geometry[int(nB) - 1])

    # GS process
    def gsProcess(xVec, *bVecs):

        '''Function which calculates a new orthogonal basis vector by the Gram Shcmidt process'''

        sum = 0
        for bVec in bVecs:
            sum += (np.dot(xVec, bVec)/np.linalg.norm(bVec))*bVec
        bNew = xVec - sum
        return(bNew)

    # Rotation around y axis
    def rotationY(self, feta, inVec, bOne = np.array([1., 0., 0.]), bTwo = np.array([0., 1., 0.]), bThree = np.array([0., 1., 0.])):

        '''Function which applies a rotation around the y axis (in the standard basis) by an input angle

        Parameters:
         feta - float: angle the vector is to be rotated around (radians)
         inVec - numpy array (dim: 1x3): the vector to be rotated
        Optional parameters:
         bOne, bTwo, bThree - numpy arrays (dim: 1x3): Standard basis or alternative basis if set (not sure matrix is then applicable though?)
        '''

        tX, tY, tZ = np.zeros(3), np.zeros(3), np.zeros(3)
        outVec = np.zeros(3)
        for bInd, bVec in enumerate([bOne, bTwo, bThree]):
            tX[bInd] = bVec[0]*np.cos(feta) + bVec[2]*np.sin(feta)
            tY[bInd] = bVec[1]
            tZ[bInd] = bVec[2]*np.cos(feta) - bVec[0]*np.sin(feta)

        outVec = np.array([np.dot(tX, inVec), np.dot(tY, inVec), np.dot(tZ, inVec)])

        return(outVec)


    def donorSetUp(self):

        '''Function to set up water molecule with donor interaction - updates the water and dummy atom positons for the object

        Parameters:
         donMol - numpy array (dim: 1x3): x, y, z coordinates of the donor atom in the molecule
         neighbour - numpy array (dim: 1x3): x, y, z coordinates of the neighbouring atom to the H donor atom in the molecule (should only be one as donor always H?)
        '''

        # Calculate vector b1 using the donor and H positions, use to calculate O position
        b1 = self.targMol - self.neighbours[0]
        b1 /= np.linalg.norm(b1)
        self.OW = self.targMol + b1*2

        # Find orthonormal basis with b1 using cross products (GS process alterantive works but was unnecessary as orthogonal basis already created)
        b2 = np.cross(self.targMol, self.neighbours[0])
        b2 /= np.linalg.norm(b2)
        b3 = np.cross(b2, b1)
        b3 /= np.linalg.norm(b3)

        # Define dummy atoms for angles from the H donor
        self.dOne = self.targMol + b2
        self.dTwo = self.targMol + b3

        # Rotate standard basis by desired angles around the y axis
        rotOne = self.rotationY(self.angles[0], np.array([1., 0., 0.]))
        rotTwo = self.rotationY(self.angles[1], np.array([1., 0., 0.]))

        # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
        bPx = np.array([b2, b3, b1]).transpose()
        # Transform the rotation vectors for the water H's to the donor basis, scale, and add to the water O
        self.HWOne = np.matmul(bPx, rotOne)*self.bondOH + self.OW
        self.HWTwo = np.matmul(bPx, rotTwo)*self.bondOH + self.OW


#    def acceptorCalc(self, b1):
#
#        '''Function which calculates the position of the water molecule for an inputted position of the acceptor atom and b1 vector, updating the positions of the water and dummy atoms.
#
#            Parameters:
#             accMol- numpy array (dim: 1x3): x, y, z coordinates of the acceptor atom
#             b1Pos - numpy array (dim: 1x3): x, y, z coordinates of the b1 site of the acceptor atom
#        '''
#
#        self.HWOne = self.targMol + b1*2
#        # Position O bond distance away from the H
#        self.OW = self.HWOne + b1*self.bondOH
#
#        # Position the two dummy atoms
#        # Find orthonormal basis with b1 using cross products (GS process alternative works but was unnecessary as orthogonal basis already created)
#        b2 = np.cross(self.targMol, b1)
#        b2 /= np.linalg.norm(b2)
#        b3 = np.cross(b2, b1)
#        b3 /= np.linalg.norm(b3)
#
#        # Define dummy atoms for angles from the H donor
#        self.dOne = self.targMol + b2
#        self.dTwo = self.targMol + b3
#
#        # For second OH bond the angle will be angleHOH - 90
#        rot = self.rotationY(-(self.angleHOH - 0.5*np.pi), np.array([1., 0., 0.]))
#        # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
#        bPx = np.array([b2, b3, b1]).transpose()
#        # Transform the rotation vectors for the second water H to the donor basis, scale, and add to the water O
#        self.HWTwo = np.matmul(bPx, rot)*self.bondOH + self.OW


    def waterSetUp(self, target):

        '''Function to set up water molecule with acceptor interaction - calculates the b1 vector dependant on the number of covalently bonded neighbours and updates the water and dummy atom positons for the object through the acceptorCalc function

            Parameters:
             accMol - numpy array (dim: 1x3): x, y, z coordinates of the acceptor atom
             neighbours - List of numpy arrays (len: number of neighbours, dim 1x3): x, y, z coordinates of the covalently bonded neighbours to the acceptor atom
        '''

        # Calculate the neighbouring bonds between neighbouring positons
        cBonds = []
        for nB in self.neighbours:
            cInitial = nB - self.targMol
            cBonds.append(cInitial)

        # Find centroid of the bonds - centre of mass?- could be same as case for 2?
        b1 = np.array([0., 0., 0.])
        for cB in cBonds:
            b1 += cB
        b1 /= len(cBonds)
        b1 /= np.linalg.norm(b1)

        # Position the two dummy atoms
        # Find orthonormal basis with b1 using cross products (GS process alternative works but was unnecessary as orthogonal basis already created)
        b2 = np.cross(self.targMol, b1)
        b2 /= np.linalg.norm(b2)
        b3 = np.cross(b2, b1)
        b3 /= np.linalg.norm(b3)

        # Define dummy atoms for angles from the H donor
        self.dOne = self.targMol + b2
        self.dTwo = self.targMol + b3

        if target == 'don':

            self.OW = self.targMol + b1*2

            # Rotate standard basis by desired angles around the y axis
            rotOne = self.rotationY(self.angles[0], np.array([1., 0., 0.]))
            rotTwo = self.rotationY(self.angles[1], np.array([1., 0., 0.]))

            # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
            bPx = np.array([b2, b3, b1]).transpose()
            # Transform the rotation vectors for the water H's to the donor basis, scale, and add to the water O
            self.HWOne = np.matmul(bPx, rotOne)*self.bondOH + self.OW
            self.HWTwo = np.matmul(bPx, rotTwo)*self.bondOH + self.OW

        elif target == 'acc':

            self.HWOne = self.targMol - b1*2
            # Position O bond distance away from the H
            self.OW = self.HWOne - b1*self.bondOH

            # For second OH bond the angle will be angleHOH - 90
            rot = self.rotationY(-(self.angleHOH - 0.5*np.pi), np.array([1., 0., 0.]))
            # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
            bPx = np.array([b2, b3, b1]).transpose()
            # Transform the rotation vectors for the second water H to the donor basis, scale, and add to the water O
            self.HWTwo = self.OW - np.matmul(bPx, rot)*self.bondOH

            self.dThree = self.HWOne + b2


    def idealisedInt(self, target):

        # Only calculate for


    def calcZMat(self, target):

        numAtoms = len(self.atomIds)

        if target == 'don':

            # For water O has three opt vars and calculates initial values
            OWzMat = {'Ow': numAtoms+3, self.targMolID: 'rDO', 'x1': 'OHx1', 'x2': 'OHx1x2'}
            OHx1 = gg.atomAngle(self.OW, self.targMol, self.dOne)
            OHx1x2 = gg.atomDihedral(self.OW, self.targMol, self.dOne, self.dTwo)

            # For water H geom; both r: bondOH; angle of first to H; second to water angleHOH; do dihedrals to first dummy
            Hw1A = (180 - 104.52/2.)
            HWOnezMat = {'H1w': numAtoms+4, 'Ow': self.bondOH, self.targMolID: Hw1A, 'x2': 'H1wOHx'}
            HWTwozMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.targMolID: Hw1A, 'x2': 'H2wOHx'}

            H1wOHx = gg.atomDihedral(self.HWOne, self.OW, self.targMol, self.dTwo)
            H2wOHx = gg.atomDihedral(self.HWTwo, self.OW, self.targMol, self.dTwo)

            optVar = {'rDO': 2.00, 'OHx1': OHx1, 'OHx1x2': OHx1x2, 'H1wOHx': H1wOHx, 'H2wOHx': H2wOHx}
            zMatList = [OWzMat, HWOnezMat, HWTwozMat]

        elif target == 'acc':

            # Trial one - add third dummy atom
            HWOnezMat = {'H1w': numAtoms+3, self.targMolID: 'rAH', 'x1': 'HAx1', 'x2': 'HAx1x2'}

            ang = gg.atomAngle(self.OW, self.HWOne, self.dThree)
            dihed = gg.atomDihedral(self.OW, self.HWOne, self.dThree, self.targMol)
            OWzMat = {'Ow': numAtoms+5, 'H1w': self.bondOH, 'x3': ang, self.targMolID: dihed}

            ang = gg.atomAngle(self.HWTwo, self.OW, self.targMol)
            HWTwozMat = {'H2w': numAtoms+6, 'Ow': self.bondOH, self.targMolID: ang, 'x2': 'HOHA'}

            # Start value for one dihedral seems to be out so calculate?
            HOHAInit = gg.atomDihedral(self.HWTwo, self.OW, self.targMol, self.dTwo)
            optVar = {'rAH': 2.00, 'HAx1': 90.0, 'HAx1x2': -90.0, 'HOHA': HOHAInit}


            zMatList = [HWOnezMat, OWzMat, HWTwozMat]

        with open('{}Int{}_ZMat.com'.format(target, self.targMolID), 'w') as output:
            print('%Chk={}Int{}_Zmat'.format(target, self.targMolID), file=output)
            print('%NProcShared=12', file=output)
            print('%Mem=46000MB', file=output)
            print('#P HF/6-31G(d) Opt(Z-Matrix,MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)
            print('Water {} interaction for {}\n'.format(target, self.targMolID), file=output)
            print('0 1', file=output)
            # Print original molecular geometry
            for atomInd, el in enumerate(self.geometry):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format((self.atomIds[atomInd])+str(atomInd+1), el[:]), file=output)
            # Print both dummy atoms - don't need to be ZMat and should maintain consistency
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x1', self.dOne[:]), file=output)
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x2', self.dTwo[:]), file=output)
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x3', self.dThree[:]), file=output)
            # Set z matrix input
            for atom in zMatList:
                zMatInput = '{:<4}'.format(list(atom.keys())[0])
                for entry in list(atom.items())[1:]:
                    if isinstance(entry[1], str):
                        zMatInput += '{:>4}{: >8}'.format(entry[0], entry[1])
                    else:
                        zMatInput += '{:>4}{: >8.2f}'.format(entry[0], entry[1])

                print(zMatInput, file=output)

            print('', file=output)
            # Enter initial variables
            for var, inVal in optVar.items():
                print('{:<8}{:>6.2f}'.format(var, inVal), file=output)
            print('\n\n', file=output)


    def writeCoords(self, target):

        with open('{}Int{}_coords.com'.format(target, self.targMolID), 'w') as output:
            print('%Chk={}Int{}'.format(target, self.targMolID), file=output)
            print('%NProcShared=24', file=output)
            print('%Mem=61000MB', file=output)
            print('#P HF/6-31G(d) Opt(MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)
            print('Water {} interaction for {}\n'.format(target, self.targMolID), file=output)
            print('0 1', file=output)
            # Print original molecular geometry
            for atomInd, el in enumerate(self.geometry):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format((self.atomIds[atomInd]), el[:]), file=output)
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x1', self.dOne[:]), file=output)
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x2', self.dTwo[:]), file=output)
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('Ow', self.OW[:]), file=output)
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('H1w', self.HWOne[:]), file=output)
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('H2w', self.HWTwo[:]), file=output)
            print('\n\n', file=output)



if __name__ == '__main__':

    '''Input arguments:
        1: str - geometry and atom ids input file
        2: str: acc or don to set target interaction
        3: int: index of acceptor or donor molecule
        4: ints: list of csv neighbour indexes
    '''

    # Pull optimised geometry from the file
    inputFile = str(sys.argv[1])
    geometry = gg.geomPulllog(inputFile, 7)
    ids = gg.atomIdentify(inputFile, 7)
    # Set target interation as either donor or acceptor for set up and neighbours
    target = str(sys.argv[2]).lower()
    neighbours = str(sys.argv[4]).split(',')

    if target == 'don':

        donor = waterPosition(geometry, ids, 'don', int(sys.argv[3]), neighbours)
        donor.waterSetUp(target)
        donor.calcZMat(target)

        # Calculates and writes the z matrix or just coordinates

    elif target == 'acc':

        if len(neighbours) == 4:
            for nB in range(4):
                neighboursList = [neighbours[nB % 4], neighbours[(nB+1) % 4], neighbours[(nB+2) % 4]]
                acceptor = waterPosition(geometry, ids, 'acc', int(sys.argv[3]), neighboursList)
                acceptor.waterSetUp(target)
                acceptor.writeCoords(target+str(nB))

        else:
            acceptor = waterPosition(geometry, ids, 'acc', int(sys.argv[3]), neighbours)
            acceptor.waterSetUp(target)
            acceptor.calcZMat(target)
            acceptor.writeCoords(target)



