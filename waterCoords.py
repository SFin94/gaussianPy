import sys
import numpy as np
import gaussGeom as gg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# GS process
def gramScmidt(xVec, *bVecs):

    '''Function which calculates a new orthogonal basis vector by the Gram Shcmidt process'''

    sum = 0
    for bVec in bVecs:
        sum += (np.dot(xVec, bVec)/np.linalg.norm(bVec))*bVec
    bNew = xVec - sum
    return(bNew)


def tripleProduct(nBonds):

    '''Function which calculates the triple product for three vectors)

        Parameters:
        nBonds: List of three vectors

        Returns:
        tripleProd: float - triple product of the three
        '''

    nBcross = np.cross(nBonds[0], nBonds[1])
    tripleProd = np.dot(nBonds[2], nBcross)
    return(tripleProd)


# Rotation around y axis
def rotationY(feta, inVec, bOne = np.array([1., 0., 0.]), bTwo = np.array([0., 1., 0.]), bThree = np.array([0., 1., 0.])):

    '''Function which applies a rotation around the y axis (in the standard basis) by an input angle (feta)

        Parameters:
        feta: float - angle the vector is to be rotated around (radians)
        inVec: numpy array (dim: 1x3) - the vector to be rotated
        Optional parameters:
        bOne, bTwo, bThree: numpy arrays (dim: 1x3) - Standard basis or alternative basis if set (not sure matrix is then applicable though?)
        '''

    tX, tY, tZ = np.zeros(3), np.zeros(3), np.zeros(3)
    outVec = np.zeros(3)
    for bInd, bVec in enumerate([bOne, bTwo, bThree]):
        tX[bInd] = bVec[0]*np.cos(feta) + bVec[2]*np.sin(feta)
        tY[bInd] = bVec[1]
        tZ[bInd] = bVec[2]*np.cos(feta) - bVec[0]*np.sin(feta)

    outVec = np.array([np.dot(tX, inVec), np.dot(tY, inVec), np.dot(tZ, inVec)])

    return(outVec)



class InteractionSite:

    ''' Class - creates object for each water position to be set up

        Attributes:
            siteInd: int - molecular index of the donor/acceptor site
            neighbourInd: List of ints - molecular indexes for the covalently bonded neighbours of the sites.
            siteType: str - 'don'/'acc', denoting the type of interaction to be set up
    '''

    #tip3P geometry
    bondOH = 0.9572
    angleHOH = np.radians(104.52)

    def __init__(self, raw):

        # Set the index of the target mol
        self.atomInd = int(raw[0])

        # Set the list of covalently bonded neighbours
        self.neighbourInd = []
        for nB in raw[1].split(','):
            self.neighbourInd.append(int(nB))

        # Assign the type of site to be set up
        self.siteType = str(raw[2])


    def localGeom(self, geometry):

        self.coords = geometry[self.atomInd-1]
        self.neighbours = []
        for nbInd in self.neighbourInd:
            self.neighbours.append(geometry[nbInd-1])

    def bVectors(self, neighbourBonds):

        # Find centroid of the bonds and normalise
        b1 = np.array([0., 0., 0.])
        for nB in neighbourBonds:
            b1 += nB
            b1 /= len(neighbourBonds)
            b1 /= np.linalg.norm(b1)

         # Find orthonormal basis with b1 using cross products (GS process alternative works but was unnecessary as orthogonal basis already created)
        b2 = np.cross(self.coords, b1)
        b2 /= np.linalg.norm(b2)
        b3 = np.cross(b2, b1)
        b3 /= np.linalg.norm(b3)

        # Order of basis vectors is due to rotation later around y axis
        self.bBasis = np.array([b2, b3, b1])


class DonorInt(InteractionSite):

    def waterPosition(self):

        # Position the oxygen off the donor H
        self.OW = self.coords - self.bBasis[2]*2

        angles = [-(np.pi-self.angleHOH)/2, -(np.pi+self.angleHOH)/2]

        # Rotate standard basis by desired angles around the y axis
        rotOne = rotationY(angles[0], np.array([1., 0., 0.]))
        rotTwo = rotationY(angles[1], np.array([1., 0., 0.]))

        # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
        bPx = self.bBasis.transpose()
            # Transform the rotation vectors for the water H's to the donor basis, scale, and add to the water O
        self.HWOne = np.matmul(bPx, rotOne)*self.bondOH + self.OW
        self.HWTwo = np.matmul(bPx, rotTwo)*self.bondOH + self.OW

        def dummyPosition(self):

            self.dThree = self.HWOne + self.bBasis[0]


class AcceptorInt(InteractionSite):

    def waterPosition(self):

        self.HWOne = self.coords - self.bBasis[2]*2
        # Position O bond distance away from the H
        self.OW = self.HWOne - self.bBasis[2]*self.bondOH

        # For second OH bond the angle will be angleHOH - 90
        rot = rotationY(-(angleHOH - 0.5*np.pi), np.array([1., 0., 0.]))
        # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
        bPx = self.bBasis.transpose()
        # Transform the rotation vectors for the second water H to the donor basis, scale, and add to the water O
        self.HWTwo = self.OW - np.matmul(bPx, rot)*self.bondOH

    def dummyPosition(self):

        self.dThree = self.HWOne + self.bBasis[0]


def waterSetUp(siteList):

        '''Function to set up water molecule with acceptor interaction - calculates the b1 vector dependant on the number of covalently bonded neighbours and updates the water and dummy atom positons for the object through the acceptorCalc function

            Parameters:
             accMol - numpy array (dim: 1x3): x, y, z coordinates of the acceptor atom
             neighbours - List of numpy arrays (len: number of neighbours, dim 1x3): x, y, z coordinates of the covalently bonded neighbours to the acceptor atom
        '''

        # Calculate the neighbouring bonds between neighbouring positons
        cBonds = []
        for site in siteList:
            for nB in site.neighbours:
                cInitial = nB - site.coords
                cBonds.append(cInitial)

        # Set the basis vectors from the site
            site.bVectors(cBonds)

        # Test the triple product of the neighbour bond vectors
        # If close to 0 then they lie in the same plane and switch b1 for the orthogonal b2 or b3
            if len(cBonds) == 3:
                print(site.bBasis)
                if tripleProduct(cBonds) < 1e-03:
                    test = []
                    if abs(tripleProduct([cBonds[0], cBonds[1], site.bBasis[0]])) < 1e-03:
                        site.bBasis[[1, 2]] = site.bBasis[[2, 1]]
                    else: site.bBasis[[0, 2]] = site.bBasis[[2, 0]]

        site.waterPosition()
        site.dummyPosition()


# Add in ZMat calculation which has only the angle and dihedral left to optimise and fixes the others

def calcIdealZMat(self, target):

    numAtoms = len(self.atomIds)

    if target == 'don':

        # For ideal water O has one opt var and need to calculate angles and dihedrals
        OHx1 = gg.atomAngle(self.OW, self.targMol, self.dOne)
        OHx1x2 = gg.atomDihedral(self.OW, self.targMol, self.dOne, self.dTwo)
        OWzMat = {'Ow': numAtoms+3, self.targMolID: 'rDO', 'x1': OHx1, 'x2': OHx1x2}

        # For water H geom; both r: bondOH; both ang: donor H and diheds: to same dummy and  left to opt
        Hw1A = (180 - 104.52/2.)
        HWOnezMat = {'H1w': numAtoms+4, 'Ow': self.bondOH, self.targMolID: Hw1A, 'x2': 'H1wOHx'}
        HWTwozMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.targMolID: Hw1A, 'x2': 'H2wOHx'}

        # Calculate initial values for opt variables
        H1wOHx = gg.atomDihedral(self.HWOne, self.OW, self.targMol, self.dTwo)
        H2wOHx = gg.atomDihedral(self.HWTwo, self.OW, self.targMol, self.dTwo)
        self.optVar = {'rDO': 2.00, 'H1wOHx': H1wOHx, 'H2wOHx': H2wOHx}

        # Set list for writing the Z matrix section
        self.zMatList = [OWzMat, HWOnezMat, HWTwozMat]

    elif target == 'acc':

        # Define interacting H: rAh to Acceptor to opt; angle and dihed to dummy atoms (fixed)
        HAx1 = gg.atomAngle(self.HWOne, self.targMol, self.dOne)
        HAx1x2 = gg.atomDihedral(self.HWOne, self.targMol, self.dOne, self.dTwo)
        HWOnezMat = {'H1w': numAtoms+3, self.targMolID: 'rAH', 'x1': HAx1, 'x2': HAx1x2}

        # Second attempt trying to maintain linear interaction
        OAng = gg.atomAngle(self.OW, self.HWOne, self.dOne)
        ODihed = gg.atomDihedral(self.OW, self.HWOne, self.dOne, self.targMol)
        OWzMat = {'Ow': numAtoms+4, 'H1w': self.bondOH, 'x2': OAng, self.targMolID: ODihed}

        # Define 2nd H with r: OH bond distance to O; angle to Acceptor and dihed to dummy (left to opt)
        H2Ang = gg.atomAngle(self.HWTwo, self.OW, self.targMol)
        HWTwozMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.targMolID: H2Ang, 'x2': 'HOAx'}

        # Calculate initial values for opt variables
        HOAx = gg.atomDihedral(self.HWTwo, self.OW, self.targMol, self.dTwo)
        self.optVar = {'rAH': 2.00, 'HOAx': HOAx}

        # Set list for writing the Z matrix section
        self.zMatList = [HWOnezMat, OWzMat, HWTwozMat]


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

        # Define interacting H: rAh to Acceptor to opt; angle and dihed to dummy atoms (opt)
        HWOnezMat = {'H1w': numAtoms+3, self.targMolID: 'rAH', 'x1': 'HAx1', 'x2': 'HAx1x2'}

        ang = gg.atomAngle(self.OW, self.HWOne, self.dThree)
        dihed = gg.atomDihedral(self.OW, self.HWOne, self.dThree, self.targMol)

        dist = gg.atomDist(self.OW, self.dThree)
        ang = gg.atomAngle(self.OW, self.dThree, self.HWOne)
        dihed = gg.atomDihedral(self.OW, self.dThree, self.HWOne, self.targMol)
        OWzMat = {'Ow': numAtoms+4, 'x3': dist, 'H1w': ang, self.targMolID: dihed}

        ang = gg.atomAngle(self.HWTwo, self.OW, self.targMol)
        HWTwozMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.targMolID: ang, 'x2': 'HOAx'}

        # Start value for one dihedral seems to be out so calculate?
        HAx1Init = gg.atomAngle(self.HWOne, self.targMol, self.dOne)
        HAx1x2Init = gg.atomDihedral(self.HWOne, self.targMol, self.dOne, self.dTwo)
        HOAxInit = gg.atomDihedral(self.HWTwo, self.OW, self.targMol, self.dTwo)
        optVar = {'rAH': 2.00, 'HAx1': HAx1Init, 'HAx1x2': HAx1x2Init, 'HOAx': HOAxInit}

        zMatList = [HWOnezMat, OWzMat, HWTwozMat]


def writeZMat(self, target):
    with open('{}Int{}_idealZMat.com'.format(target, self.targMolID), 'w') as output:
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
        if target == 'acc':
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x3', self.dThree[:]), file=output)
        # Set z matrix input
        for atom in self.zMatList:
            zMatInput = '{:<4}'.format(list(atom.keys())[0])
            for entry in list(atom.items())[1:]:
                if isinstance(entry[1], str):
                    zMatInput += '{:>4}{: >8}'.format(entry[0], entry[1])
                else:
                    zMatInput += '{:>4}{: >8.2f}'.format(entry[0], entry[1])

            print(zMatInput, file=output)

        print('', file=output)
        # Enter initial variables
        for var, inVal in self.optVar.items():
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
        if target == 'acc':
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x3', self.dThree[:]), file=output)
        print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('Ow', self.OW[:]), file=output)
        print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('H1w', self.HWOne[:]), file=output)
        print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('H2w', self.HWTwo[:]), file=output)
        print('\n\n', file=output)



if __name__ == '__main__':

# Input file format: siteInd neighbourInds target lp(optional)

# Want to edit so either single command line argument or multiple at once
# If an input file then it will have geometry store - but same for each molecule
# then input rows of atomInd neighbours don/acc

    with open(str(sys.argv[1]), 'r') as inputFile:
        input = inputFile.readlines()

    # Pull the geometry and atomIDs from the log file
    geomFile = input[0].split()[0]
    numAtoms = int(input[0].split()[1])

    geometry = gg.geomPulllog(geomFile, numAtoms)
    ids = gg.atomIdentify(geomFile, 7)


    # The remaining lines will contain the information for each donor/acceptor site
# Read in raw data; then for each site set up geometry - do initially without lone pairs
    siteIDs, siteList = [], []
    # Read in raw data; then for each site set up geometry - do initially without lone pairs
    for el in input[1:]:
        siteIDs.append(el.split()[0])
        if el.split()[3] == 'don':
            siteList.append(DonorInt(el.split()[1:]))
        elif el.split()[3] == 'acc':
            siteList.append(AcceptorInt(el.split()[1:]))

    for site in siteList:
        print(site.siteType)
        site.localGeom(geometry)

    waterSetUp(siteList)

#
#    if target == 'don':
#
#        donor = waterPosition(geometry, ids, 'don', int(sys.argv[3]), neighbours)
#        donor.waterSetUp(target)
#        donor.calcIdealZMat(target)
#        donor.writeZMat(target)
#
#        # Calculates and writes the z matrix or just coordinates
#
#    elif target == 'acc':
#
#        if len(neighbours) == 4:
#            for nB in range(4):
#                neighboursList = [neighbours[nB % 4], neighbours[(nB+1) % 4], neighbours[(nB+2) % 4]]
#                acceptor = waterPosition(geometry, ids, 'acc', int(sys.argv[3]), neighboursList)
#                acceptor.waterSetUp(target)
#                acceptor.writeCoords(target+str(nB))
#
#        else:
#            acceptor = waterPosition(geometry, ids, 'acc', int(sys.argv[3]), neighbours)
#            acceptor.waterSetUp(target)
#            acceptor.calcIdealZMat(target)
#            acceptor.writeZMat(target)
#            acceptor.writeCoords(target)
#
## Now should be able to edit final section as doesn't need to be seperate - does it need to even be a class?
#
