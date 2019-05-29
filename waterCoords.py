import sys
import numpy as np
import gaussGeom as gg


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

        # Set the id index of the target mol
        self.atomID = raw[0]
        self.atomInd = int(raw[1])

        # Set the list of covalently bonded neighbours
        self.neighbourInd = []
        for nB in raw[2].split(','):
            self.neighbourInd.append(int(nB))

        # Assign the type of site to be set up
        self.siteType = str(raw[3])


    def localGeom(self, geometry):

        self.coords = geometry[self.atomInd-1]
        self.neighbours = []
        for nbInd in self.neighbourInd:
            self.neighbours.append(geometry[nbInd-1])

    def bVectors(self):

        # Find centroid of the bonds and normalise for first basis vector
        cBonds = self.neighbours - self.coords
        b1 = np.sum(cBonds, axis=0)/len(self.neighbourInd)
        b1 /= np.linalg.norm(b1)

        # Find orthonormal basis from b1 using cross products
        b2 = np.cross(self.coords, b1)
        b2 /= np.linalg.norm(b2)
        b3 = np.cross(b2, b1)
        b3 /= np.linalg.norm(b3)

        # Order of basis vectors is due to rotation later around y axis
        self.bBasis = np.array([b2, b3, b1])

        # Test the triple product of the neighbour bond vectors
        # If close to 0 then they lie in the same plane and switch b1 for the orthogonal b2 or b3
        if len(self.neighbourInd) == 3:
            if abs(tripleProduct(cBonds)) < 1e-03:
                if abs(tripleProduct([cBonds[0], cBonds[1], site.bBasis[0]])) < 1e-03:
                    site.bBasis[[1, 2]] = site.bBasis[[2, 1]]
                else: site.bBasis[[0, 2]] = site.bBasis[[2, 0]]


    def writeZMat(self, geometry, atomIDs, name='zMat'):
        with open('{}Int{}_{}.com'.format(self.siteType, self.atomID, name), 'w') as output:
            print('%Chk={}Int{}_{}'.format(self.siteType, self.atomID, name), file=output)
            print('%NProcShared=12', file=output)
            print('%Mem=46000MB', file=output)
            print('#P HF/6-31G(d) Opt(Z-Matrix,MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)
            print('Water {} interaction for {}\n'.format(self.siteType, self.atomID), file=output)
            print('0 1', file=output)
            # Print original molecular geometry
            for atomInd, el in enumerate(geometry):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format((atomIDs[atomInd])+str(atomInd+1), el[:]), file=output)
            # Print dummy atom coordinates
            for dInd, dAtom in enumerate(self.dummyAtoms):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x' + str(dInd+1), dAtom[:]), file=output)
            # Process z matrix into string and print for water position
            for waterAtom in self.zMatList:
                zMatInput = '{:<4}'.format(list(waterAtom.keys())[0])
                for entry in list(waterAtom.items())[1:]:
                    if isinstance(entry[1], str):
                        zMatInput += '{:>4}{: >8}'.format(entry[0], entry[1])
                    else:
                        zMatInput += '{:>4}{: >8.2f}'.format(entry[0], entry[1])
                print(zMatInput, file=output)
            # Print out the variables section
            print('', file=output)
            # Enter initial variables
            for var, inVal in self.optVar.items():
                print('{:<8}{:>6.2f}'.format(var, inVal), file=output)
            print('\n\n', file=output)


    def writeCoords(self, geometry, atomIDs, name='coords'):

        with open('{}Int{}_{}.com'.format(self.siteType, self.atomID, name), 'w') as output:
            print('%Chk={}Int{}_{}'.format(self.siteType, self.atomID, name), file=output)
            print('%NProcShared=24', file=output)
            print('%Mem=61000MB', file=output)
            print('#P HF/6-31G(d) Opt(MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)
            print('Water {} interaction for {}\n'.format(self.siteType, self.atomID), file=output)
            print('0 1', file=output)
            # Print original molecular geometry
            for atomInd, el in enumerate(geometry):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format((atomIDs[atomInd]), el[:]), file=output)
            for dInd, dAtom in enumerate(self.dummyAtoms):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x' + str(dInd+1), dAtom[:]), file=output)
            for waterAtom in zip(['Ow', 'H1w', 'H2w'],[self.waterO, self.waterH1, self.waterH2]):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(waterAtom[0], waterAtom[1][:]), file=output)
            print('\n\n', file=output)


class DonorInt(InteractionSite):

    def waterPosition(self):

        # Position the oxygen off the donor H
        self.waterO = self.coords - self.bBasis[2]*2

        angles = [-(np.pi-self.angleHOH)/2, -(np.pi+self.angleHOH)/2]

        # Rotate standard basis by desired angles around the y axis
        rotOne = rotationY(angles[0], np.array([1., 0., 0.]))
        rotTwo = rotationY(angles[1], np.array([1., 0., 0.]))

        # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
        bPx = self.bBasis.transpose()
            # Transform the rotation vectors for the water H's to the donor basis, scale, and add to the water O
        self.waterH1 = self.waterO - np.matmul(bPx, rotOne)*self.bondOH
        self.waterH2 = self.waterO - np.matmul(bPx, rotTwo)*self.bondOH

    def dummyPosition(self):

        # Define dummy atoms for angles from the H donor
        dOne = self.coords + self.bBasis[0]
        dTwo = self.coords + self.bBasis[1]
        self.dummyAtoms = [dOne, dTwo]


    def idealzMat(self):

        # Currently just copied and pasted in to place
        # For ideal water O has one opt var and need to calculate angles and dihedrals
        OHx1 = gg.atomAngle(self.waterO, self.coords, self.dummyAtoms[0])
        OHx1x2 = gg.atomDihedral(self.waterO, self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        waterOzMat = {'Ow': numAtoms+3, self.atomID: 'rDO', 'x1': OHx1, 'x2': OHx1x2}

        # For water H geom; both r: bondOH; both ang: donor H and diheds: to same dummy and  left to opt
        Hw1A = (180 - 104.52/2.)
        waterH1zMat = {'H1w': numAtoms+4, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H1wOHx'}
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H2wOHx'}

        # Calculate initial values for opt variables
        H1wOHx = gg.atomDihedral(self.waterH1, self.waterO, self.coords, self.dummyAtoms[1])
        H2wOHx = gg.atomDihedral(self.waterH2, self.waterO, self.coords, self.dummyAtoms[1])
        self.optVar = {'rDO': 2.00, 'H1wOHx': H1wOHx, 'H2wOHx': H2wOHx}

        # Set list for writing the Z matrix section
        self.zMatList = [waterOzMat, waterH1zMat, waterH2zMat]

    def maxIntzMat(self):

        # Currently just copied and pasted in to place
        # For water O has three opt vars and calculates initial values
        waterOzMat = {'Ow': numAtoms+3, self.atomID: 'rDO', 'x1': 'OHx1', 'x2': 'OHx1x2'}
        OHx1 = gg.atomAngle(self.waterO, self.coords, self.dummyAtoms[0])
        OHx1x2 = gg.atomDihedral(self.waterO, self.coords, self.dummyAtoms[0], self.dummyAtoms[1])

        # For water H geom; both r: bondOH; angle of first to H; second to water angleHOH; do dihedrals to first dummy
        Hw1A = (180 - 104.52/2.)
        waterH1zMat = {'H1w': numAtoms+4, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H1wOHx'}
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H2wOHx'}

        H1wOHx = gg.atomDihedral(self.waterH1, self.waterO, self.coords, self.dummyAtoms[1])
        H2wOHx = gg.atomDihedral(self.waterH2, self.waterO, self.coords, self.dummyAtoms[1])

        self.optVar = {'rDO': 2.00, 'OHx1': OHx1, 'OHx1x2': OHx1x2, 'H1wOHx': H1wOHx, 'H2wOHx': H2wOHx}
        self.zMatList = [waterOzMat, waterH1zMat, waterH2zMat]


class AcceptorInt(InteractionSite):

    def waterPosition(self):

        self.waterH1 = self.coords - self.bBasis[2]*2
        # Position O bond distance away from the H
        self.waterO = self.waterH1 - self.bBasis[2]*self.bondOH

        # For second OH bond the angle will be angleHOH - 90
        rot = rotationY(-(self.angleHOH - 0.5*np.pi), np.array([1., 0., 0.]))
        # Construct transition matrix from standard basis to donor basis (inv is transpose). Order is matched to rotation done (would be rotating b2 around b3); making them b1 and b2 respectively
        bPx = self.bBasis.transpose()
        # Transform the rotation vectors for the second water H to the donor basis, scale, and add to the water O
        self.waterH2 = self.waterO - np.matmul(bPx, rot)*self.bondOH

    def dummyPosition(self):

        # Define dummy atoms for zmatrix definition
        dOne = self.coords + self.bBasis[0]
        dTwo = self.coords + self.bBasis[1]
        dThree = self.waterH1 + self.bBasis[0]
        self.dummyAtoms = [dOne, dTwo, dThree]

    def idealzMat(self):

        # Currently just copied and pasted in to place
          # Define interacting H: rAh to Acceptor to opt; angle and dihed to dummy atoms (fixed)
        HAx1 = gg.atomAngle(self.waterH1, self.coords, self.dummyAtoms[0])
        HAx1x2 = gg.atomDihedral(self.waterH1, self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        waterH1zMat = {'H1w': numAtoms+3, self.atomID: 'rAH', 'x1': HAx1, 'x2': HAx1x2}

        # Second attempt trying to maintain linear interaction
        OAng = gg.atomAngle(self.waterO, self.waterH1, self.dummyAtoms[0])
        ODihed = gg.atomDihedral(self.waterO, self.waterH1, self.dummyAtoms[0], self.coords)
        waterOzMat = {'Ow': numAtoms+4, 'H1w': self.bondOH, 'x2': OAng, self.atomID: ODihed}

        # Define 2nd H with r: OH bond distance to O; angle to Acceptor and dihed to dummy (left to opt)
        H2Ang = gg.atomAngle(self.waterH2, self.waterO, self.coords)
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: H2Ang, 'x2': 'HOAx'}

        # Calculate initial values for opt variables
        HOAx = gg.atomDihedral(self.waterH2, self.waterO, self.coords, self.dummyAtoms[1])
        self.optVar = {'rAH': 2.00, 'HOAx': HOAx}

        # Set list for writing the Z matrix section
        self.zMatList = [waterH1zMat, waterOzMat, waterH2zMat]


    def maxIntzMat(self):

        # Currently just copied and pasted in to place
        # Define interacting H: rAh to Acceptor to opt; angle and dihed to dummy atoms (opt)
        waterH1zMat = {'H1w': numAtoms+3, self.atomID: 'rAH', 'x1': 'HAx1', 'x2': 'HAx1x2'}

        ang = gg.atomAngle(self.waterO, self.waterH1, self.dummyAtoms[2])
        dihed = gg.atomDihedral(self.waterO, self.waterH1, self.dummyAtoms[2], self.coords)

        dist = gg.atomDist(self.waterO, self.dummyAtoms[2])
        ang = gg.atomAngle(self.waterO, self.dummyAtoms[2], self.waterH1)
        dihed = gg.atomDihedral(self.waterO, self.dummyAtoms[2], self.waterH1, self.coords)
        waterOzMat = {'Ow': numAtoms+4, 'x3': dist, 'H1w': ang, self.atomID: dihed}

        ang = gg.atomAngle(self.waterH2, self.waterO, self.coords)
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: ang, 'x2': 'HOAx'}

        # Start value for one dihedral seems to be out so calculate?
        HAx1Init = gg.atomAngle(self.waterH1, self.coords, self.dummyAtoms[0])
        HAx1x2Init = gg.atomDihedral(self.waterH1, self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        HOAxInit = gg.atomDihedral(self.waterH2, self.waterO, self.coords, self.dummyAtoms[1])
        self.optVar = {'rAH': 2.00, 'HAx1': HAx1Init, 'HAx1x2': HAx1x2Init, 'HOAx': HOAxInit}

        self.zMatList = [waterH1zMat, waterOzMat, waterH2zMat]



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
    ids = gg.atomIdentify(geomFile, numAtoms)


    # The remaining lines will contain the information for each donor/acceptor site
# Read in raw data; then for each site set up geometry - do initially without lone pairs
    siteIDs, siteList = [], []
    # Read in raw data; then for each site set up geometry - do initially without lone pairs
    for el in input[1:]:
        siteIDs.append(el.split()[0])
        if el.split()[3] == 'don':
            siteList.append(DonorInt(el.split()))
        elif el.split()[3] == 'acc':
            siteList.append(AcceptorInt(el.split()))

    for site in siteList:
        site.localGeom(geometry)
        site.bVectors()
        site.waterPosition()
        site.dummyPosition()

        site.idealzMat()
        site.writeZMat(geometry, ids, name='idealzMat')
        site.writeCoords(geometry, ids, name='idealCoords')
        site.maxIntzMat()
        site.writeZMat(geometry, ids, name='maxIntzMat')
        site.writeCoords(geometry, ids, name='maxIntCoords')
