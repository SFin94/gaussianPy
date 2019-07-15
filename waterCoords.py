import sys
import numpy as np
import gaussGeom as gg


# GS process
def gramScmidt(xVec, *bVecs):

    '''Function which calculates a new orthogonal basis vector by the Gram Schmidt process'''

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


def axisRot(feta, rotAxis, inVec):

    tX = np.array([np.cos(feta) + rotAxis[0]*rotAxis[0]*(1 - np.cos(feta)), rotAxis[0]*rotAxis[1]*(1 - np.cos(feta))-rotAxis[2]*np.sin(feta), rotAxis[0]*rotAxis[2]*(1 - np.cos(feta))+rotAxis[1]*np.sin(feta)])
    tY = np.array([rotAxis[1]*rotAxis[0]*(1 - np.cos(feta))+rotAxis[2]*np.sin(feta), np.cos(feta) + rotAxis[1]*rotAxis[1]*(1 - np.cos(feta)), rotAxis[1]*rotAxis[2]*(1 - np.cos(feta))-rotAxis[0]*np.sin(feta)])
    tZ = np.array([rotAxis[2]*rotAxis[0]*(1 - np.cos(feta))-rotAxis[1]*np.sin(feta), rotAxis[2]*rotAxis[1]*(1 - np.cos(feta))+rotAxis[0]*np.sin(feta), np.cos(feta) + rotAxis[2]*rotAxis[2]*(1 - np.cos(feta))])

    if inVec.ndim == 1:
        outVec = np.array([np.dot(tX, inVec), np.dot(tY, inVec), np.dot(tZ, inVec)])
    else:
        outVec = np.zeros(inVec.shape)
        for ind in range(inVec.shape[0]):
            outVec[ind] = [np.dot(tX, inVec[ind]), np.dot(tY, inVec[ind]), np.dot(tZ, inVec[ind])]
    return(outVec)



def totalRot(fetaX, fetaY, fetaZ, inVec):

    '''Function which applies rotations around the x, y and z axis (in that order; in the standard basis) by different input angles (fetaX, fetaY, fetaZ; where the angle is the rotation is around the x, y, or z, respectively)

        Parameters:
         fetaX: float - angle (radians) of rotation around the x axis
         fetaY: float - angle (radians) of rotation around the y axis
         fetaZ: float - angle (radians) of rotation around the z axis
         inVec: numpy array (dim: nx3) - the input vectors to be rotated

        Return:
         outVec: numpy array (dim: nx3) - the rotated vector for each input vector
        '''

    # Set up three arrays: these are the rows of the transformation matrix
    tX = np.array([np.cos(fetaZ)*np.cos(fetaY), np.cos(fetaZ)*np.sin(fetaY)*np.sin(fetaX) - np.sin(fetaZ)*np.cos(fetaX), np.cos(fetaZ)*np.sin(fetaY)*np.cos(fetaX) + np.sin(fetaZ)*np.sin(fetaX)])
    tY = np.array([np.sin(fetaZ)*np.cos(fetaY), np.sin(fetaZ)*np.sin(fetaY)*np.sin(fetaX) + np.cos(fetaZ)*np.cos(fetaX), np.sin(fetaZ)*np.sin(fetaY)*np.cos(fetaX) - np.cos(fetaZ)*np.sin(fetaX)])
    tZ = np.array([-np.sin(fetaY), np.cos(fetaY)*np.sin(fetaX), np.cos(fetaY)*np.cos(fetaX)])

    if inVec.ndim == 1:
        outVec = np.array([np.dot(tX, inVec), np.dot(tY, inVec), np.dot(tZ, inVec)])
    else:
        outVec = np.zeros(inVec.shape)
        for ind in range(inVec.shape[0]):
            outVec[ind] = [np.dot(tX, inVec[ind]), np.dot(tY, inVec[ind]), np.dot(tZ, inVec[ind])]
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
        self.atomID = raw[0] + raw[1]
        self.atomInd = int(raw[1])

        # Set the list of covalently bonded neighbours
        self.neighbourInd = []
        for nB in raw[2].split(','):
            self.neighbourInd.append(int(nB))

        # Assign the type of site to be set up
        self.siteType = str(raw[3])

    def localGeom(self, geometry):

        # Set the coordinates of the covalently bonded neighbours
        self.coords = geometry[self.atomInd-1]
        neighbours = []
        for nbInd in self.neighbourInd:
            neighbours.append(geometry[nbInd-1])

        # Calculate bond vectors
        self.neighbourBonds = neighbours - self.coords

        # Find centroid of the bonds and normalise for first basis vector
        b1 = np.sum(self.neighbourBonds, axis=0)/len(self.neighbourInd)
        b1 /= np.linalg.norm(b1)

        # Find orthonormal basis from b1 using cross products
        b2 = np.cross(self.coords, b1)
        b2 /= np.linalg.norm(b2)
        b3 = np.cross(b2, b1)
        b3 /= np.linalg.norm(b3)

        # Test the triple product of the neighbour bond vectors
        # If close to 0 then they lie in the same plane and switch b1 for the orthogonal b2 or b3
        tol = 1e-03
        if (len(self.neighbourInd) == 3) and (abs(tripleProduct(self.neighbourBonds))) < tol:
            if abs(tripleProduct([self.neighbourBonds[0], self.neighbourBonds[1], b2])) < tol:
                self.bBasis = np.array([b2, b1, b3])
            else:
                self.bBasis = np.array([b1, b3, b2])
            return(True)
        # Test if two neighbours whether they are linear, if so switch basis vectors
        elif (len(self.neighbourInd) == 2) and (abs(np.dot(self.neighbourBonds[0], self.neighbourBonds[1])) < tol):
            print('linear')
            if abs(np.dot(self.neighbourBonds[0], b2)) < tol:
                self.bBasis = np.array([b2, b1, b3])
            else:
                self.bBasis = np.array([b1, b3, b2])
            return(True)

        # Test which of b3/b2 are orthogonal to the plane of the bonds to define x and y (set y as orthogonal)
        elif (len(self.neighbourInd) == 2) and (abs(tripleProduct([self.neighbourBonds[0], self.neighbourBonds[1], b3])) < tol):
            self.bBasis = np.array([b3, b2, b1])

        else:
            self.bBasis = np.array([b2, b3, b1])
        return(False)


    def transformBasis(self, transformMat=None, fetaX=0, fetaY=0, fetaZ=0, rotate=False):

        ''' Function which rotates and updates the basis vectors by a linear transformation (w.r.t the standard basis) of rotations of angles fetaX, fetaY and fetaZ around the corresponding axes

            Parameters:
             fetaX: float - angle (radians) of rotation around the x axis
             fetaY: float - angle (radians) of rotation around the y axis
             fetaZ: float - angle (radians) of rotation around the z axis
        '''

        # Calculates linear transformation T w.r.t standard basis (C)
        standardBasis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if rotate == True:
            transformMat = totalRot(fetaX, fetaY, fetaZ, standardBasis)

        # Calculate transformation matrix w.r.t site basis (B) (using P transpose as transition matrix from C to B)
        totalTransform = np.matmul(self.bBasis.transpose(), transformMat)

        # Apply transformation to basis vectors to update to new rotated basis
        self.bBasis = np.matmul(totalTransform, self.bBasis)

    def setPositions(self):

        # Set water positions from the acceptor/donor atom
        self.waterPos = self.coords - np.matmul(self.waterGeom, self.bBasis)

        # Calculate the dummy atom positions (donor doesn't actually need d3)
        self.dummyAtoms = self.dummySetUp()


    def writeOutput(self, geometry, atomIDs, format='zMat', extraID=''):

        ''' Function which writes a gaussian input file with the constrained optimisation input in z matrix form

            Inputs:
             geometry: x, y, z coordinates of each atom in the molecule
             atomIDs: List of atomIDs in the molecule
             name: str - optional add on identifier for the input file (default: 'zMat')
        '''

        # Set name for file - need unique ID
        fileID = self.atomID + extraID
        if self.siteType == 'acc':
            if self.nNeighbours == 4:
                fileID += '_' + str(self.neighbourInd[0])
            elif self.inverse == True:
                fileID += '_inv'

        with open('{}Int{}_{}.com'.format(self.siteType, fileID, format), 'w') as output:
            print('%Chk={}Int{}_{}'.format(self.siteType, fileID, format), file=output)
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
            if 'zMat' in format:
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

            # Write in coordinates format
            else:
                for waterAtom in zip(['Ow', 'H1w', 'H2w'],[self.waterPos[1], self.waterPos[0], self.waterPos[2]]):
                    print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(waterAtom[0], waterAtom[1][:]), file=output)

            print('\n\n', file=output)


class DonorInt(InteractionSite):

    '''Child class of Interaction Site
        Sets variables and water position for a donor interaction
    '''

    def __init__(self, raw):

        InteractionSite.__init__(self, raw)

        # Calculate angles for rotation from basis to H positions and peform rotation for O-H bond vectors
        angles = [-(np.pi-self.angleHOH)/2, -(np.pi+self.angleHOH)/2]
        rotOne = totalRot(0, angles[0], 0, np.array([1., 0., 0.]))
        rotTwo = totalRot(0, angles[1], 0, np.array([1., 0., 0.]))

        # Only differ in negative/positive x coord so may be quicker to do one function call then calculate second

        # Set water geometry positions w.r.t standard basis and donor atom as origin
        O = np.array([0, 0, 2])
        H1 = O + rotOne*self.bondOH
        H2 = O + rotTwo*self.bondOH
        self.waterGeom = np.array([H1, O, H2])

    def dummySetUp(self):

        dOne = self.coords + self.bBasis[0]
        dTwo = self.coords + self.bBasis[1]
        return([dOne, dTwo])

    def idealzMat(self):

        # Currently just copied and pasted in to place
        # For ideal water O has one opt var and need to calculate angles and dihedrals
        OHx1 = gg.atomAngle(self.waterPos[1], self.coords, self.dummyAtoms[0])
        OHx1x2 = gg.atomDihedral(self.waterPos[1], self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        waterOzMat = {'Ow': numAtoms+3, self.atomID: 'rDO', 'x1': OHx1, 'x2': OHx1x2}

        # For water H geom; both r: bondOH; both ang: donor H and diheds: to same dummy and  left to opt
        Hw1A = (180 - 104.52/2.)
        waterH1zMat = {'H1w': numAtoms+4, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H1wOHx'}
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H2wOHx'}

        # Calculate initial values for opt variables
        H1wOHx = gg.atomDihedral(self.waterPos[0], self.waterPos[1], self.coords, self.dummyAtoms[1])
        H2wOHx = gg.atomDihedral(self.waterPos[2], self.waterPos[1], self.coords, self.dummyAtoms[1])
        self.optVar = {'rDO': 2.00, 'H1wOHx': H1wOHx, 'H2wOHx': H2wOHx}

        # Set list for writing the Z matrix section
        self.zMatList = [waterOzMat, waterH1zMat, waterH2zMat]

    def maxIntzMat(self):

        # Currently just copied and pasted in to place
        # For water O has three opt vars and calculates initial values
        waterOzMat = {'Ow': numAtoms+3, self.atomID: 'rDO', 'x1': 'OHx1', 'x2': 'OHx1x2'}
        OHx1 = gg.atomAngle(self.waterPos[1], self.coords, self.dummyAtoms[0])
        OHx1x2 = gg.atomDihedral(self.waterPos[1], self.coords, self.dummyAtoms[0], self.dummyAtoms[1])

        # For water H geom; both r: bondOH; angle of first to H; second to water angleHOH; do dihedrals to first dummy
        Hw1A = (180 - 104.52/2.)
        waterH1zMat = {'H1w': numAtoms+4, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H1wOHx'}
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: Hw1A, 'x2': 'H2wOHx'}

        H1wOHx = gg.atomDihedral(self.waterPos[0], self.waterPos[1], self.coords, self.dummyAtoms[1])
        H2wOHx = gg.atomDihedral(self.waterPos[2], self.waterPos[1], self.coords, self.dummyAtoms[1])

        self.optVar = {'rDO': 2.00, 'OHx1': OHx1, 'OHx1x2': OHx1x2, 'H1wOHx': H1wOHx, 'H2wOHx': H2wOHx}
        self.zMatList = [waterOzMat, waterH1zMat, waterH2zMat]


class AcceptorInt(InteractionSite):

    '''Child class of Interaction Site
        Sets variables and water position for an acceptor interaction
        '''

    def __init__(self, raw, nNeighbours=None, lp=0, inv=False):

        # Instantiate parent class
        InteractionSite.__init__(self, raw)

        if nNeighbours == None:
            self.nNeighbours = len(self.neighbourInd)
        else:
            self.nNeighbours = nNeighbours

        # Calculate water geometry w.r.t standard basis and origin as acceptor atom
        rot = totalRot(0, -(self.angleHOH - 0.5*np.pi), 0, np.array([1., 0., 0.]))
        H1 = np.array([0, 0, 2])
        O = H1 + np.array([0, 0, self.bondOH])
        H2 = O + rot*self.bondOH
        self.waterGeom = np.array([H1, O, H2])

        # Set additional acceptor type attributes
        self.lonePairs = lp
        self.inverse = inv


    def dummySetUp(self):

        dOne = self.coords + self.bBasis[0]
        dTwo = self.coords + self.bBasis[1]
        dThree = self.waterPos[0] + self.bBasis[0]
        return([dOne, dTwo, dThree])

    def idealzMat(self):

        # Currently just copied and pasted in to place
          # Define interacting H: rAh to Acceptor to opt; angle and dihed to dummy atoms (fixed)
        HAx1 = gg.atomAngle(self.waterPos[0], self.coords, self.dummyAtoms[0])
        HAx1x2 = gg.atomDihedral(self.waterPos[0], self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        waterH1zMat = {'H1w': numAtoms+3, self.atomID: 'rAH', 'x1': HAx1, 'x2': HAx1x2}

        # Second attempt trying to maintain linear interaction
        OAng = gg.atomAngle(self.waterPos[1], self.waterPos[0], self.dummyAtoms[0])
        ODihed = gg.atomDihedral(self.waterPos[1], self.waterPos[0], self.dummyAtoms[0], self.coords)
        waterOzMat = {'Ow': numAtoms+4, 'H1w': self.bondOH, 'x2': OAng, self.atomID: ODihed}

        # Define 2nd H with r: OH bond distance to O; angle to Acceptor and dihed to dummy (left to opt)
        H2Ang = gg.atomAngle(self.waterPos[2], self.waterPos[1], self.coords)
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: H2Ang, 'x2': 'HOAx'}

        # Calculate initial values for opt variables
        HOAx = gg.atomDihedral(self.waterPos[2], self.waterPos[1], self.coords, self.dummyAtoms[1])
        self.optVar = {'rAH': 2.00, 'HOAx': HOAx}

        # Set list for writing the Z matrix section
        self.zMatList = [waterH1zMat, waterOzMat, waterH2zMat]


    def maxIntzMat(self):

        # Currently just copied and pasted in to place
        # Define interacting H: rAh to Acceptor to opt; angle and dihed to dummy atoms (opt)
        waterH1zMat = {'H1w': numAtoms+3, self.atomID: 'rAH', 'x1': 'HAx1', 'x2': 'HAx1x2'}

        ang = gg.atomAngle(self.waterPos[1], self.waterPos[0], self.dummyAtoms[2])
        dihed = gg.atomDihedral(self.waterPos[1], self.waterPos[0], self.dummyAtoms[2], self.coords)

        dist = gg.atomDist(self.waterPos[1], self.dummyAtoms[2])
        ang = gg.atomAngle(self.waterPos[1], self.dummyAtoms[2], self.waterPos[0])
        dihed = gg.atomDihedral(self.waterPos[1], self.dummyAtoms[2], self.waterPos[0], self.coords)
        waterOzMat = {'Ow': numAtoms+4, 'x3': dist, 'H1w': ang, self.atomID: dihed}

        ang = gg.atomAngle(self.waterPos[2], self.waterPos[1], self.coords)
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: ang, 'x2': 'HOAx'}

        # Start value for one dihedral seems to be out so calculate?
        HAx1Init = gg.atomAngle(self.waterPos[0], self.coords, self.dummyAtoms[0])
        HAx1x2Init = gg.atomDihedral(self.waterPos[0], self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        HOAxInit = gg.atomDihedral(self.waterPos[2], self.waterPos[1], self.coords, self.dummyAtoms[1])
        self.optVar = {'rAH': 2.00, 'HAx1': HAx1Init, 'HAx1x2': HAx1x2Init, 'HOAx': HOAxInit}

        self.zMatList = [waterH1zMat, waterOzMat, waterH2zMat]


def inputParse(input):
    # Create an acceptor/donor interactionSite object for each water interaction site

    # Empty list for interaction site object
    siteList = []

    for el in input:
        # Skip line if comment
        if el[0] != '#':
            # If donor then doesn't need anymore processing
            if el.split()[3] == 'don':
                siteList.append(DonorInt(el.split()))

            elif el.split()[3] == 'acc':

                # If 4 neighbours then have to instantiate for each set of three and inverse set
                neighbours = el.split()[2].split(',')
                if len(neighbours) == 4:
                    for x in range(4):
                        nbThree = [neighbours[x%4] + ',' + neighbours[(x+1)%4] + ',' + neighbours[(x+2)%4]]
                        # Switch the original neighbour list for the three indexes in the input line
                        siteList.append(AcceptorInt(el.split()[:2] + nbThree + el.split()[3:], nNeighbours=4, inv=True))

                else:
                    # Set lonepairs and inverse if additional arguments
                    input = el.split()
                    if 'inv' in input:
                        inverse = True
                        input.remove('inv')
                    else: inverse = False
                    if len(input) > 4:
                        siteList.append(AcceptorInt(input[:-1], lp=int(input[-1]), inv=inverse))
                    else:
                        siteList.append(AcceptorInt(input, inv=inverse))
    return(siteList)


if __name__ == '__main__':

# Input file format: siteInd neighbourInds target lp(optional)
    with open(str(sys.argv[1]), 'r') as inputFile:
        input = inputFile.readlines()

    # Pull the geometry and atomIDs from the log file
    geomFile = input[0].split()[0]
    numAtoms = int(input[0].split()[1])

    geometry = gg.geomPulllog(geomFile, numAtoms)
    ids = gg.atomIdentify(geomFile, numAtoms)

    # Create an acceptor/donor interactionSite object for each water interaction site
    siteList = inputParse(input[1:])

    for site in siteList:
        planar = site.localGeom(geometry)
        site.setPositions()
        site.idealzMat()
        site.writeOutput(geometry, ids, format='idealzMat')

        if site.siteType == 'acc':

            # Test to see if linear or planar triple bonding plane, then set up reverse position
            if (planar == True) or (site.inverse == True):
                site.inverse = True  # Set for file ID extension
                site.bBasis = axisRot(np.radians(180), site.bBasis[0], site.bBasis)
                site.setPositions()
                site.idealzMat()
                site.writeOutput(geometry, ids, format='idealzMat')

            # Test to see if lone pair set up require
            if site.lonePairs == 2:

                site.bBasis = axisRot(np.radians(60), site.bBasis[0], site.bBasis)
                site.setPositions()
                site.idealzMat()
                site.writeOutput(geometry, ids, extraID='_lp1', format='idealzMat')
                site.bBasis = axisRot(np.radians(-120), site.bBasis[0], site.bBasis)
                site.setPositions()
                site.idealzMat()
                site.writeOutput(geometry, ids, extraID='_lp2', format='idealzMat')


