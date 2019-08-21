import sys
import numpy as np
import gaussGeom as gg


def tripleProduct(nBonds):

    """Function which calculates the triple product for three vectors)

        Parameters:
         nBonds: List of three vectors

        Returns:
         tripleProd: float - triple product of the three
    """

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

    """Function which applies rotations around the x, y and z axis (in that order; in the standard basis) by different input angles (fetaX, fetaY, fetaZ; where the angle is the rotation is around the x, y, or z, respectively)

        Parameters:
         fetaX: float - angle (radians) of rotation around the x axis
         fetaY: float - angle (radians) of rotation around the y axis
         fetaZ: float - angle (radians) of rotation around the z axis
         inVec: numpy array (dim: nx3) - the input vectors to be rotated

        Return:
         outVec: numpy array (dim: nx3) - the rotated vector for each input vector
    """

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

    """Parent class for interaction site objects for each water-residue interaction to be set up.

        Attributes:
            atomID: str - ID (QM atom type + index) of the interaction site
            atomInd: int - index of the interaction site
            inverse: bool - flag whether inverse set up is required
            lonepairs: int - number of lone pairs on the interaction site
            neighbourInd: List of ints - molecular indexes for the covalently bonded neighbours of the sites
            siteType: str ['don'/'acc'] - the interaction type to be set up
            bBasis: numpy array (dim: 3 x 3) - orthogonal basis set used to set up the water and dummy atom positions

        Methods:
         planarCheck(self)
         localGeom(self, geometry)
         transformBasis(self, transformMat=None, fetaX=0, fetaY=0, fetaZ=0, rotate=False)
         setPositions(self)
         writeOutput(self, geometry, atomIDs, format='zMat', extraID='')
    """

    #tip3P geometry
    bondOH = 0.9572
    angleHOH = np.radians(104.52)

    def __init__(self, raw, inv, lin):

        """Constructor to initialise InteractionSite object.

            An interaction site object is created for each set up required. This can be multiple set ups for a single interaction site. The interactions which are to be created are listed within the input text file with each line an interaction site within the molecule. The 'raw' input is each of these lines. File format:
                    atomType atomInd neighbourInds(csv) siteType [inv] [lp:X]
                Where for lone pair input, X is the number of lone pairs.

        Parameters:
         raw: str - raw input for the interaction site, detailing the site ID, index and type; neighbour indexes; lone pair or inverse set up information.
        """

        # Set the id index of the target mol
        self.atomID = raw[0] + raw[1]
        self.atomInd = int(raw[1])

        # Set inverse and linear set up flags
        self.inverse = inv
        self.linear = lin

        # Test to see if lone pairs set up needed
        if 'lp' in raw[-1]:
            self.lonePairs = int(raw[-1][-1])
        else:
            self.lonePairs = 0

        # Set the list of covalently bonded neighbours
        self.neighbourInd = []
        for nB in raw[2].split(','):
            self.neighbourInd.append(int(nB))

        # Assign the type of site to be set up
        self.siteType = str(raw[3])


    def planarCheck(self):

        """Class method which checks whether the bonds are linear (two neighbours) or lie within a single plane (three neighbours).

        If the bonds are planar/linear then b1 cannot be set using centroid of the bonds so set as the cross product of two neighbour bonds
        For three neighbours, the attribute self.inverse is set to true if the triple product of the neighbour bonds lie within a larger inverse tolerence so that water positions either side of the bonding plane are trialled. If the bonds are (within planarTol) of planar True is returned.
        For two neighbours, if the two neghbour bonds are (within planarTol) of linear True is returned.

        Returns:
         Bool - Result of planarity/linearity check
        """
        # Set tolerence for planar/linear check and looser tolerence to check if inverse set up required
        invTol = 2
        planarTol = 1e-02

        # Tests planarity of 3 neighbour bonds using triple product or if inverse set up is required
        if len(self.neighbourBonds) == 3:
            self.inverse = ((abs(tripleProduct(self.neighbourBonds))) < invTol)
            return(abs(tripleProduct(self.neighbourBonds)) < planarTol)

        # Tests linearity of 2 neighbour bonds using triple product
        elif len(self.neighbourBonds) == 2:
            return(abs(np.dot(self.neighbourBonds[0], self.neighbourBonds[1])) < planarTol)

        # Returns False if any other number of bonds
        else:
            return(False)


    def localGeom(self, geometry):

        """Class method which checks the local geometry of the interaction site and sets up an orthogonal basis set to use in placing dummy atoms and water

        Class attributes set:
         bBasis: numpy array (dim: 3 x 3) - orthogonal basis set used to set up the water and dummy atom positions. The basis is set so that bx lies within the bonding plane (2 neighbours) and by lies orthogonal to the bonding plane. [NB: If there are 3 or more neighbours then order of bx/by does't matter)]. The interaction vector is in the direction of bz, which lies furthest away from the neighbour bonds to postion the water molecule in the most space.

        Parameters:
         geometry: numpy array (dim: nAtoms x 3) - x, y, z coordinates of each atom in the molecule

        """

        # Set the coordinates of the covalently bonded neighbours
        self.coords = geometry[self.atomInd-1]
        neighbours = []
        for nbInd in self.neighbourInd:
            neighbours.append(geometry[nbInd-1])

        # Calculate bond vectors
        self.neighbourBonds = neighbours - self.coords

        # Check planarity/linearity of neighbour bonds and set b1 basis vector accordingly (True: cross product; False: centroid of neighbours bonds)
        if (self.planarCheck()):
            bz = np.cross(self.neighbourBonds[0], self.neighbourBonds[1])
        else:
            bz = np.sum(self.neighbourBonds, axis=0)/len(self.neighbourInd)
        bz /= np.linalg.norm(bz)

        # Find orthonormal basis from bz using cross products
        bx = np.cross(self.coords, bz)
        bx /= np.linalg.norm(bx)
        by = np.cross(bx, bz)
        by /= np.linalg.norm(by)

        # For two neighbour bonds, test which of bx/y lie orthogonal (should be y) to the bonding plane and which parallel (should be x), and set the basis vector order accordingly.
        if (len(self.neighbourInd) == 2) and (abs(tripleProduct([self.neighbourBonds[0], self.neighbourBonds[1], by])) < 1e-02):
            self.bBasis = np.array([by, bx, bz])
        else:
            self.bBasis = np.array([bx, by, bz])


    def transformBasis(self, transformMat=None, fetaX=0, fetaY=0, fetaZ=0, rotate=False):

        """Class method which rotates and updates the basis vectors (self.bBasis) by a linear transformation (w.r.t the standard basis) of rotations of angles fetaX, fetaY and fetaZ around the corresponding axes

        Parameters:
         fetaX: float - angle (radians) of rotation around the x axis
         fetaY: float - angle (radians) of rotation around the y axis
         fetaZ: float - angle (radians) of rotation around the z axis
        """

        # Calculates linear transformation T w.r.t standard basis (C)
        standardBasis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if rotate == True:
            transformMat = totalRot(fetaX, fetaY, fetaZ, standardBasis)

        # Calculate transformation matrix w.r.t site basis (B) (using P transpose as transition matrix from C to B)
        totalTransform = np.matmul(self.bBasis.transpose(), transformMat)

        # Apply transformation to basis vectors to update to new rotated basis
        self.bBasis = np.matmul(totalTransform, self.bBasis)


    def setPositions(self):

        """Class method which set the positions of the water molecule and the dummy atoms.

        The water position is set by shifting the water molecule gemetry from the location (coordinates) of the interaction site. Self.watergeom is set by the child class dependning on the interaction type (acc/don).
        Dummy atom positions call the child class methods to set the positions.
        """

        # Set water positions from the acceptor/donor atom
        self.waterPos = self.coords - np.matmul(self.waterGeom, self.bBasis)

        # Calculate the dummy atom positions (donor doesn't actually need d3)
        self.dummyAtoms = self.dummySetUp()


    def writeOutput(self, geometry, atomIDs, format='zMat', extraID=''):

        """Class method which writes a gaussian Optimisation input file for the water/residue interaction in either mixed (cartesian/zMatrix) or full cartesian geometry.

        Parameters:
         geometry: Numpy array (dim: nAtoms x 3) - x, y, z coordinates of each atom in the molecule
         atomIDs: List of str - atomIDs of the atoms in the molecule
         format: str - (default: zMat) specifies if the water position is written in zMatrix or cartesian form and can contain extra set up/format information
         extraID: str - Extra information used to identify the interacton set up (e.g. lonepairs or inverse set-up)
        """

        # Set name for file - need unique ID
        fileID = self.atomID + extraID
        if self.siteType == 'acc':
            if self.nNeighbours == 4:
                fileID += '_' + str(self.neighbourInd[0])

        # Create .com file and write job spec, title, and charge/multiplicity
        with open('{}Int{}_{}.com'.format(self.siteType, fileID, format), 'w') as output:
            print('%Chk={}Int{}_{}'.format(self.siteType, fileID, format), file=output)
            print('%NProcShared=12', file=output)
            print('%Mem=46000MB', file=output)

            if 'zMat' in format:
                print('#P HF/6-31G(d) Opt(Z-Matrix,MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)
            elif 'maxInt' in format:
                print('#P HF/6-31G(d) Opt(ModRedundant,MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)
            else:
                print('#P HF/6-31G(d) Opt(MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)

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
                            zMatInput += '{:>4}{: >10}'.format(entry[0], entry[1])
                        else:
                            zMatInput += '{:>4}{: 10.4f}'.format(entry[0], entry[1])
                    print(zMatInput, file=output)
                # Print out the variables section
                print('', file=output)
                # Enter initial variables
                for var, inVal in self.optVar.items():
                    print('{:<8}{:>8.4f}'.format(var, inVal), file=output)

            # Write water geometry in coordinates format - both acc/don save in order H1, O, H2
            else:
                for waterAtom in zip(['Ow', 'H1w', 'H2w'],[self.waterPos[1], self.waterPos[0], self.waterPos[2]]):
                    print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}\n'.format(waterAtom[0], waterAtom[1][:]), file=output)

            # Addition of maxInt one with bonds and angle frozen, index starts after molecule and dummy atoms
            if 'maxInt' in format:
                H1Ind = atomInd+dInd+1
                print('{} {} F'.format(H1Ind, H1Ind+1), file=output)
                print('{} {} F'.format(H1Ind+1, H1Ind+2), file=output)
                print('{} {} {} F'.format(H1Ind, H1Ind+1, H1Ind+2), file=output)

            print('\n\n', file=output)


class DonorInt(InteractionSite):

    """ Child class of Interaction Site -  Sets variables and water position for a donor interaction

        Class attributes:
         self.waterGeom: numpy array (dim: 3 x 3) - vectors of the 3 tip3p water atoms defining the relative geometry
         self.optVar: dict - the variables to be optimised
         self.zMatList: list of dicts - the zMatrix configurations (R, A, D) for the tip3p water atoms

        Methods:
         dummySetUp(self)
         idealzMat(self, numAtoms)
         maxIntzMat - still used?
    """

    def __init__(self, raw, inv=False, lin=True):

        """Constructor to intiialise donor class object.

            Parent class IntractionSite is called with raw input to inherit any attributes and the donor interaction water geometry is set.

            Parameters:
             raw: str - interaction site raw information passed to parent InteractioSite constructor
             inv: bool - boolean flag for whether inverse set up is wanted or not
             lin: bool - boolean flag for whether linear set up is wanted or not. Default setup is linear only.
        """

        # Calls parrent class ocnstructor
        InteractionSite.__init__(self, raw, inv, lin)

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

        """Class method which sets the two dummy atom positions.

            For a donor interaction the two dummy atoms are positioned: basis vector bx from the interaction site, and basis vector by from the interaction site.

            Returns:
            List of
            dOne: numpy array 1 x 3 - x, y, z coordinates of dummy atom one
            dTwo: numpy array 1 x 3 - x, y, z coordinates of dummy atom two
        """

        dOne = self.coords + self.bBasis[0]
        dTwo = self.coords + self.bBasis[1]
        return([dOne, dTwo])


    def idealzMat(self, numAtoms):

        """Class method which sets up the zMatrix for an 'idealised' interaction between
            the tip3p water and the donor atom.

            The zMatrix set up is calculated with the interaction distance (rDO) and the dihedral rotation
            of the water molecule (HODx/-HODx) set to be optimised variables.

            Class attributes set:
             self.optVar: dict - the variables to be optimised
             self.zMatList: list of dicts - the zMatrix configurations (R, A, D) for the tip3p water atoms

            Parameters:
             numAtoms: int - the number of atoms in the molecule
        """

        # Define water O z-matrix configuration
        OHx1 = gg.atomAngle(self.waterPos[1], self.coords, self.dummyAtoms[0])
        OHx1x2 = gg.atomDihedral(self.waterPos[1], self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        waterOzMat = {'Ow': numAtoms+3, self.atomID: 'rDO', 'x1': OHx1, 'x2': OHx1x2}

        # Define water H z-matrix configurations
        HOD = (180 - 104.52/2.)
        waterH1zMat = {'H1w': numAtoms+4, 'Ow': self.bondOH, self.atomID: HOD, 'x2': 'H1ODx'}
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: HOD, 'x2': 'H2ODx'}

        # Calculate and set initial values for opt variables
        HODx = gg.atomDihedral(self.waterPos[0], self.waterPos[1], self.coords, self.dummyAtoms[1])
        self.optVar = {'rDO': 2.00, 'H1ODx': HODx, 'H2ODx': -HODx}

        # Set list for writing the Z matrix section
        self.zMatList = [waterOzMat, waterH1zMat, waterH2zMat]


class AcceptorInt(InteractionSite):

    """ Child class of Interaction Site -  Sets variables and water position for an acceptor interaction

        Class attributes:
         self.waterGeom: numpy array (dim: 3 x 3) - vectors of the 3 tip3p water atoms defining the relative geometry
         self.optVar: dict - the variables to be optimised
         self.zMatList: list of dicts - the zMatrix configurations (R, A, D) for the tip3p water atoms

        Methods:
         dummySetUp(self)
         idealzMat(self, numAtoms)
         maxIntzMat - still used?
    """

    def __init__(self, raw, inv=False, lin=True, nNeighbours=None):

        """Constructor to intiialise acceptor class object.

            Parent class IntractionSite is called with raw input to inherit any attributes and the acceptor interaction water geometry is set.

            Parameters:
             raw: str - interaction site raw information passed to parent InteractioSite constructor
             inv: bool - boolean flag for whether inverse set up is wanted or not
             lin: bool - boolean flag for whether linear set up is wanted or not. Default setup is linear only.
             nNeighbours: int (default: None) - Used to recognise when 4 covalently bonded neighbours but only three are used for the set up and an inverse set up is required.
        """

        # Instantiate parent class
        InteractionSite.__init__(self, raw, inv, lin)

        if nNeighbours == None:
            self.nNeighbours = len(self.neighbourInd)
        else:
            self.nNeighbours = nNeighbours
#
#        if nNeighbours == 4:
#            self.inverse = True

        # Calculate water geometry w.r.t standard basis and origin as acceptor atom
        rot = totalRot(0, -(self.angleHOH - 0.5*np.pi), 0, np.array([1., 0., 0.]))
        H1 = np.array([0, 0, 2])
        O = H1 + np.array([0, 0, self.bondOH])
        H2 = O + rot*self.bondOH
        self.waterGeom = np.array([H1, O, H2])


    def dummySetUp(self):

        """Class method which sets the three dummy atom positions.

            For an acceptor interaction the three dummy atoms are positioned: basis vector bx from the interaction site; basis vector by from the interaction site, and basis vector bx from H1 (nteracting H) of the tip3p water.

            Returns:
            List of
                dOne: numpy array 1 x 3 - x, y, z coordinates of dummy atom one
                dTwo: numpy array 1 x 3 - x, y, z coordinates of dummy atom two
                dThree: numpy array 1 x 3 - x, y, z coordinates of dummy atom three
        """

        dOne = self.coords + self.bBasis[0]
        dTwo = self.coords + self.bBasis[1]
        dThree = self.waterPos[0] + self.bBasis[0]
        return([dOne, dTwo, dThree])


    def idealzMat(self, numAtoms):

        """Class method which sets up the zMatrix for an 'idealised' interaction between
            the tip3p water and the acceptor atom.

            The zMatrix set up is calculated with the interaction distance (rAH); the angle of the interaction (OHx) and the dihedral rotation of the water molecule (HOAx) set to be optimised variables.

            Class attributes set:
             optVar: dict - the variables to be optimised
             zMatList: list of dicts - the zMatrix configurations (R, A, D) for the tip3p water atoms

            Parameters:
             numAtoms: int - the number of atoms in the molecule
        """

        # Define interacting water H z-matrix configuration
        HAx1 = gg.atomAngle(self.waterPos[0], self.coords, self.dummyAtoms[0])
        HAx1x2 = gg.atomDihedral(self.waterPos[0], self.coords, self.dummyAtoms[0], self.dummyAtoms[1])
        waterH1zMat = {'H1w': numAtoms+3, self.atomID: 'rAH', 'x1': HAx1, 'x2': HAx1x2}

        # Define water O z-matrix configuration
        OHx = gg.atomAngle(self.waterPos[1], self.waterPos[0], self.dummyAtoms[1])
        OHxA = gg.atomDihedral(self.waterPos[1], self.waterPos[0], self.dummyAtoms[1], self.coords)
        waterOzMat = {'Ow': numAtoms+4, 'H1w': self.bondOH, 'x2': 'OHx', self.atomID: OHxA}

        # Define non-interacting water H z-matrix configuration
        HOA = gg.atomAngle(self.waterPos[2], self.waterPos[1], self.coords)
        waterH2zMat = {'H2w': numAtoms+5, 'Ow': self.bondOH, self.atomID: HOA, 'x2': 'HOAx'}

        # Calculate and set initial values for opt variables
        HOAx = gg.atomDihedral(self.waterPos[2], self.waterPos[1], self.coords, self.dummyAtoms[1])
        self.optVar = {'rAH': 2.00, 'HOAx': HOAx, 'OHx': OHx}

        # Set list for writing the Z matrix section
        self.zMatList = [waterH1zMat, waterOzMat, waterH2zMat]


def inputParse(inputFile):

    """Function which parses the input lines from a file for each desired interaction to be set up.

    Input lines within a text file should be formatted:
        line 1:             geometry.log nAtoms
        line 2 onwards:     atomType atomInd neighbourInds(csv) siteType [lp:X] [inv]

        Where, the atom type is the QM atom type/name, atomInd: the QM atom index, neighbourInds: the QM atom indexes of the covalently bonded neighbours, siteType: values: 'acc' or 'don' only, describing the H-bond interaction type for the site, lp:X: optional input, where X is the number of lone pairs on the site, inv: optional input, denotes whether inverse set up is required or not.

        Example input:
            residueMP2.log 10
            C 1 2,3
            N 2 4,5 2 inv
        Lines can be commented out (#...) resulting in the parser ignoring them
    """

    # Create an acceptor/donor InteractionSite object for each water interaction
    with open(inputFile, 'r') as iFile:
        input = iFile.readlines()

    # Pull the geometry and atomIDs from the log file
    geomFile = input[0].split()[0]
    numAtoms = int(input[0].split()[1])

    geometry = gg.geomPulllog(geomFile, numAtoms)[0]
    ids = gg.atomIdentify(geomFile, numAtoms)

    # Empty list for interaction site object
    siteList = []

    for el in input[1:]:
        # Skip line if comment
        if el[0] != '#':

            # Test to see if lone pairs or if inverse set up needed
            inverse = ('inv' in el)
            if inverse == True:
                linear = ('lin' in el)
            else:
                linear = True

            # If donor then doesn't need anymore processing
            if el.split()[3] == 'don':
                siteList.append(DonorInt(el.split(), inv=inverse, lin=linear))

            elif el.split()[3] == 'acc':

                # If 4 neighbours then have to instantiate for each set of three and inverse set
                neighbours = el.split()[2].split(',')
                if len(neighbours) == 4:
                    for x in range(4):
                        nbThree = [neighbours[x%4] + ',' + neighbours[(x+1)%4] + ',' + neighbours[(x+2)%4]]
                        # Switch the original neighbour list for the three indexes in the input line
                        siteList.append(AcceptorInt(el.split()[:2] + nbThree + el.split()[3:], nNeighbours=4, inv=True, lin=False))
                else:
                    siteList.append(AcceptorInt(el.split(), inv=inverse, lin=linear))
    return(siteList, geometry, ids)


