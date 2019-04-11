import sys
import numpy as np
import gaussGeom as gg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# GS process
def gsProcess(xVec, *bVecs):
    sum = 0
    for bVec in bVecs:
        sum += (np.dot(xVec, bVec)/np.linalg.norm(bVec))*bVec
    bNew = xVec - sum
    return(bNew)

# Rotation around y axis
def rotationY(bOne, bTwo, bThree, feta, inVec):

    tX, tY, tZ = np.zeros(3), np.zeros(3), np.zeros(3)
    outVec = np.zeros(3)
    for bInd, bVec in enumerate([bOne, bTwo, bThree]):
        tX[bInd] = bVec[0]*np.cos(feta) + bVec[2]*np.sin(feta)
        tY[bInd] = bVec[1]
        tZ[bInd] = bVec[2]*np.cos(feta) - bVec[0]*np.sin(feta)

    outVec = np.array([np.dot(tX, inVec), np.dot(tY, inVec), np.dot(tZ, inVec)])

    return(outVec)

# Donor interaction - set H and C indexes
HInd = 10
CInd = 7

# Set geometry constants for TIP3P water molecule
bondOH = 0.9572
angleHOH = np.radians(104.52)
angles = [-(np.pi-angleHOH)/2, -(np.pi+angleHOH)/2]

# Pull optimised geometry from the file
inputFile = str(sys.argv[1])
geometry = gg.geomPull(inputFile, 42)
ids = gg.atomIdentify(inputFile, 42)

# Set standard basis
x1 = np.array([1., 0., 0.])
x2 = np.array([0., 1., 0.])
x3 = np.array([0., 0., 1.])

# set positions of H and C of molecule. Calculate vector b1 and use to calculate O position
HMol = geometry[HInd-1]
CMol = geometry[CInd-1]
b1 = HMol - CMol
b1 /= np.linalg.norm(b1)
OW = HMol + b1*2

# Find two other basis vectors to b1 using cross products (GS process alterantive which works but was unnecessary as orthogonal basis already created)
b2 = np.cross(HMol, CMol)
b2 /= np.linalg.norm(b2)
b3 = np.cross(b2, b1)
b3 /= np.linalg.norm(b3)

# Define dummy atoms for angles from the H donor
dOne = HMol + b2
dTwo = HMol + b3

# Calculate the two O-H bond vectors
OHOne = rotationY(b2, b3, b1, angles[0], b2)# - OW
OHTwo = rotationY(b2, b3, b1, angles[1], b2)

trialRot = rotationY(x1, x2, x3, angles[0], x1)
trialRottwo = rotationY(x1, x2, x3, angles[1], x1)
#OHTwo = OHOne*np.array([-1, 1, 1])

# Scale to correct length
#HWOne = OW + bondOH*OHOne
#HWTwo = OW + bondOH*OHTwo


# Inverse is transpose
bPx = np.array([b2, b3, b1]).transpose()
# Transform to new basis
HWOne = np.matmul(bPx, trialRot)*bondOH + OW
HWTwo = np.matmul(bPx, trialRottwo)*bondOH + OW


# Set plot up
#fig = plt.figure()
#ax = fig.gca(projection='3d')

# Plot for vectors
#range = np.linspace(0, 1, 20)
#labs = ['b1', 'b2', 'b3', 'H1', 'H2','x1', 'x2', 'x3']
#for ind, bVec in enumerate([b1, b2, b3, HOne, HTwo, x1, x2, x3]):
#    ax.scatter3D(bVec[0]*range, bVec[1]*range, bVec[2]*range, label=labs[ind])

# Plot points
#names = ['dOne', 'dTwo', 'HMol', 'CMol', 'OWater', 'HWOne', 'HWTwo']
#for ind, atom in enumerate([dOne, dTwo, HMol, CMol, OW, HWOne, HWTwo]):
#    ax.scatter3D(atom[0], atom[1], atom[2], label=names[ind])
#
#plt.legend()
#plt.show()

with open('donorTry.com', 'w') as output:
    print('%Chk=donorTry.com', file=output)
    print('%NProcShared=24', file=output)
    print('%Mem=61000MB', file=output)
    print('#P HF/6-31G(d) Opt(Z-matrix,MaxCycles=100) Geom(PrintInputOrient) SCF(Conver=9) Int(Grid=UltraFine)\n', file=output)
    print('donor set up \n', file=output)
    print('0 1', file=output)
    # Print original molecular geometry
    for atomInd, el in enumerate(geometry):
        print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format((ids[atomInd]), el[:]), file=output)
    print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x1', dOne[:]), file=output)
    print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('x2', dTwo[:]), file=output)
    print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('Ow', OW[:]), file=output)
    print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('H1w', HWOne[:]), file=output)
    print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format('H2w', HWTwo[:]), file=output)
    print('\n\n', file=output)

#OHTwo = rotation(x1, x2, x3, angles[1], x1) - O

#if __name__ == '__main__':
#
#    # Variables for TIP3P water molecule geometry
#    bondOH = 0.9572
#    angleHOH = np.radians(127.74)
#
#    inputFile = str(sys.argv[1])
#
#    # Pull optimised geometry from the file
#    coordinates = gg.geomPull(inputFile, 42)
#    ids = gg.atomIdentify(inputFile, 42)



