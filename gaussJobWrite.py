import sys
import argparse
import numpy as np

# Script which write the input file for a gaussian calculation
# Want input arguments of the job type
# Modredundant output for the scan

usage = "usage: %(prog)s [fileName] [args]"
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("fileName", nargs=1, type=str, help="The input .com file without the .com extension")
parser.add_argument("-j", "--job", dest="jobType", nargs='*', type=str, default=['Opt', 'Freq'])
parser.add_argument("-g", "--geom", dest="geom", type=str, default='file')
parser.add_argument("-m", "--method", dest="method", nargs=1, type=str, default=['M062X'],
                    help="The method to be used, give the correct gaussian keyword")
parser.add_argument("-b", "--basis", dest="basisSet", nargs=1, type=str, default=['6-311G(d,p)'],
                    help="The basis set to be used, give the correct gaussian keyword")
parser.add_argument("--mr", "--mod", dest="modRed", action='store_true',
                    help="Flag whether to expect moderedundant input or not, set to true for input")
parser.add_argument("--smd", dest="smd", action='store_true',
                    help="Flag whether to include SMD keyword for solvation or not, set to true for input")
parser.add_argument("-p", dest="preset", nargs=1, type=int,
                    help="Preset flag to set required prcoessors and mem")

args = parser.parse_args()

# Initially set skipConnectivity flag to false
skipConnectivity = False

# Sets filename and removes '.com' if present at the end of the name
fileName = args.fileName[0]
if fileName[-4:] == '.com':
    fileName = fileName[:-4]

# Set the job method and type keywords for the new input file
jobMethod = args.method[0]+'/'+args.basisSet[0]

# Set the job type and
jobType = ''
jobTitle = fileName
for jT in args.jobType:
    if jT == 'Opt':
        if args.modRed == True:
            jobType += 'Opt(ModRedundant) '
        else: jobType += 'Opt '
    if jT == 'Freq':
        jobType += 'Freq '
    if jT == 'TS':
        jobType += 'Opt(TS,NoEigen,RCFC) Freq '
    if jT == 'scan':
        jobType += 'Opt(ModRedundant,MaxCycles=100) '
        args.modRed = True
    jobTitle = jobTitle + ' ' + jT

# Read in original .com input file
try:
    with open(fileName + '.com', 'r') as inputFile:
        inputRaw = inputFile.read().splitlines()
except IOError:
    print("Error opening .com file", sys.stderr)

# Tracks section breaks to identify input sections in file. Multiple blank lines should signify EOF only
sections = []
for linInd in range(len(inputRaw)-1):
    if inputRaw[linInd] == '':
        sections.append(linInd)
        if inputRaw[linInd + 1] == '':
            break
    if 'connectivity' in inputRaw[linInd]:
        skipConnectivity = True
ind = 1

# Adds SCRF command and SMD for solvation in water if flag raised at input
if args.smd == True:
    jobType += 'SCRF(SMD) '
# Set the job Spec up with standard convergence criteria
jobSpec = '#P ' + jobMethod + ' ' + jobType + ' SCF(Conver=9) Int(Grid=UltraFine)'

# Sets charges + multiplicity and/or molecular geometry from original file
if args.geom == 'chk':
    jobSpec += ' Geom(Check) Guess(Read)'
    moleculeGeom = [inputRaw[sections[ind]+1]]
    ind += 1
if args.geom == 'file':
    moleculeGeom = inputRaw[sections[ind]+1:sections[ind+1]]
    ind += 1
if args.geom == 'allchk':
    jobSpec += ' Geom(AllCheck) Guess(Read)'

# Omits connectivty block if present
if skipConnectivity == True:
    ind += 1

# Sets modredundant input from next section or user input
if args.modRed == True:
    if ind == (len(sections)-1):
        modRedundant = input("Enter modRedundant input (csv for multiple lines):").split(',')
    else:
        modRedundant = inputRaw[sections[ind]+1:sections[ind+1]]

# Parses in presets and sets variables from that?

# Writes new .com file
with open(fileName+'.com', 'w+') as output:
    print('%Chk=' + fileName, file=output)
    print('%NProcShared=12', file=output)
    print('%Mem=36000MB', file=output)
    print(jobSpec + '\n', file=output)
    if args.geom != 'allchk':
        print(jobTitle + '\n', file=output)
        for el in moleculeGeom:
            print(el, file=output)
    if args.modRed == True:
        print('', file=output)
        for el in modRedundant:
            print(el, file=output)
    print('\n\n', file=output)

#print('%Chk=' + fileName)
#print('%NProcShared=12')
#print('%Mem=36000MB')
#print(jobSpec + '\n')
#if args.geom != 'allchk':
#    print(jobTitle + '\n')
#    for el in moleculeGeom:
#        print(el)
#if args.modRed == True:
#    print('')
#    for el in modRedundant:
#        print(el)
#print('\n\n')


