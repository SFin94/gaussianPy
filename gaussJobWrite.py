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
parser.add_argument("-p", dest="preset", nargs=1, type=int,
                    help="Preset flag to set required prcoessors and mem")

args = parser.parse_args()

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
        jobType += 'Opt '
    if jT == 'Freq':
        jobType += 'Freq '
    if jT == 'TS':
        jobType += 'Opt(TS,NoEigen,RCFC) Freq '
    if jT == 'scan':
        jobType += 'Opt(ModRedundant,MaxCycles=100) '
        args.modRed[0] = True
    jobTitle = jobTitle + ' ' + jT

# Assume file structure as
try:
    with open(fileName + '.com', 'r') as inputFile:
        inputRaw = inputFile.read().splitlines()
except IOError:
    print("Error opening .com file", sys.stderr)

# Tracks section breaks to identify input sections in file
section = []
for index, el in enumerate(inputRaw):
    if el == '':
        section.append(index)
    if 'connectivity' in el:
        skipConnectivity = True
    else: skipConnectivity = False
ind = 1

# Set the job Spec
jobSpec = '#P ' + jobMethod + ' ' + jobType + ' SCF(Conver=9) Int(Grid=UltraFine)'
if args.geom == 'chk':
    jobSpec += ' Geom(Check) Guess(Read)'

# Sets charges + multiplicity and/or molecular geometry from original file
if args.geom in ['file','chk']:
    moleculeGeom = inputRaw[section[ind]+1:section[ind+1]]
    ind += 1

if skipConnectivity == True:
    ind += 1

if args.modRed == True:
    modRedundant = inputRaw[section[ind]+1:section[ind+1]]

# Add possibility of using allcheck? Uses the same job title and multiplicity
#elif args.geom[0] == 'allchk':
#    jobSpec += 'Geom(AllCheck)

# Parses in presets and sets variables from that?

with open(fileName+'New.com', 'w+') as output:
    print('%Chk=' + fileName, file=output)
    print('%NProcShared=12', file=output)
    print('%Mem=36000MB', file=output)
    print(jobSpec + '\n', file=output)
    if args.geom != 'allchk':
        print(jobTitle + '\n', file=output)
        for el in moleculeGeom:
            print(el, file=output)
    if args.modRed == True:
        print('\n')
        for el in modRedundant:
            print(el, file=output)
    print('\n\n', file=output)


