import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gaussTherm as gT


# Front end runner to pull thermochemistry data for molecules
molNames, molFile, molecules = gT.thermalV1(sys.argv[1])

# Create dataframe
if len(sys.argv) > 2:
    thermalProp = gT.DataFramer(molNames, molFile, molecules, fileName=sys.argv[2])
else:
    thermalProp = gT.DataFramer(molNames, molFile, molecules)



