"""C5G7 Benchmark - Multi-Group Diffusion Solver
Geometry and material data internal
Comparison with OpenMOC"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# ===================================================================================================
# MATERIAL DATA (7-Group)
# ===================================================================================================

# Material definitions
materials = {
    # Mix 0: Moderator
    0: {
        'name': 'Moderator',
        'SigmaT': np.array([1.59206E-01, 4.12970E-01, 5.90310E-01, 5.84350E-01, 
                           7.18000E-01, 1.25445E+00, 2.65038E+00]),
        'Chi': np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 
                        0.0, 0.0, 0.0]),
        'SigmaS': np.array([
            [4.44777E-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.13400E-01, 2.82334E-01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7.23470E-04, 1.29940E-01, 3.45256E-01, 0.0, 0.0, 0.0, 0.0],
            [3.74990E-06, 6.23400E-04, 2.24570E-01, 9.10284E-02, 7.14370E-05, 0.0, 0.0],
            [5.31840E-08, 4.80020E-05, 1.69990E-02, 4.15510E-01, 1.39138E-01, 2.21570E-03, 0.0],
            [0.0, 7.44860E-06, 2.64430E-03, 6.37320E-02, 5.11820E-01, 6.99913E-01, 1.32440E-01],
            [0.0, 1.04550E-06, 5.03440E-04, 1.21390E-02, 6.12290E-02, 5.37320E-01, 2.48070E+00]
        ]),
        'NuSigF': np.zeros(7)
    },
    
    # Mix 1: UO2 Fuel
    1: {
        'name': 'UO2',
        'SigmaT': np.array([1.77949E-01, 3.29805E-01, 4.80388E-01, 5.54367E-01, 
                           3.11801E-01, 3.95168E-01, 5.64406E-01]),
        'Chi': np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 
                        0.0, 0.0, 0.0]),
        'SigmaS': np.array([
            [1.27537E-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.23780E-02, 3.24456E-01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [9.43740E-06, 1.63140E-03, 4.50940E-01, 0.0, 0.0, 0.0, 0.0],
            [5.51630E-09, 3.14270E-09, 2.67920E-03, 4.52565E-01, 1.25250E-04, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.56640E-03, 2.71401E-01, 1.29680E-03, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.02550E-02, 2.65802E-01, 8.54580E-03],
            [0.0, 0.0, 0.0, 0.0, 1.00210E-08, 1.68090E-02, 2.73080E-01]
        ]),
        'NuSigF': np.array([2.005998E-02, 2.027303E-03, 1.570599E-02, 4.518301E-02, 
                           4.334208E-02, 2.020901E-01, 5.257105E-01])
    },
    
    # Mix 2: MOX 4.3%
    2: {
        'name': 'MOX4.3',
        'SigmaT': np.array([1.78731E-01, 3.30849E-01, 4.83772E-01, 5.66922E-01, 
                           4.26227E-01, 6.78997E-01, 6.82852E-01]),
        'Chi': np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 
                        0.0, 0.0, 0.0]),
        'SigmaS': np.array([
            [1.28876E-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.14130E-02, 3.25452E-01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.22900E-06, 1.63950E-03, 4.53188E-01, 0.0, 0.0, 0.0, 0.0],
            [5.04050E-09, 1.59820E-09, 2.61420E-03, 4.57173E-01, 1.60460E-04, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.53940E-03, 2.76814E-01, 2.00510E-03, 0.0],
            [0.0, 0.0, 0.0, 0.0, 9.31270E-03, 2.52962E-01, 8.49480E-03],
            [0.0, 0.0, 0.0, 0.0, 9.16560E-09, 1.48500E-02, 2.65007E-01]
        ]),
        'NuSigF': np.array([0.0217530045, 0.0025351033, 0.0162679915, 0.0654740997, 
                           0.0307240878, 0.6666509616, 0.7139904304])
    },
    
    # Mix 3: MOX 7.0%
    3: {
        'name': 'MOX7.0',
        'SigmaT': np.array([0.181323, 0.334368, 0.493785, 0.591216, 
                           0.474198, 0.833601, 0.853603]),
        'Chi': np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 
                        0.0, 0.0, 0.0]),
        'SigmaS': np.array([
            [0.130457, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.041792, 0.328428, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.5105E-06, 0.0016436, 0.458371, 0.0, 0.0, 0.0, 0.0],
            [5.1329E-09, 2.2017E-09, 0.0025331, 0.463709, 0.00017619, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0054766, 0.282313, 0.002276, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0087289, 0.249751, 0.0088645],
            [0.0, 0.0, 0.0, 0.0, 9.0016E-09, 0.013114, 0.259529]
        ]),
        'NuSigF': np.array([0.023813952, 0.0038586888, 0.0241340014, 0.09436622, 
                           0.0457698761, 0.9281814045, 1.0432001182])
    },
    
    # Mix 4: MOX 8.7%
    4: {
        'name': 'MOX8.7',
        'SigmaT': np.array([0.183045, 0.336705, 0.500507, 0.606174, 
                           0.502754, 0.921028, 0.955231]),
        'Chi': np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 
                        0.0, 0.0, 0.0]),
        'SigmaS': np.array([
            [0.131504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.042046, 0.330403, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.6972E-06, 0.0016463, 0.461792, 0.0, 0.0, 0.0, 0.0],
            [5.1938E-09, 2.6006E-09, 0.0024749, 0.468021, 0.00018597, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.005433, 0.285771, 0.0023916, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0083973, 0.247614, 0.0089681],
            [0.0, 0.0, 0.0, 0.0, 8.928E-09, 0.012322, 0.256093]
        ]),
        'NuSigF': np.array([0.0251860041, 0.0047395095, 0.029478054, 0.1122499985, 
                           0.0553030128, 1.0749988378, 1.23929836992])
    },
    
    # Mix 5: Fission Chamber
    5: {
        'name': 'FissionChamber',
        'SigmaT': np.array([0.126032, 0.29316, 0.28425, 0.28102, 
                           0.33446, 0.56564, 1.17214]),
        'Chi': np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 
                        0.0, 0.0, 0.0]),
        'SigmaS': np.array([
            [0.0661659, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.059070, 0.240377, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.00028334, 0.052435, 0.183425, 0.0, 0.0, 0.0, 0.0],
            [1.4622E-06, 0.0002499, 0.092288, 0.0790769, 0.00003734, 0.0, 0.0],
            [2.0642E-08, 0.000019239, 0.0069365, 0.16999, 0.099757, 0.00091742, 0.0],
            [0.0, 2.9875E-06, 0.001079, 0.02586, 0.20679, 0.316774, 0.049793],
            [0.0, 4.214E-07, 0.00020543, 0.0049256, 0.024478, 0.23876, 1.0991]
        ]),
        'NuSigF': np.array([1.3234E-08, 1.4345E-08, 1.1285993E-06, 1.27629932E-05, 
                           3.538502E-07, 1.7400989E-06, 5.0633019E-06])
    },
    
    # Mix 6: Guide Tube
    6: {
        'name': 'GuideTube',
        'SigmaT': np.array([1.26032E-01, 2.93160E-01, 2.84240E-01, 2.80960E-01, 
                           3.34440E-01, 5.65640E-01, 1.17215E+00]),
        'Chi': np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 
                        0.0, 0.0, 0.0]),
        'SigmaS': np.array([
            [6.61659E-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.90700E-02, 2.40377E-01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.83340E-04, 5.24350E-02, 1.83297E-01, 0.0, 0.0, 0.0, 0.0],
            [1.46220E-06, 2.49900E-04, 9.23970E-02, 7.88511E-02, 3.73330E-05, 0.0, 0.0],
            [2.06420E-08, 1.92390E-05, 6.94460E-03, 1.70140E-01, 9.97372E-02, 9.17260E-04, 0.0],
            [0.0, 2.98750E-06, 1.08030E-03, 2.58810E-02, 2.06790E-01, 3.16765E-01, 4.97920E-02],
            [0.0, 4.21400E-07, 2.05670E-04, 4.92970E-03, 2.44780E-02, 2.38770E-01, 1.09912E+00]
        ]),
        'NuSigF': np.zeros(7)
    }
}

# ===================================================================================================
# GEOMETRY DATA
# ===================================================================================================

# Pin definitions
pins = {
    0: {'type': 'box', 'mats': [0], 'pitch': 1.26},  # moderator
    1: {'type': 'pin', 'mats': [1, 0], 'pitch': 1.26, 'radius': 0.54},  # UO2
    2: {'type': 'pin', 'mats': [2, 0], 'pitch': 1.26, 'radius': 0.54},  # MOX4.3
    3: {'type': 'pin', 'mats': [3, 0], 'pitch': 1.26, 'radius': 0.54},  # MOX7.0
    4: {'type': 'pin', 'mats': [4, 0], 'pitch': 1.26, 'radius': 0.54},  # MOX8.7
    5: {'type': 'pin', 'mats': [5, 0], 'pitch': 1.26, 'radius': 0.54},  # fission chamber
    6: {'type': 'pin', 'mats': [6, 0], 'pitch': 1.26, 'radius': 0.54}   # Guide Tube
}

# C5G7 core grid (51x51)
# Pattern: UO2 assemblies (bottom-left), MOX assemblies (top-right), Reflector (right, top)
grid = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,6,1,1,6,1,1,5,1,1,6,1,1,6,1,1, 2,3,6,4,4,6,4,4,5,4,4,6,4,4,6,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,6,4,4,6,4,4,5,4,4,6,4,4,6,3,2, 1,1,6,1,1,6,1,1,5,1,1,6,1,1,6,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
] + [[0]*51]*17)  # Reflector region (17 more lines)

# ===================================================================================================
# DIFFUSION SOLVENT
# ===================================================================================================

class C5G7DiffusionSolver:
    def __init__(self, mesh_refinement=1):
        """mesh_refinement: How many parts we will divide each cell into (1, 2, 3, ...)"""
        self.mesh_ref = mesh_refinement
        self.Nx = 51 * mesh_refinement
        self.Ny = 51 * mesh_refinement
        self.ngroups = 7
        self.Nc = self.Nx * self.Ny
        self.h = 1.26 / mesh_refinement  # cm
        
        print(f"Mesh: {self.Nx}×{self.Ny} (refinement={mesh_refinement})")
        print(f"Cell size: {self.h:.4f} cm")
        print(f"Total unknowns: {self.ngroups * self.Nc}")
        
    def homogenize_cell(self, pin_id):
        """Homogenize pin cell"""
        pin = pins[pin_id]
        
        if pin['type'] == 'box':
            # homogeneous region
            mat = materials[pin['mats'][0]]
            return mat['SigmaT'], mat['Chi'], mat['NuSigF'], mat['SigmaS']
        else:
            # Pin geometry (fuel + moderator)
            mat_fuel = materials[pin['mats'][0]]
            mat_mod = materials[pin['mats'][1]]
            
            # Volume fractions
            vol_fuel = (np.pi * pin['radius']**2) / (pin['pitch']**2)
            vol_mod = 1.0 - vol_fuel
            
            # Volume-weighted average
            st = vol_fuel * mat_fuel['SigmaT'] + vol_mod * mat_mod['SigmaT']
            chi = mat_fuel['Chi']  # Fission spectrum from fuel
            nsf = vol_fuel * mat_fuel['NuSigF']  # No fission in moderator
            ss = vol_fuel * mat_fuel['SigmaS'] + vol_mod * mat_mod['SigmaS']
            
            return st, chi, nsf, ss
    
    def setup_problem(self):
        """Calculate homogenized sections"""
        print("\nHomogenization...")
        
        self.D = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.Sigma_rem = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.NuSigF = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.Chi = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.Sigma_s_mat = np.zeros((self.ngroups, self.ngroups, self.Nx, self.Ny))
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                # Find pin ID from original grid
                i_orig = i // self.mesh_ref
                j_orig = j // self.mesh_ref
                pin_id = grid[i_orig, j_orig]
                
                st, chi, nsf, ss = self.homogenize_cell(pin_id)
                
                for g in range(self.ngroups):
                    self.D[g, i, j] = 1.0 / (3.0 * st[g])
                    self.Sigma_rem[g, i, j] = st[g] - ss[g, g]
                    self.NuSigF[g, i, j] = nsf[g]
                    self.Chi[g, i, j] = chi[g]
                    
                    for gp in range(self.ngroups):
                        self.Sigma_s_mat[g, gp, i, j] = ss[g, gp]
        
        print("✓ Homogenization Done ")
    
    def build_matrices(self):
        """Create system matrices"""
        print("Matrices are being created...")
        
        rows_M, cols_M, vals_M = [], [], []
        rows_F, cols_F, vals_F = [], [], []
        
        for g in range(self.ngroups):
            for i in range(self.Nx):
                for j in range(self.Ny):
                    idx = g * self.Nc + i * self.Ny + j
                    
                    # Diagonal term
                    m_diag = self.Sigma_rem[g, i, j]
                    
                    # Diffusion terms
                    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = i+di, j+dj
                        
                        if 0 <= ni < self.Nx and 0 <= nj < self.Ny:
                            # Hinterland
                            d_avg = 2 * self.D[g, i, j] * self.D[g, ni, nj] / \
                                   (self.D[g, i, j] + self.D[g, ni, nj] + 1e-20)
                            coeff = d_avg / self.h**2
                            m_diag += coeff
                            rows_M.append(idx)
                            cols_M.append(g * self.Nc + ni * self.Ny + nj)
                            vals_M.append(-coeff)
                        else:
                            # Boundary conditions
                            # X-, Y-: Reflective (ZeroCurrent)
                            if (i == 0 and di == -1) or (j == 0 and dj == -1):
                                pass  # Derivative = 0
                            # X+, Y+: Vacuum (Marshak)
                            elif (i == self.Nx-1 and di == 1) or (j == self.Ny-1 and dj == 1):
                                alpha = 0.5
                                d_boundary = self.D[g, i, j] / (self.h * (alpha + self.D[g, i, j] / self.h))
                                m_diag += d_boundary
                    
                    rows_M.append(idx)
                    cols_M.append(idx)
                    vals_M.append(m_diag)
                    
                    # Scattering and fission terms
                    for gp in range(self.ngroups):
                        if gp != g:
                            s_val = self.Sigma_s_mat[g, gp, i, j]
                            if s_val > 1e-30:
                                rows_M.append(idx)
                                cols_M.append(gp * self.Nc + i * self.Ny + j)
                                vals_M.append(-s_val)
                        
                        f_val = self.Chi[g, i, j] * self.NuSigF[gp, i, j]
                        if f_val > 1e-30:
                            rows_F.append(idx)
                            cols_F.append(gp * self.Nc + i * self.Ny + j)
                            vals_F.append(f_val)
        
        self.M = sp.csr_matrix((vals_M, (rows_M, cols_M)), 
                               shape=(self.ngroups*self.Nc, self.ngroups*self.Nc))
        self.F = sp.csr_matrix((vals_F, (rows_F, cols_F)), 
                               shape=(self.ngroups*self.Nc, self.ngroups*self.Nc))
        
        print(f"✓ M matrix: {self.M.shape}, nnz={self.M.nnz}")
        print(f"✓ F matrix: {self.F.shape}, nnz={self.F.nnz}")
    
    def solve(self, max_iter=500, tol=1e-8):
        """power iteration"""
        print(f"\nPower Iteration (tol={tol:.0e}, max_iter={max_iter})")
        print("="*70)
        
        t0 = time.time()
        
        # LU factorization
        solve = spla.factorized(self.M.tocsc())
        
        # Starting estimate
        phi = np.ones(self.ngroups * self.Nc)
        phi /= np.linalg.norm(phi)
        k_eff = 1.0
        
        for it in range(1, max_iter + 1):
            fission_source = self.F @ phi
            phi_new = solve(fission_source / k_eff)
            phi_new = np.maximum(phi_new, 0)
            
            fission_source_new = self.F @ phi_new
            
            sum_old = np.sum(fission_source)
            sum_new = np.sum(fission_source_new)
            
            if sum_old > 1e-30:
                k_new = k_eff * (sum_new / sum_old)
            else:
                k_new = k_eff
            
            err = abs(k_new - k_eff) / k_new
            phi = phi_new / (np.linalg.norm(phi_new) + 1e-30)
            k_eff = k_new
            
            if it % 10 == 0 or err < tol:
                print(f"Iter {it:4d}: k_eff = {k_eff:.7f} | Error = {err:.3e}")
            
            if err < tol:
                print("="*70)
                print(f"✓ CONVERGENCE (iter={it}, Time={time.time()-t0:.2f}s)")
                break
        else:
            print("="*70)
            print(f"⚠ Max. Iteration ({max_iter})")
        
        self.phi = phi
        self.k_eff = k_eff
        self.solve_time = time.time() - t0
        
        return k_eff, phi

# ===================================================================================================
# MAIN PROGRAM
# ===================================================================================================

if __name__ == "__main__":
    print("="*70)
    print("C5G7 BENCHMARK - MULTI-GROUP DIFFUSION SOLVER")
    print("="*70)
    
    # Mesh refinement selection
    # 1 = 51x51 (fast, less accurate)
    # 2 = 102x102 (medium)
    # 3 = 153x153 (slower, more accurate)
    MESH_REF = 1
    
    # Create solver
    solver = C5G7DiffusionSolver(mesh_refinement=MESH_REF)
    
    # problem setup
    solver.setup_problem()
    solver.build_matrices()
    
    # solve
    k_eff, phi = solver.solve()
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"k_eff (Diffusion)    = {k_eff:.6f}")
    print(f"Solution time        = {solver.solve_time:.2f} s")
    print(f"Mesh                 = {solver.Nx}×{solver.Ny}")
    print()
    print("REFERENCE VALUES:")
    print("-" * 70)
    print("OECD/NEA Benchmark   = 1.18655 ± 0.00010")
    print("Monte Carlo (Serpent)= 1.18648 ± 0.00008")
    print("MCNP6                = 1.18656 ± 0.00012")
    print("OpenMOC (MOC, fine)  = ~1.1863")
    print("Diffusion (literature)= 1.1850 - 1.1870 (mesh-dependent)")
    print("-" * 70)

    delta_pcm = (k_eff - 1.18655) * 1e5
    print(f"\nDifference from benchmark = {delta_pcm:+.0f} pcm")

    if abs(delta_pcm) < 200:
        print("✓ Result is within acceptable range!")
    else:
        print("⚠ Difference is large - try mesh refinement")
    
    
    # visualization
    print("\n" + "="*70)
    print("visualization")
    print("="*70)
    
    phi_3d = phi.reshape(solver.ngroups, solver.Nx, solver.Ny)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Fluxes by group
    group_names = ['Group 0\n(Fast)', 'Group 2\n(Epithermal)', 'Group 6\n(Thermal)',
                   'Total Flux', 'Fission Power', 'Pin Detail']
    
    # Group 0 (Fast)
    im0 = axes[0,0].imshow(phi_3d[0].T, cmap='jet', origin='lower')
    axes[0,0].set_title(f'{group_names[0]}\nk_eff={k_eff:.5f}')
    plt.colorbar(im0, ax=axes[0,0])
    
    # Group 2 (Epithermal)
    im1 = axes[0,1].imshow(phi_3d[2].T, cmap='jet', origin='lower')
    axes[0,1].set_title(group_names[1])
    plt.colorbar(im1, ax=axes[0,1])
    
    # Group 6 (Thermal)
    im2 = axes[0,2].imshow(phi_3d[6].T, cmap='jet', origin='lower')
    axes[0,2].set_title(group_names[2])
    plt.colorbar(im2, ax=axes[0,2])
    
    # total flux
    total_flux = np.sum(phi_3d, axis=0)
    im3 = axes[1,0].imshow(total_flux.T, cmap='jet', origin='lower')
    axes[1,0].set_title(group_names[3])
    plt.colorbar(im3, ax=axes[1,0])
    
    # Fission power
    fission_power = np.zeros((solver.Nx, solver.Ny))
    for g in range(solver.ngroups):
        fission_power += solver.NuSigF[g] * phi_3d[g]
    
    im4 = axes[1,1].imshow(fission_power.T, cmap='hot', origin='lower')
    axes[1,1].set_title(group_names[4])
    plt.colorbar(im4, ax=axes[1,1])
    
    # Pin power peaking
    max_power = np.max(fission_power)
    avg_power = np.mean(fission_power[fission_power > 0])
    peaking = max_power / avg_power
    
    # geometry overlay
    axes[1,2].imshow(grid.T, cmap='tab10', origin='lower', alpha=0.5)
    axes[1,2].contour(fission_power.T, levels=10, colors='black', linewidths=0.5)
    axes[1,2].set_title(f'Geometry + Power Contour\nPeaking={peaking:.3f}')
    
    plt.tight_layout()
    plt.savefig('c5g7_full_results.png', dpi=150)
    print("✓ Png Saved : c5g7_full_results.png")
    
    # Statistics
    print("\nADDITIONAL ANALYSES:")
    print("-" * 70)
    print(f"Pin power peaking factor = {peaking:.4f}")
    print(f"Maximum power            = {max_power:.4e}")
    print(f"Average power            = {avg_power:.4e}")
    print()
    print("Total flux by group:")
    for g in range(solver.ngroups):
        integral = np.sum(phi_3d[g])
        print(f"  Group {g}: {integral:.4e}")

    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)