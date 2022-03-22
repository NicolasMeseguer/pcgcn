import sys
import numpy as np
import torch
import scipy

# Python 3.6.x (15)
py_ver_x = sys.version_info[0]
py_ver_y = sys.version_info[1]
if(py_ver_x == 3 and py_ver_y == 6):
    print('Python version: ' + str(py_ver_x) + '.' + str(py_ver_y) + '.x\t| Success.')
else:
    print('Python version: ' + str(py_ver_x) + '.' + str(py_ver_y) + '.x\t| Error...')
    exit(0)

# NumPy (1.19.5)
np_ver = np.__version__
if(np_ver == "1.19.5"):
    print('NumPy  version: ' + np_ver + '\t| Success.')
else:
    print('NumPy  version: ' + np_ver + '\t| Error...')
    exit(0)

# PyTorch (1.2.0)
torch_ver = torch.__version__
if(torch_ver == "1.2.0"):
    print('Torch  version: ' + torch_ver + '\t| Success.')
else:
    print('Torch  version: ' + torch_ver + '\t| Error...')
    exit(0)

# SciPy (1.5.4)
scipy_ver = scipy.__version__
if(scipy_ver == "1.5.4"):
    print('SciPy  version: ' + scipy_ver + '\t| Success.')
else:
    print('SciPy  version: ' + scipy_ver + '\t| Error...')
    exit(0)

exit(1)