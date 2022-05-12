import sys
import os
import numpy as np
import torch
import scipy
from pcgcn.utils import print_color_return, tcolors

# Python 3.6.x (15)
py_ver_x = sys.version_info[0]
py_ver_y = sys.version_info[1]
if(py_ver_x == 3 and py_ver_y == 6):
    print('Python version: ' + str(py_ver_x) + '.' + str(py_ver_y) + '.x\t| ' + print_color_return(tcolors.OKGREEN, "Success") + '.')
else:
    print('Python version: ' + str(py_ver_x) + '.' + str(py_ver_y) + '.x\t| ' + print_color_return(tcolors.FAIL, "Error") + '.')
    exit(0)

# NumPy (1.19.5)
np_ver = np.__version__
if(np_ver == "1.19.5"):
    print('NumPy  version: ' + np_ver + '\t| ' + print_color_return(tcolors.OKGREEN, "Success") + '.')
else:
    print('NumPy  version: ' + np_ver + '\t| ' + print_color_return(tcolors.FAIL, "Error") + '.')
    exit(0)

# PyTorch (1.2.0)
torch_ver = torch.__version__
if(torch_ver == "1.2.0"):
    print('Torch  version: ' + torch_ver + '\t| ' + print_color_return(tcolors.OKGREEN, "Success") + '.')
else:
    print('Torch  version: ' + torch_ver + '\t| ' + print_color_return(tcolors.FAIL, "Error") + '.')
    exit(0)

# SciPy (1.5.4)
scipy_ver = scipy.__version__
if(scipy_ver == "1.5.4"):
    print('SciPy  version: ' + scipy_ver + '\t| ' + print_color_return(tcolors.OKGREEN, "Success") + '.')
else:
    print('SciPy  version: ' + scipy_ver + '\t| ' + print_color_return(tcolors.FAIL, "Error") + '.')
    exit(0)

# Check if pyconfig.h exists in /Include
pyconfig_location = (sys.executable).replace('python', 'Include/pyconfig.h')
if(not os.path.exists(pyconfig_location)):
    print("Graphlaxy can't be used | " + print_color_return(tcolors.FAIL, "ERROR") + ".")
    install = input("Do you want to be able to use it? (Y/n)  ")

    # Copies pyconfig.h to /Include dir
    if(install == 'Y'):
        copy_string = "cp " + pyconfig_location.replace('Include/', '') + " " + pyconfig_location.replace('/pyconfig.h', '')
        os.system(copy_string)
        print('Graphlaxy can be used \t| ' + print_color_return(tcolors.OKGREEN, "Success") + '.')
        print("Check the README, Section 'Notes' bullet 1, for further information about this action.")

else:
    print('Graphlaxy can be used \t| ' + print_color_return(tcolors.OKGREEN, "Success") + '.')

exit(1)