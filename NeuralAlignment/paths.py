import os
from os.path import join
import getpass
user = getpass.getuser()


ROOTDIR = os.path.abspath(join(os.path.dirname( __file__ )))

# Default paths
RESULTROOT 	= join(ROOTDIR, 'results')
if not os.path.exists(RESULTROOT):
    os.makedirs(RESULTROOT)
    print(f'Created RESULTROOT: {RESULTROOT}')
WEIGHTROOT 	= join(ROOTDIR, 'regr-weights')
DATAROOT	= join(ROOTDIR, 'data')
ACTVROOT    = join(ROOTDIR, 'model-actv') # root for storing activations
LOGROOT     = join(ROOTDIR, 'logs')
if not os.path.exists(LOGROOT):
    os.makedirs(LOGROOT)
    print(f'Created LOGROOT: {LOGROOT}')
