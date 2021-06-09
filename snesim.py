import argparse
import numpy as np
from PIL import Image
import mpslib as mps

parser = argparse.ArgumentParser(description='Obtain geological realizations by training image')
parser.add_argument('--ti', metavar='ti', type=str,
                    help='path to training image')
parser.add_argument('--hd', type=str, default="",
                    help='path to hard data file')
parser.add_argument('--n', metavar='n', type=int, default=1000,
                    help='Number of realizations')
parser.add_argument('--o', metavar='n', type=int, default=1000,
                    help='Output filename')
args = parser.parse_args()

training_img = Image.open(parser.ti)

training_image = np.array(training_img.convert('1')).astype(np.float64)[:,:,None]

O=mps.mpslib(method='mps_snesim_tree', n_threads = 8)

O.par['n_cond'] = 1000
O.par['soft_data_fnam']=parser.hd
O.par['n_real']=parser.n
O.par['simulation_grid_size'] = (92,92,1)
O.par['n_threads'] = 8
O.ti = training_image

O.run()

data_matrix = np.array([O[i].flatten() for i in range(0,len(O))]).T

np.save(parser.o, data_matrix)