import subprocess

import sys
from matplotlib import image as mpimg
import numpy as np
import os
import time
import cv2

t0 = time.time()

DIPHA_CONST = 8067171840
DIPHA_IMAGE_TYPE_CONST = 1
DIM = 3

dipha_output_filename = 'inputs/complex.bin'
vert_filename = 'inputs/vert.txt'
dipha_edge_filename = 'inputs/dipha.edges';
dipha_edge_txt = 'inputs/dipha_edges.txt';

def dmt_2d(patch, Th):
    nx, ny = patch.shape
    nz = 1
    im_cube = np.zeros([nx, ny, nz])
    im_cube[:, :, 0] = patch

    with open(dipha_output_filename, 'wb') as output_file:
        np.int64(DIPHA_CONST).tofile(output_file)
        np.int64(DIPHA_IMAGE_TYPE_CONST).tofile(output_file)
        np.int64(nx * ny * nz).tofile(output_file)
        np.int64(DIM).tofile(output_file)
        np.int64(nx).tofile(output_file)
        np.int64(ny).tofile(output_file)
        np.int64(nz).tofile(output_file)
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    val = int(-im_cube[i, j, k]*255)
                    np.float64(val).tofile(output_file)
        output_file.close()

    with open(vert_filename, 'w') as vert_file:
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    vert_file.write(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(int(-im_cube[i, j, k] * 255)) + '\n')
        vert_file.close()

    subprocess.call(["mpiexec", "-n", "1", "dipha-graph-recon/build/dipha", "inputs/complex.bin", "inputs/diagram.bin", "inputs/dipha.edges", str(nx), str(ny), str(nz)])

    def fread(fid, nelements, dtype):

        data_array = np.fromfile(fid, dtype, nelements)
        data_array.shape = (nelements, 1)
        return data_array

    fid = open(dipha_edge_filename, 'r' );

    dipha_identifier = fread(fid, 1, 'int64' );
    diagram_identifier = fread(fid, 1, 'int64' );
    num_pairs = fread( fid, 1, 'int64');

    file1 = open(dipha_edge_txt, "w")
    for i in range(num_pairs[0][0]):
        bverts = fread(fid, 1, 'int64');
        file1.write(str(bverts[0][0])+"\t")
        everts = fread(fid, 1, 'double');
        file1.write(str(int(everts[0][0]))+"\t")
        pers = fread(fid, 1, 'double');
        file1.write(str(int(pers[0][0]))+"\n")
        
    file1.close() 

    subprocess.call(["src/a.out", "inputs/vert.txt", "inputs/dipha_edges.txt", str(Th), "output/"])

    vert_out = np.loadtxt('output/dimo_vert.txt');

    mask = np.zeros([nx, ny])
    for i in range(len(vert_out)):
        mask[int(vert_out[i,0]), int(vert_out[i,1])] = 1;

    return mask

def dmt(patch, Th):

	B, C, H, W = patch.shape
	patch = np.array(patch.detach().cpu())
	Th = np.array(Th.detach().cpu())
	dmt_map = np.zeros((B, C, H, W))
    
	for i in range(B):
		dmt_map[i,0,:,:]= dmt_2d(patch[i,0,:,:], Th[i] * 255)

	return dmt_map

if __name__ == "__main__":

    # image = mpimg.imread('results/patch_revised.png')
    image = mpimg.imread('mito/sample_pred.png')
    # image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

    mask = dmt_2d(image, 250)

    cv2.imwrite("mito/mask_250.png", mask*255)

    print(time.time() - t0)