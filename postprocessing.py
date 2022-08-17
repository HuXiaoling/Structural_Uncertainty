from bdb import set_trace
import numpy as np
from matplotlib import image as mpimg
import cv2
from skimage.measure import label
import argparse, os, glob

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def generate_final(skeleton, mask_binary):

    union = np.logical_or(skeleton, mask_binary)
    largestCC = getLargestCC(union)

    return largestCC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type= str, default = "CREMI")
    parser.add_argument('--sample_id', type= int, default = 1)

    args = parser.parse_args()

    dir = 'experiments/'

    path_ori = os.path.join(dir, args.dataset, 'patch')
    path_target = os.path.join(dir, args.dataset, 'sample' + str(args.sample_id))

    mylist = [f for f in glob.glob(os.path.join(path_ori, "*_pred_binary.png"))]

    for i in range(len(mylist)):
        print(i)
        mask_binary = mpimg.imread(mylist[i])
        skeleton = mpimg.imread(mylist[i].replace('patch', 'sample' + str(args.sample_id))[:-15]+'skeleton.png')
        # import pdb; pdb.set_trace()
    
        cv2.imwrite(mylist[i].replace('patch', 'sample' + str(args.sample_id))[:-15]+ 'final.png', generate_final(skeleton, mask_binary)*255)