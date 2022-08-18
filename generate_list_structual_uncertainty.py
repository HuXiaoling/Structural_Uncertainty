from matplotlib import image as mpimg
import numpy as np
import cv2

def generate_unmap(imnum):
    mask = np.zeros((194, 188, 9))
    variance = np.zeros((194, 188))

    for i in range(9):
        mask[:,:,i] = mpimg.imread('heat_map/img{}_1_2_sample'.format(imnum) + str(i+1) + '_final.png')

    for j in range(194):
        for k in range(188):
            variance[j, k] = np.var(mask[j,k,:])

    variance = variance/np.max(variance)

    #np.save('output-for-tool/img{}_1_2_sample.npy'.format(imnum).replace('sample','uncertainty'), variance)
    return variance


def generate_cc(img):
    n_digits = 1
    step_size = 0.1
    cc_list = []

    #print unique values
    stats = np.unique(img, return_counts=True)
    print("Initial stats : \n{}".format(stats))

    round_img = img.round(decimals=n_digits) # keeping three decimal places

    stats = np.unique(round_img, return_counts=True)
    print("Round stats : \n{}".format(stats))

    for thresh_val in np.arange(step_size, 1.0+step_size, step_size):
        binary_img = (round_img > thresh_val-step_size) & (round_img < thresh_val+step_size)
        binary_img = binary_img.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img , 8, cv2.CV_32S)

        if num_labels > 1: # default 1 label for background
            for il in range(1,len(stats)): # don't consider index 0 as that is BG
                cc_list.append([thresh_val, stats[il,cv2.CC_STAT_AREA]])

    print("Number of connected components: {}".format(len(cc_list)))
    for entry in cc_list:
        print(entry)


def main():
    uncmap = generate_unmap(10)
    generate_cc(uncmap)



if __name__ == "__main__":
    main()



'''
looking at heatmap shared by xiaoling, the number of structures looks like:

img10_1_2_sample
0.4 - 26
0.7 - 11
0.9 - 11
1.0 - 7
total = 55 

img9_1_2_sample
0.4 - 18
0.7 - 8
0.9 - 4
1.0 - 8
total = 38 
'''