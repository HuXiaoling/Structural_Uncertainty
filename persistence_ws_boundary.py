"""
Persistence-based Watershed Algorithm on 2D images
Try flatten version
# _3 = np.transpose(np.unravel_index(np.flatnonzero(img_cc == img_cc[_coord_younger_cc]),img_cc.shape))
"""
from builtins import print
import numpy as np
from PIL import Image
import networkx as nx
from scipy.io import savemat
from scipy import io
import nibabel as nib
import time
import cv2
import pdb
from matplotlib import image as mpimg

t0 = time.time()

def _load_2D_img(char):
	img = np.array(Image.open(char).convert('L')) # convert('L'), i.e. rgb to greyscale - 2D
	return img

def _build_graph(img):
	'''Build the proximity graph.
	Given numpy Image matrix, build graph. 2D - 4 connectivity
	img: float, 2D
		image matrix

	Return: 
		Graph. coordinates, values, edge_tag, connected_component index
			1D: n*m nodes (1D coordinates), [(n-1)*m + m*(n-1)] edges
	'''

	G = nx.generators.lattice.grid_2d_graph(img.shape[0], img.shape[1], create_using=nx.Graph)

	f_img_cc = np.arange(0, len(G)) # initialize cc numpy array = index of v
	f_boundary = np.zeros(len(G)).astype(int) # initialize boundary numpy array = index of v
	f_img_cc_birth = np.arange(0, len(G)) # initialize cc_birth(coordinates) numpy array = 0 (will be replaced by 2D coords)

	# initialize nodes and edges attr
	for _i, _coords in enumerate(G.nodes()): # _coords - node: (1, 2)
		G._node[_coords].update({'value': img[_coords]})

	return G, f_img_cc, f_boundary, f_img_cc_birth

def _lower_star(G, v, img):
	'''Get the lower star (4 connectivity for 2D)
	G: graph
	v: one node of graph, 2D
	img: numpy image matrix, 2D
	Return:
		edge sets.
		lower_star(v) = {(u, v) in E | f(u) < f(v)}
	'''
	lower_star = []
	for _node in G.neighbors(v):
		if (img[_node] <= img[v]):
			lower_star.append((_node, v))

	return lower_star # length should less than 4

def PH_watershed(img_ori, Th, verbose = False, test = False):
	'''Persistence-based watershed algorithm for 2D images.
	img_ori: 2D matrix
	verbose: boolean
		whether talk or not talk.
	'''
	def _update_cc_pd(_coord_younger_cc, _younger_value, _coord_older_cc, _older_value, img, img_cc, img_cc_birth, img_shape):
		'''
		_coord_younger_cc is a index, converted from a node (also coordinate)
		_younger_value == img[img_cc_birth[_coord_younger_cc]], is younger cc (smallest value in younger cc)
		yonger_cc would merge into older_cc, so at that point, _coord_younger_cc is the death coords, and img_cc_birth[_coord_younger_cc] is birth coords
		'''
		_younger_cc_index = img_cc[_coord_younger_cc]
		_younger_cc_birth = img_cc_birth[_coord_younger_cc]
		_younger_cc_img_value = img[_coord_younger_cc]	# f(v), not the smallest value
		_older_cc_index = img_cc[_coord_older_cc]
		_older_cc_birth = img_cc_birth[_coord_older_cc]
		_older_cc_img_value = img[_coord_older_cc]	# not the smallest value
		if test: assert(img[_younger_cc_birth] ==  _younger_value)

		_v_cc_lists = np.flatnonzero(img_cc == _younger_cc_index) # list, all index that cc = img_cc[_coord_younger_cc]
		# _v_cc_lists = np.argwhere(img_cc == _younger_cc_index)
		_v_cc_lists_len = len(_v_cc_lists)

		# update cc
		# img_cc[ img_cc == _younger_cc_index ] = _older_cc_index # [ img_cc == _younger_cc_index ] equals to _v_cc_lists
		img_cc[ _v_cc_lists ] = _older_cc_index
		img_cc_birth[ _v_cc_lists ] = _older_cc_birth
	
		return 0

	img_shape = img_ori.shape
	count = 0

	# Build the proximity graph
	# dimension of img_cc, img_cc_birth, f_img should be 1D
	G, img_cc, boundary, img_cc_birth = _build_graph(img_ori)
	T = nx.Graph() # subgraph of G that includes all nodes and edges whose value < t -> f(v)
	img = img_ori.flatten() # not a copy
	
	# sort according to f(v), smallest to largest
	for _v_2dcoords, _value in sorted(G.nodes(data=True), key=lambda x: x[1]['value']):
		_v = np.ravel_multi_index(_v_2dcoords, img_shape) # convert matrix index to 1D flatten array index
		t = img[_v] # function value f(v), # Graph value should equal to img numpy value
		T.add_node(_v_2dcoords)

		# update cc map
		lower_star = _lower_star(G, _v_2dcoords, img_ori) # (_node, v) # return 2D edges

		for _edge_2dcoords in lower_star:
			_edge = np.ravel_multi_index(_edge_2dcoords[0], img_shape) # convert matrix index to 1D flatten array index
			if test:
				if img[_v] > img[_edge]:
					assert(T.has_node(_edge_2dcoords[0]) == True), "u must be already in T, because f(u) <= f(v)"

			if img_cc[_edge] == img_cc[_v]:
				# do not update cc map, skip edge (_edge, _v)
				if test: assert(img_cc_birth[_v] == img_cc_birth[_edge])
			else:

				# _v_value and _u_value is the smallest value in two cc
				_v_value = img[img_cc_birth[_v]] # cc value (all voxels belongs to cc index), which is cc_birth (the smallest img value in all cc)
				_u_value = img[img_cc_birth[_edge]] # _edge is coordinates of node u
				if test: assert(_v_value == np.amin(img[img_cc == img_cc[_v]])) # the smallest value in v_cc should be the same with the value of cc_birth
				if test: assert(_u_value == np.amin(img[img_cc == img_cc[_edge]]))

				# younger and older
				# coordinates: v -> _v; u -> _edge
				# cc index: v -> img_cc[_v]; u -> img_cc[_edge]
				# use >=, add = considering the [[1, 1], [1, 1]] situation

				if (t - _v_value > Th) & (t - _u_value > Th):
					boundary[_edge] = 1
					boundary[np.ravel_multi_index(_edge_2dcoords[1], img_shape)] = 1
				elif (_v_value >= _u_value) & (t - _v_value <= Th) : # v is younger, update all v cc to u

					# _v_value is younger cc (smallest value in younger cc)
					# img[_v] is f(v)
					_update_cc_pd(_coord_younger_cc=_v, _younger_value=_v_value, _coord_older_cc=_edge, _older_value=_u_value, img=img, img_cc=img_cc, img_cc_birth=img_cc_birth, img_shape=img_shape)
				elif (_u_value > _v_value) & (t - _u_value <= Th): # u is younger, , update all u cc to v
					# _u_value is youngerr cc (smallest value in younger cc)
					# img[edge[0]] is f(u)
					_update_cc_pd(_coord_younger_cc=_edge, _younger_value=_u_value, _coord_older_cc=_v, _older_value=_v_value, img=img, img_cc=img_cc, img_cc_birth=img_cc_birth, img_shape=img_shape)
				else:
					continue
	assert(len(T) == len(G))

	return img_cc, boundary, img_shape

def persistence_boundary_2d(patch, Th):

	H, W = patch.shape
	_, boundary, _ = PH_watershed(patch, Th, verbose = False, test = True)

	return np.reshape(boundary, (H, W))

def persistence_boundary(patch, Th):

	B, C, H, W = patch.shape
	patch = np.array(patch.detach().cpu())
	Th = np.array(Th.detach().cpu())
	boundary_map = np.zeros((B, C, H, W))

	for i in range(B):
		boundary_map[i,0,:,:] = persistence_boundary_2d(patch[i,0,:,:], Th[i])

	return boundary_map

if __name__ == "__main__":

    image = mpimg.imread('results/patch_revised.png')
    image = cv2.resize(image, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)

    mask = persistence_boundary_2d(image, 0.1)

    cv2.imwrite("results/mask_boundary.png", mask*255)

    print(time.time() - t0)