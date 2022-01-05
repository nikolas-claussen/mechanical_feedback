#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Sep  14 11:13:23 2020

@author: nikolas, nclaussen@ucsb.edu


This module implements a windowed Radon transform as used in Ref.
https://doi.org/10.7554/eLife.27454. Rhe Radon transform is the integral
transform which takes a function f defined on the plane to a function Rf
defined on the (two-dimensional) space of lines in the plane,
whose value at a particular line is equal to the line integral of the
function over that line https://en.wikipedia.org/wiki/Radon_transform.

A windowed Radon transform subdivides an input image domain Omega into many
small rectangles and comutes the Radon transform of each. Thus, it can capture
the local anisotropy of the image. The output of the "true" Radon transform
of a function f(x,y) is anothe function Rf(alpha,s) where alpha is the line
angle and s its distance to the origin along the x-axis.

In addition, there are a number of related convenience functions.

"""

import numpy as np

from collections import Iterable

from skimage.transform import radon, rotate
from skimage.morphology import h_maxima, disk

from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.sparse import csc_matrix
from scipy.interpolate import NearestNDInterpolator

from sklearn import cluster

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


### Process detected lines


def flatten(lst, max_depth=1000, iter_count=0):
    """
    Flatten a list of lists into a list.

    Also works with inhomogeneous lists, e.g., [[0,1],2]. The argument
    depth determines how "deep" to flatten the list, e.g. with max_depth=1:
    [[(1,0), (1,0)]] -> [(1,0), (1,0)].

    Parameters
    ----------
    lst : list
        list-of-lists.
    max_depth : int, optional
        To what depth to flatten the list.
    iter_count : int, optional
        Helper argumenr for recursion depth determination.
    Returns
    -------
    iterator
        flattened list.

    """
    for el in lst:
        if (isinstance(el, Iterable) and not isinstance(el, (str, bytes))
                and iter_count < max_depth):
            yield from flatten(el, max_depth=max_depth,
                               iter_count=iter_count+1)
        else:
            yield el


def remap_angles(angles, convert_to_deg=True):
    """Remap nematic angles with range 0-180 with 0=180, to range -90 & 90, with 90=-90. Output in degrees"""
    if convert_to_deg:
        angles = angles*(180/np.pi)
    angles[angles >= 90] = angles[angles >= 90]-180
    return angles


def average_angles(thetas, weights=np.array([1])):
    """Average angles via q-tensor. Note: ill-defined if angles are @ 90 degrees to one another"""
    # compute tensors
    q_matrices = np.array([[np.sin(thetas)**2, -np.sin(thetas)*np.cos(thetas)],
                           [-np.sin(thetas)*np.cos(thetas), np.cos(thetas)**2]])
    q_matrix = (q_matrices*weights).sum(axis=-1)/weights.sum()
    eigenvector = np.linalg.eigh(q_matrix)[1][:,-1]
    eigenvector = eigenvector * np.sign(eigenvector[0]+1e-5)
    angle = np.pi-np.arctan2(eigenvector[0], eigenvector[1])
    
    return angle


def merge_lines(lines, distance_threshold=5):
    """
    Merge lines detected by windowed radon transform.
    
    Merging removes duplicate lines (in particular from overlaping windows,
    and reduces clutter. Returns a single array of shape (...,4) of lines.
    """
    if type(lines) is list:
        lines = np.stack([np.hstack(x) for x in wr.flatten(lines, max_depth=2)])
    clustering = cluster.AgglomerativeClustering(n_clusters=None, linkage='ward',
                                                 distance_threshold=distance_threshold).fit(lines[:,2:])
    labels = clustering.labels_
    # now merge points etc accordingly, weighting by intensity.
    lines_merged = []
    for n in range(labels.max()):
        lines_to_merge = lines[labels==n]
        if lines_to_merge.shape[0] == 1:
            lines_merged.append(lines_to_merge[0])
            continue
        weights = lines_to_merge[:,0]
        weights = weights/weights.sum()
        center = (lines_to_merge[:,2:].T *weights).sum(axis=-1)
        angle = average_angles(lines_to_merge[:,1], weights=weights)
        intensity = (lines_to_merge[:,0]*weights).sum()
        lines_merged.append(np.hstack([intensity, angle, center]))
    return np.stack(lines_merged)


### Plotting

def plot_line_segments_over_image(image, lines, figsize, segment_length=7,
                                  segment_kwargs={}, imshow_kwargs={}):
    if len(segment_kwargs) == 0:
        segment_kwargs = {'s': 20, 'color': 'red', 'lw': 2}
    fig = plt.figure(figsize=figsize)
    for l in lines:
        m = l[0]
        if m > 100:
            a, b = np.sin(l[1]), np.cos(l[1])
            plt.plot(length*np.array([-a, 0, a])+l[2],
                     length*np.array([-b, 0, b])+l[3],
                     color=segment_kwargs['color'], lw=segment_kwargs['lw'])
            plt.scatter(l[2], l[3], color=segment_kwargs['color'], s=segment_kwargs['s'])
    plt.imshow(image, **imshow_kwargs)


def color_by_angle(image, lines, cmap, angle_min, angle_max,
                   sigma=20, angle_transform=None):
    """
    Color image by orientation of detected edges.
    
    Parameters
    ----------
    image : np.array of shape (n_y, n_x)
        Intensity image. Must be rescaled to values
        between 0 and 1.
    lines : np.array of shape (...,4)
        Detected lines
    cmap : colormap to use
    angle_min, angle_max : int
        min and max angle values in colormap
    sigma : float
        Smooting parameter
    angle_transform : None or callable
        Transform to apply to angle image, e.g.
        np.abs() or remap_angles().
        
    Returns
    -------
    rgba_color : np.array of shape (n_y, n_x, 4)
        RGBA array.
    """
    
    # interpolate the angle
    x, y = (np.arange(range(x)) for x in image.shape[::-1])
    X, Y = np.meshgrid(x, y)
    
    pts = lines[:,2:]
    vals = lines[:,1]
    interpolator = NearestNDInterpolator(pts, vals)
    angle_image = gaussian_filter(interpolator(X, Y), sigma=sigma)

    if angle_transform is not None:
        angle_image = angle_transform(angle_image)
    
    norm = Normalize(vmin=angle_min, vmax=angle_max)
    rgba_color = cmap(norm(angle_image))
    rgba_color[:,:,-1] = image
    
    return rgba_color


### Tensor processing


def smooth_tensor(m, sigma):
    """Smooth tensor of shape (n_y, n_x, d1, d1)."""
    d = m.shape[2:]
    m_new = np.stack([[gaussian_filter(m[:, :, i, j], sigma)
                       for i in range(d[0])]
                     for j in range(d[1])]).transpose(2, 3, 0, 1)
    return m_new


def remove_trace(m):
    """Get traceless part of tensor field (shape  (n_y, n_x, d, d))."""
    m_tr = np.tensordot(np.trace(m, axis1=2, axis2=3), np.eye(m.shape[-1]),
                        axes=0)/2
    return m - m_tr


### Subroutines for windowed radon transform


def get_segment_com(im, angle, offset, pad=2):
    """Get parallel distance of line segment from image center."""
    e_len = int((im.shape[0]-1)/2)
    c = (-offset+e_len)
    rotated = rotate(im*disk(e_len), 90-180*angle/np.pi)
    rotated = rotated[c-pad: c+pad+1].sum(axis=0)
    rotated -= rotated.min()
    rotated /= (rotated.sum()+1e-2)
    return (rotated * np.arange(-e_len, e_len+1)).sum()


def shrink_to_centroid(binary_img):
    """Shrink connected components of binary image to their centroids"""
    labeled = label(binary_img)
    shrunk = np.zeros(binary_img.shape)
    for ind in [np.round(x['centroid']).astype(int) for x in regionprops(labeled)]:
        shrunk[ind[0], ind[1]] = 1
    return shrunk


def get_radon_tf_matrix(edge_length, theta=None, line_density=True,
                        correction=True):
    """
    Get matrix representation of Radon transform for given image size.

    The Radon transform is a linear operation and can therefore be represented
    by a matrix (which, additionally, is very sparse). For small image sizes
    (roughly < 100*100 pixels), using this matrix representation is much
    faster (x40) than skimage.transform.radon (by avoiding image rotation/
    interpolation). However, it can take some time to compute the matrix
    representation (2 mins for a 50*50 image).

    Uses the circle=True convention from skimage.transform.radon (i.e.
    assumes input image is zero outside the inscribed circle).

    Parameters
    ----------
    edge_length : int
        Image size. Must be odd.
    theta : np.array, optional
        Array of angles at which to evaluate the Radon transform.
        The default is None (set theta to
        np.linspace(0, 180, edge_length, endpoint=False)).
    line_density : bool, optional
        Whether to re-weight the radon transform so as to
        return not line integrals but line densities. If this
        is enabled, 2 pixels of the "offset" coordinate are cut off
        to avoid 0 division error.
    correction : bool, optional
        Whether to correct artifacts of radon transform
        (which can become prominent at small pixel values).
        Only works in combination with line_density.

    Returns
    -------
    matrix_rep : scipy.sparse.csc_matrix
        Sparse matrix representation of Radon transform.

    """
    assert edge_length % 2, "Edge length must be odd"
    e_len = int((edge_length-1)/2)
    if theta is None:
        theta = np.linspace(0, 180, edge_length, endpoint=False)
    delta = np.arange(-e_len, e_len+1)
    line_weights = 1/(2*np.sqrt(e_len**2-(delta)**2))
    cutoff = 2
    line_weights[:cutoff] = line_weights[-cutoff:] = 0
        
    if not line_density:
        matrix_rep = np.zeros((edge_length*len(theta),
                               edge_length, edge_length))
    if line_density:
        matrix_rep = np.zeros(((edge_length-2*cutoff)*len(theta),
                               edge_length, edge_length))    
    ones = disk(e_len)
    trans_ones = radon(ones, theta=theta, preserve_range=True,
                       circle=True,)
    trans_ones = (trans_ones.T * line_weights).T[cutoff:-cutoff]
    
    for x in range(edge_length):
        for y in range(edge_length):
            # Ensure input image is zero outside the inscribed circle
            if (x-e_len)**2+(y-e_len)**2 > e_len**2:
                continue
            test = np.zeros((edge_length, edge_length))
            test[x, y] = 1
            trans = radon(test, theta=theta, preserve_range=True,
                          circle=True)
            if line_density:
                trans = (trans.T * line_weights).T[cutoff:-cutoff]
            if correction:
                trans = trans / trans_ones
            matrix_rep[:, x, y] = trans.flatten()
    if not line_density:
        matrix_rep = matrix_rep.reshape(edge_length*len(theta),
                                        edge_length*edge_length)
    if line_density:
        matrix_rep = matrix_rep.reshape((edge_length-2*cutoff)*len(theta),
                                        edge_length*edge_length)
    matrix_rep = csc_matrix(matrix_rep)
    return matrix_rep


def windowed_radon(img, radon_matrix, theta=None, method='h_maxima',
                   threshold_rel=2, threshold_rel_global=False,
                   threshold_mean=1.5, return_lines=False, debug=False):
    """
    Get coarse-grained anisotropy tensor for img via windowed Radon transform.
    
    Basic version with no support for curved surfaces, and only one option
    for maxima detection, to make code easier to read.

    See https://doi.org/10.7554/eLife.27454. This function first
    calculates a windowed Radon transform, then detects maxima of the Radon
    transform and finally computes a coarse-grained anisotropy tensor for each
    Radon patch according to tensorify_radon. Alternatively, the function can
    also return the centroid, orientation and intensity of each detected edge
    segment (for verification purposes). This information is returned in a
    2d-array like nested list format, each entry corresponding to the list
    of detected line segments from one particular Radon window.

    By default, the Radon transform is weighted so a to calculate the line
    density along each ray (this is done approximately).

    The crucial step in anisotropy detection is finding the maxima of the
    Radon transform. This is implemented using the h-maxima transform. It
    is necessary to tune the "h"-parameter of the h-maxima transform and check
    that your choice correctly detects all edge segment of interest.

    As an example of why maxima detection is so important, consider the Radon
    transform of an image containing a single line, and average over the
    projection distance delta, (so as to obtain a function of angle only). The
    result is _independent_ of the original line orientation!

    For performance reasons, this function uses a sparse matrix representation
    of the Radon transform, computed in advance using get_radon_tf_matrix.

    Parameters
    ----------
    img : np.array
        Input image.
    radon_matrix : sparse.csc_matrix
        Sparse matrix representing the Radon transform (remember, it's linear).
        Computed by get_radon_tf_matrix. Determines the size of the radon
        window.
    theta : np.array, optional
        Angles at which radon transform is computed. Defaults to
        np.linspace(0, np.pi, edge_length, endpoint=False) where edge_length
        is the window size (defined by the shape of radon_matrix).
    method : str, optional
        Either 'h_maxima' or 'global_maximum' (faster).
    threshold_rel : float, optional
        Threshold for maxima detection in radon transform. For "h_maxima",
        this is the minimal height of maxima in units of the Radon transform's
        standard deviation. For "peak_local_max", it is the minimal height of
        maxima in multiples of the global maximum.
    threshold_mean : float, optional
        2nd auxilliary threshold. All Radon transform maxima must be at least
        threshold_mean * mean(Radon transform). This removes e.g. the global
        maximum always returned by the h maxima transform if it is not
        sufficiently pronounced.
    threshold_rel_global :  float, optional
        Global threshold for h-maxima computation, in units of image
        std deviation. If non-zero, overrules the local h-maxima thresholds.
        Use when large regions of image contain no anisotropy.
    return_lines : bool, optional
        Whether to also return a list of centroids, orientations & intensities
        of all detected edges instead of just the coarse-grained anisotropy
        tensor.

    Returns
    -------
    lines : list of lists of np.arrays of shape (..., 4)
        Returned only if return_lines is True. Lines detected in each window,
        formatted as follows (x_coord, y_coord, 
    m :  np.array of shape (...,..., 2, 2)
        Coarse grained anisotropy tensor. The first two axes are "spatial"
        indices and have the following extent:
            ceil(2*(img_shape - edge_length+1)/(edge_length+1)),
        where edge_length is the radon window size.

    """
    edge_length = np.sqrt(radon_matrix.shape[1]).astype(int)
    e_len = int((edge_length-1)/2)
    if theta is None:
        theta = np.linspace(0, np.pi, int(radon_matrix.shape[0]/(edge_length-4)),
                            endpoint=False)
    e_len_cut = int((radon_matrix.shape[0] / theta.size-1)/2)
    delta = np.arange(-e_len_cut, e_len_cut+1) 
    # needed because line_density radon transform returns cut-off
    
    # director matrix
    q_matrix = np.array([[np.sin(theta)**2, -np.sin(theta)*np.cos(theta)],
                         [-np.sin(theta)*np.cos(theta), np.cos(theta)**2]])

    global_h = threshold_rel_global * img.std()

    # iterate over sub-arrays
    m = []
    lines = []
    for r in np.arange(e_len, img.shape[0]-e_len, e_len+1):
        m_row = []
        lines_row = []
        for c in np.arange(e_len, img.shape[1]-e_len, e_len+1):
            patch = img[r-e_len:r+e_len+1, c-e_len:c+e_len+1]
            radon_window = radon_matrix.dot(patch.flatten())
            radon_window = radon_window.reshape(2*e_len_cut+1, theta.size)
            # compute mean based threshold for maxima
            thr_mean = (radon_window.min()+
                        threshold_mean*(np.median(radon_window)-radon_window.min()))
            if method == 'h_maxima':
                h = (global_h if global_h
                     else radon_window.std()*threshold_rel+1e-2)
                max_mask = h_maxima(radon_window, h)
                max_mask = binary_dilation(max_mask, iterations=1)
                max_mask *= (radon_window > thr_mean)
            if method == 'global_maximum':
                max_mask = np.zeros(radon_window.shape)
                ind = np.unravel_index(np.argmax(radon_window, axis=None), radon_window.shape)
                max_mask[ind] = radon_window[ind] > thr_mean
            if method == 'thr_mean':
                max_mask = radon_window > thr_mean
                
            if return_lines and max_mask.any():
                max_mask = shrink_to_centroid(max_mask)
                # now, get centers of lines for each maximum
                max_intensity = radon_window[max_mask.astype(bool)]
                max_delta = delta[np.where(max_mask)[0]]
                max_theta = theta[np.where(max_mask)[1]]
                # get distance of line segment centroids to image center,
                # component orthogonal to line orientation
                max_center = max_delta * np.stack([np.cos(max_theta),
                                                   -np.sin(max_theta)])
                # component parallel to line orientation
                par_com = np.array([get_segment_com(patch, ang, off,
                                                    pad=2)
                                    for ang, off in zip(max_theta, max_delta)])
                # add to coordinates of radon window center
                max_center += par_com * np.stack([np.sin(max_theta),
                                                  np.cos(max_theta)])
                max_loc = max_center.T + np.array([c, r])
                lines_row.append(list(zip(max_intensity, max_theta, max_loc)))
            else:
                lines_row.append([])
            if debug:
                print(r,c)
                for l in lines_row[-1]:
                    print('theta', 180/np.pi*l[1])
                    print('loc', l[2])
                    a, b = np.sin(l[1]), np.cos(l[1])
                    plt.plot(5*np.array([-a, 0, a])+l[2][0]+e_len-c,
                             5*np.array([-b, 0, b])+l[2][1]+e_len-r, color='red', lw=1)
                plt.imshow(patch*disk(e_len), vmin=0, vmax=1)
                plt.show()                
            radon_window = (radon_window * max_mask).sum(axis=0)
            m_row.append((q_matrix * radon_window).sum(axis=2))
        lines.append(lines_row)
        m.append(m_row)
    m = np.stack(m)
    if return_lines:
        return lines, m
    return m


def windowed_radon_nomax(img, radon_matrix, theta=None, method='max-mean'):
    """
    Get coarse-grained anisotropy tensor for img via windowed Radon transform.

    Alternative idea which replaces the maxima detection in the radon
    plane by the standard deviation or the maximum-minimum differenece 
    over the delta-axis (radon transform offset axis). Should be much faster.
    Also, no need to tune parameters.
    
    This is suitable to determine the anisotropy director.
    
    The advantage over simply taking the maximum etc in the radon plane
    is that in case of no anisotropy, the tensor vanishes.
    
    However, this function cannot return the positions and angles
    of detected line segments, only the coarse-grained anisotropy tensor.

    Parameters
    ----------
    img : np.array
        Input image.
    radon_matrix : sparse.csc_matrix
        Sparse matrix representing the Radon transform (remember, it's linear).
        Computed by get_radon_tf_matrix. Determines the size of the radon
        window.
    theta : np.array
        Angles at which radon transform is computed. Defaults to
        np.linspace(0, np.pi, edge_length, endpoint=False) where edge_length
        is the window size (defined by the shape of radon_matrix).
    method : str, optional
        What function to use to reduce delta-axis. Can be 'std'
        'ptp' (i.e. range) or 'max-mean'. Recommended: 'max-mean'.

    Returns
    -------
    m :  np.array of shape (...,..., 2, 2)
        Coarse grained anisotropy tensor. The first two axes are "spatial"
        indices and have the following extent:
            ceil(2*(img_shape - edge_length+1)/(edge_length+1)),
        where edge_length is the radon window size.

    """
    edge_length = np.sqrt(radon_matrix.shape[1]).astype(int)
    e_len = int((edge_length-1)/2)
    if theta is None:
        theta = np.linspace(0, np.pi, int(radon_matrix.shape[0]/edge_length),
                            endpoint=False)
    e_len_cut = int((radon_matrix.shape[0] / theta.size-1)/2)
    # director matrix
    q_matrix = np.array([[np.sin(theta)**2, -np.sin(theta)*np.cos(theta)],
                         [-np.sin(theta)*np.cos(theta), np.cos(theta)**2]])

    # iterate over sub-arrays
    m = []
    for r in np.arange(e_len, img.shape[0]-e_len, e_len+1):
        m_row = []
        lines_row = []
        for c in np.arange(e_len, img.shape[1]-e_len, e_len+1):
            patch = img[r-e_len:r+e_len+1, c-e_len:c+e_len+1]
            radon_window = radon_matrix.dot(patch.flatten())
            radon_window = radon_window.reshape(2*e_len_cut+1, theta.size)
            if method == 'std':      
                reduced = radon_window.std(axis=0)
            if method == 'ptp':      
                reduced = radon_window.ptp(axis=0)
            if method == 'max-mean':      
                reduced = radon_window.max(axis=0)-radon_window.mean(axis=0)

            m_row.append((q_matrix * reduced).sum(axis=2))
        m.append(m_row)
    m = np.stack(m)
    return m