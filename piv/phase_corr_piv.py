# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:59:20 2020

A python version of Sebastian's MATLAB PIV code, which uses windowed
phase correlations.

@author: nikolas
"""

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter
from math import ceil


def midpoint(x):
    """Apply midpoint rule to array x along axis 0"""
    return np.vstack([x[:1], (x[1:] + x[:-1])/2, x[-1:]])


def phase_corr_piv(frame_1, frame_2, edge_length):
    """
    Compute the 2d PIV field between two frames of a movies.

    This function works convolutively. For each patch of frame_a, the shift
    which maximizes correlation with the corresponding patch of frame_b is
    found using the phase correlation method.

    Parameters
    ----------
    frame_a : 2d np.array
        Reference image.
    frame_b : 2d np.array
        Second frame. Needs to be of same shape as frame_a.
    edge_length : int
         Edge length of PIV box. Must be odd. If velocity is expected to be
         larger, use a larger edge length.

    Returns
    -------
    velocity_field : 3d np.array
        Computed PIV field. First axis indexes vector field components
        (v_x, v_y). The length of the other two axes is given by
        ceil(frame_shape/edge_length), where frame_shape is the x-
        resp. y- dimension of the input image.

    """
    assert edge_length % 2, "Edge length must be odd"
    e_len = int((edge_length-1)/2)

    velocity_field = np.zeros((2, ceil(frame_1.shape[0]/edge_length),
                                  ceil(frame_1.shape[1]/edge_length)))
    for r in range(ceil(frame_1.shape[0]/edge_length)):
        r_min = max(0, (r-1)*edge_length-e_len)
        r_max = min((r+1)*edge_length+e_len+1, frame_1.shape[0]-1)
        for c in range(ceil(frame_1.shape[1]/edge_length)):
            c_min = max(0, (c-1)*edge_length-e_len)
            c_max = min((c+1)*edge_length+e_len+1, frame_1.shape[1]-1)
            patch_1 = frame_1[r_min:r_max, c_min:c_max]
            patch_2 = frame_2[r_min:r_max, c_min:c_max]
            
            velocity_field[:, r, c] = -phase_cross_correlation(
                patch_1, patch_2, return_error=False)

    return velocity_field


def movie_piv(movie, edge_length, rescale_fact=1, sigma=5):
    """
    Compute 2d PIV field for an entire movie using phase_corr_piv.

    The returned velocity field has units pixel * frame rate.

    Parameters
    ----------
    movie : 3d np.array
        Input movie. First axis is taken to be time.
    edge_length : int
        Edge length of PIV box. Must be odd. If velocity is expected to be
        larger, use a larger edge length.
    rescale_fact : float, optional
        Rescale images by given factor (<1) before calculating PIV field. The
        default is 1 (no rescaling).
    sigma : float or None, optional
        Smooth returned velocity field with Gaussian filter of given sigma.
        If sigma is None, no filtering is performed. The default is None.

    Returns
    -------
    velocity_series : 4d np.array
        PIV velocity field. First axis indexes time, second axis indexes vector
        components (v_x, v_y). The length of the other two axes is given by
        ceil(frame_shape/edge_length), where frame_shape is the x-
        resp. y- dimension of the input image. If rescale_fact != 1,
        frame_shape has to be replaced by the rescaled shape. The PIV field for
        time t is the average between phase_corr_piv(movie[t], movie[t+1]) and
        ...(movie[t-1], movie[t]), except at endpoints.
    """
    velocity_series = []
    if rescale_fact < 1:
        movie = np.stack([transform.rescale(x, rescale_fact) for x in movie])

    for frame_1, frame_2 in zip(movie[:-1], movie[1:]):
        velocity_field = phase_corr_piv(frame_1, frame_2, edge_length=edge_length)
        if sigma is not None:
            velocity_field = np.stack([gaussian_filter(x, sigma=sigma)
                                       for x in velocity_field])
        velocity_series.append(velocity_field)
    velocity_series = np.stack(velocity_series) / rescale_fact
    velocity_series = midpoint(velocity_series)

    return velocity_series
