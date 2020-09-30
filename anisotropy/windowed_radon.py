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

A windowed Radon transform subdivides an input image domain Omega into many small
rectangles and comutes the Radon transform of each. Thus, it can capture
the local anisotropy of the image. The output of the "true" Radon transform
of a function f(x,y) is anothe function Rf(alpha,s) where alpha is the line
angle and s its distance to the origin along the x-axis. 


"""

import numpy as np
from skimage.transform import radon
from scipy.special import erf
from joblib import Parallel, delayed       


def functional_calc(a, f, is_vectorized=True):
    """
    Apply a function f to hermitian matrix by applying it to eigenvalues.
    
    Parameters
    ----------
    a : np.array of shape (..., M, M)
        (Array of) hermitian matrices.
    f : callable
        Real or complex valued function of a single real variable.
    is_vectorized : bool, optional
        Whether f can be applied directly to arrays, e.g. np.sin 
        or lambda x: x**2.

    Returns
    -------
    f_a : np.array of same shape as a
        f(a).

    """
    w, v = np.linalg.eigh(a) 
    w = f(w) if is_vectorized else np.vectorize(f)(w)
    w = np.einsum('...i,ij->...ij', w, np.eye(a.shape[-1]))
    return np.einsum('...ij,...jk,...lk->...il', v, w, v)


def tensorify_radon(angles, window_len, sigma=1, 
                    maxima_only=False):
    """
    Convert a randon transformed function (i.e. array of values of Rf for some function f)
    into a coarse-grained anisotropy tensor m, see https://doi.org/10.7554/eLife.27454.
    This function returns the matrix representing the linear map Rf -> m. The
    Radon transform is presumed to be taken over a disk of radius L. 

    Each point Rf(alpha, alpha) in the Radon transform plane represents the integral
    of f along the line paramterized by (alpha = angle of line normal to x-axis,
    delta = minimal distance of line to origin). In the calculation of the anisotropy
    tensor, each line is weighted by exp(1/2 * (r/sigma)**2), where r is the
    distance of the line to the origin, averaged over the points on the line
    within the L-disk. 
    
    For the coordinates/shape of the Radon transform array, the skimage 
    conventions are used, i.e. skimage.transform.radon, in particular with circle=True.

    Parameters
    ----------
    angles : np.array
        Angles for which to calculate the radon transform. The default is 
        np.linspace(0,180,60).
    window_len : int
        HALF of radius of window for windowed Radon transform. The default is 20.
    sigma : float, optional
        Smoothing parameter of Gaussian filter. The default is 1.
    maxima_only : bool, optional
        Whether to only include maxima of the Radon transform (corresponding)
        to distinct lines in the original image or to integrate over the 
        entire range of (alpha,delta). The default is False.

    Returns
    -------
    m_matrix : np.array of shape (2,2,)+radon_tf_shape
        Matrix mapping Rf -> m, via np.sum(m*Rf, axis=(2,3)).
    """
    
    if maxima_only:
        raise NotImplementedError
    
    # coordinates for radon_tf array, the window_len+1 ensures the correct placement
    # of origin according to skimage conventions. Convert angles to rad
    alpha, delta = np.meshgrid((np.pi/180)*angles,
                               np.arange(-window_len+1, window_len+1, dtype=float)[::-1]) 
    
    # calc weights - disk size according to skimage convention
    ln = np.sqrt(window_len**2/4+delta**2) # length of line inside window
    ln_weight = 1/(4*ln) * (erf(ln/np.sqrt(2*sigma**2)) - erf(-ln/np.sqrt(2*sigma**2)))
    weights = np.exp(-delta**2/(2*sigma**2)) * ln_weight
    
    # trapezoidal rule
    trap_X, trap_Y = (np.ones(alpha.shape)/2, np.ones(alpha.shape)/2)
    trap_X[:, 1:-1] = 1
    trap_Y[1:-1, :] = 1
    trap = trap_X*trap_Y
        
    # tensor components
    n_x = np.sin(alpha)
    n_y = -np.cos(alpha)
    m_matrix = np.stack([[n_x**2, n_x*n_y], [n_x*n_y, n_y**2]])

    m_matrix = m_matrix * weights * trap
    
    return m_matrix
    

def windowed_radon(im, angles=np.linspace(0,180,60), window_len=21, sigma=10,
                   step=8, p=1, n_jobs=4):
    """
    Compute coarse-grained anisotropy tensor from image via windowed Radon 
    transform, see https://doi.org/10.7554/eLife.27454. This function first
    calculates a windowed Radon transform, the computes a coarse-grained
    anisotropy tensor for each Radon patch according to tensorify_radon.
    
    Unfortunately, this functions is not very fast, taking 3 mins for a 1k * 1k
    image with 60 projection angles and window size 21.
    To speed things up, decrease the number of angles (linear acceleration)
    or increase the step size (quadratic acceleration). windowed_radon is a 
    linear function, so one can average images before computing it, delivering
    another speed up. Also, one can downsample an initial high-res image before
    putting it into the windowed radon, as long as that does not blur the 
    anisotropy

    Parameters
    ----------
    im : np.array
        Input image.
    angles : np.array, optional
        Angles for which to calculate the radon transform. The default is 
        np.linspace(0,180,60).
    window_len : int, optional
        HALF of radius of window for windowed Radon transform. The default is 20.
    sigma : float, optional
        Standard deviation of Gaussian filter. The default is 10.
    step : int, optional
        Downsample factor - spacing of points where the coarse grained tensor 
        is evaluated. The default is 8.
    p : int, otional
        Power to raise radon transform I(alpha, delta) when calculating
        coarse grained tensor. Higher values emphasize maxima (edges).
        The default is 1.
    njobs : int, optional
        Number of cores to use during parallel calculation og windowed
        Radon transform. The default is 4.
    Returns
    -------
    m :  np.array of shape (...,...,2,2)
        Coarse grained anisotropy tensor.

    """
    # Main issue here is that the skimage radon transform is fairly slow,
    # approx 15-20 ms per window with default settings from above. This
    # is due to the use of scipy.ndimage image rotation, which uses interpolation.
    # There are much faster alg's, but I would have to implement them myself.
    # Alternatively, one could use th Hough line instead of the Radon transform,
    # which is x20 fast, but:
    # a) It generates more artifacts, less mathematically controlled
    # b) Uses a different coordinate convention (pain)
    
    # Note: Radon transform and calc of m from Rf are both linear operations
    # so that one can average images first, then calculate the anisotropy tensor

    
    m_matrix = tensorify_radon(angles, window_len, sigma=sigma)
    # make circular mask for radon transform
    X, Y = np.meshgrid(np.arange(-window_len,window_len), 
                       np.arange(-window_len,window_len))
    mask = (X**2 + Y**2 <= window_len**2).astype(int) 
    def loop_op(im_window):
        return np.sum(m_matrix * radon(im_window*mask, theta=angles, circle=True)**p,
                      axis=(2,3))
    
    # iterate over sub-arrays spaced by downsample factor
    m = []
    for i in np.arange(window_len, im.shape[0]-window_len, step):
        m.append(Parallel(n_jobs=n_jobs)(delayed(loop_op)(im[i-window_len:i+window_len, 
                                                             j-window_len:j+window_len])
                                    for j in np.arange(window_len, im.shape[1]-window_len, step)))
    # stack and 
    m = np.stack(m)
    if p != 1:
        m = functional_calc(m, lambda x: x**(1/p))
    
    return m
