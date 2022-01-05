# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:17:52 2021

@author: nikolas

This module contains functions for surface geometry. The set up is as follow:

A surface, embedded in 3d space, is defined by a parametrization f: R2->R3.
Numerically, f is reprensented by 2 arrays (charts, shape) of shapes
(2, n_y, n_x) and (3, n_y, n_x) respectively. "charts" defines the
parametrization domain and "shape" the corresponding points in 3d. n_x/n_y
are the number of pixels in the x and y direction.

I assume that the "charts" array is a regular rectangular grid whose step
size is given by scale (either a tuple (scale_x, scale_y) or a single number
in which case the spacing is isotropic).

The functions in this module compute matrix representations of 3 main
geometric objects:
    (1) df, or the Jacobian which maps vectors from the parameter domain to R3
    (2) g, the induced metric on the surface
    (3) S, the shape operator characterizing the curvature of the surface

All these matrices are with respect to the standard basis of parameter space
R2, d/dx and d/dy.

Finally, there are also convenience functions which compute the angle between
two vector fields using the induced metric and the norm of a vector.

Unit tests need to be written, but I did check all functions by hand
in the kugelei/shape_and_PIV_analysis_old.ipynb notebook.

"""

import numpy as np
from scipy import ndimage


def get_top_eigvec_nonsym(tens):
    """Get top eigenvector of field (n_y,n_x,d,d) of non-symmetric tensors."""
    kappa, w = np.linalg.eig(tens)
    inds = np.argsort(kappa, axis=-1)
    kappa = np.take_along_axis(kappa, inds, axis=-1)
    w = w.transpose(0, 1, 3, 2)  # inds are x, y, #eigenvec, vec comp's
    w = np.stack([np.take_along_axis(w[:, :, :, i], inds, axis=-1)
                  for i in [0, 1]])[:, :, :, 1]
    w = w * np.sign(w[1])
    w = w.transpose(1, 2, 0)

    return kappa, w


def filter_tensor(tensor, median_footprint):
    """Apply median filter to tensor field."""
    return np.stack([[ndimage.median_filter(tensor[:, :, i, j],
                                            footprint=median_footprint)
                      for i in [0, 1]] for j in [0, 1]]).transpose(2, 3, 0, 1)


def get_jac(shape, scale=1):
    """Compute Jacobian."""
    scale = scale if isinstance(scale, list) else 2*(scale,)
    return np.stack(np.gradient(shape, *scale, axis=(2, 1)), axis=-1)


def get_normal(shape, scale=1):
    """Compute unit normal."""
    grad = get_jac(shape, scale=scale)
    n = np.cross(grad[:, :, :, 0], grad[:, :, :, 1], axisa=0, axisb=0, axis=0)
    n = n / np.linalg.norm(n, axis=0)
    return n


def get_metric(shape, scale=1, median_footprint=np.ones((15, 1))):
    """Compute matrix of induced metric."""
    grad = get_jac(shape, scale=scale)
    g = np.einsum('ixya,ixyb->xyab', grad, grad)
    if median_footprint is not None:
        g = filter_tensor(g, median_footprint=median_footprint)
    return g


def get_b(shape, scale=1, median_footprint=np.ones((15, 1))):
    """Compute matrix of 2nd fundamental form."""
    grad = get_jac(shape, scale=scale)
    n = get_normal(shape, scale=scale)
    hess = np.stack(np.gradient(grad, scale, scale, axis=(2, 1)), axis=-1)
    b = np.einsum('ixyab,ixy->xyab', hess, n)
    if median_footprint is not None:
        b = filter_tensor(b, median_footprint=median_footprint)
    return b


def get_shape_op(shape, scale=1, median_footprint=np.ones((15, 1))):
    """Get matrix of shape operator."""
    b = get_b(shape, scale=scale, median_footprint=median_footprint)
    g = get_metric(shape, scale=scale, median_footprint=median_footprint)
    g_inv = np.linalg.inv(g)
    S = np.einsum('xyij,xyjk->xyik', g_inv, b)
    return S


def get_norm(vf, g):
    """Compute norm of vectorfield vf (shape (n_y, n_x, 2)) using metric g."""
    return np.sqrt(np.einsum('xyi,xyij,xyj->xy', vf, g, vf))


def get_angle(vf1, vf2, g):
    """Compute angle between two vector fields in degrees using metric g."""
    norm1 = get_norm(vf1, g)
    norm2 = get_norm(vf2, g)
    inner = np.einsum('xyi,xyij,xyj->xy', vf1, g, vf2)
    angle = np.arccos((1-1e-5)*inner/(norm1*norm2)) * 360/(2*np.pi)
    return angle
