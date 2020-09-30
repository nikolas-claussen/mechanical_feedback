#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:34:43 2020

@author: nikolas, nclaussen@ucsb.edu

This is a version of lie_derivative.py with reduced functionality (no
 ability to deal with data on triangular meshes) which does not have any
 external dependencies except numpy.
"""

import numpy as np
import string


def lie_chart(u, tens,
              covariant=False, density=False,
              delta_t = 1,
              grid_coords=None):
    """
    Compute Lie derivative of tensor field using finite differences.

    Numerically computes the Lie derivative of a time dependent arbitrary
    rank tensor field t along vector field u, in 2 or 3 dimensions.
    Can calculate the Lie derivative on R^2/R^3 (if you are working
    with surfaces, you need to use a chart). If #time points == 1, the
    autonomous Lie derivative is calculated.

    Input the values of t, u in the chart (assumed to be a 2d/3d rectangular
    grid, not nec. evenly spaced, with n_x/n_y = #points along x-axis/y-axis).

    IMPORTANT CONVENTIONS FOR CHARTS: Arrays/matrices represent images/tensors
    defined on points on a Cartesian grid. For a python  arr:
        arr[i,j] = row i, column j
    My convention for the map array indices -> cartesian coordinates is that
    row == y-axis, column == x-axis. The x/y - coordinates of a row/column
    are specified using the grid_coords argument. In case of a 3d grid, the
    array index order is y-axis - x-axis - z-axis. By default, x-coords are
    ascending and y-coordinates descending:
        arr[i,j] coorresponds to x-coord = j, y-coord = #rows-i

    Beyond the indexes standing for positions on the rectangular grid, there
    are also vector and tensor indices. Here, my convention is
        - vector index (k=0) == x-component, (k=1) == y-component,
          (k=2) == z-component, ...

    In case of 3d, the array index order is y-axis - x-axis - z-axis, and
    the y-axis is still "flipped".

    The Lie derivative can be calculated for all kinds of tensor fields.
    The number of tensor components is inferred  automatically from the input,
    but the user must specify whether the tensor is co-, contravariant or mixed
    (i.e. which indices are "upper"/"lower"). This is done with the "covariant"
    keyword.

    Lie derivatives of tensor densities can be computed with the "density"
    keyword.

    Parameters
    ----------
    u : np.array of shape (#time points, n_y, n_x, 2) or
                          (#time points, n_y, n_x, n_z,3)
        Time dependent vector field.
    tens : np.array of shape (#time points, n_y, n_x [n_z,], ...) or
                             (#time points, n_y, n_x, n_z, ...)
        Time dependent tensor field, function, vector, or rank(r,s) tensor,
        depending on the shape. ... are the tensor component indices.
        No indices: tense is a scalar function, 1 index:  (co)vector, ...
    covariant : bool or list of bool, optional
        Which tensor indices are covariant. True: index is covariant,
        False: index is contravariant. If a single True/False is supplied,
        then all/no indices are covariant. The default is False.
    density : bool, optional
        Whether the input transforms as a tensor density. The default is False.
    delta_t : float, optional
        Spacing in time. Choose so that the output is compatible with the units
        of u, e.g. if u is in pixels/minute and the sample rate is 30s, take
        delta_t=0.5. The default is 1.
    grid_coords : list of np.arrays, optional
        Coordinates of rectangular grid, grid_coords[0] = y-axis coords,
        grid_coords[1] = x-axis, ... (e.g. grid_coords[0][i] == y-coordinate
        of tens[#time point,i,:]). The default is unitary spacing in all
        dimensions, with x-coords are ascending and y-coordinates descending
        (see above).

    Returns
    -------
    lie_tens : np.array of shape (#time points, n_y, n_x, ...) or
                                 (#time points, n_y, n_x, n_z,...)
        Lie derivative L_u tens.
    """
    # preliminary argument parsing
    dim = u.shape[-1]
    tensor_rank = len(tens.shape)-dim-1
    # depends on whether or not in_chart!
    covariant = (covariant if isinstance(covariant, list)
                 else (covariant,)*tensor_rank)
    if grid_coords is None:  # default grid coordinates
        grid_coords = [np.arange(tens.shape[2]),
                       np.arange(tens.shape[1])[::-1]]
        if dim == 3:
            grid_coords.append(tens.shape[3])

    grid_coords = (grid_coords if isinstance(grid_coords, list)
                   else list(grid_coords))
    assert len(u.shape) == 2+dim and len(tens.shape) >= 1+dim, \
           "shapes incorrect"
    assert u.shape[:1+dim] == tens.shape[:1+dim], \
           "u, tens shapes incompatible"

    def my_grad(x):
        # axis order is due to coordinate convention
        return np.stack(np.gradient(x, *grid_coords,
                                    axis=(2, 1)+tuple(range(2+1, dim+1))))
    s_ind = 'xyz'[:dim]

    # get matrix of partial derivatives of vector and tensor field
    dc_u = my_grad(u)
    dc_tens = my_grad(tens)

    # first term in Lie derivative, always the same
    # f-strings create the right index expression for chart & trimesh case
    uc_dc_tens = np.einsum(f't{s_ind}c,ct{s_ind}...->t{s_ind}...', u, dc_tens)
    # partial time derivative - 0 in time-independent case
    dt_tens = np.gradient(tens, delta_t, axis=0) if tens.shape[0] > 1 else 0
    # density term
    div_u = np.einsum(f'ct{s_ind}c->t{s_ind}', dc_u)*tens if density else 0
    # sum up all the contributions so far
    lie_tens = dt_tens+uc_dc_tens+div_u

    # Now, interate over the co- and contra-variant indices.
    # The challenge is to construct the correct np.einsum str
    tensor_indices = string.ascii_lowercase[9:9+tensor_rank]
    base_str = f'ct{s_ind}i,t{s_ind}'+tensor_indices
    for index, is_cov in enumerate(covariant):
        contr_index = tensor_indices[index]
        if is_cov:
            # rename the tensor index to be contracted to i
            contr_str = base_str.replace(contr_index, 'i')
            # make the target index order
            target_str = f'->t{s_ind}'+tensor_indices.replace(contr_index, 'c')
            lie_tens += np.einsum(contr_str+target_str, dc_u, tens)
        else:
            contr_str = base_str.replace(contr_index, 'c')
            target_str = f'->t{s_ind}'+tensor_indices.replace(contr_index, 'i')
            lie_tens += - np.einsum(contr_str+target_str, dc_u, tens)
    return lie_tens
