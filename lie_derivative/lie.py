#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:34:43 2020

@author: nikolas, nclaussen@ucsb.edu

This module can calculate the Lie derivative
 of vector and tensor fields on surfaces. This can be done in two ways,
 either using a chart (where the fields are assumed to be defined on a
 rectangular 2d or 3d grid), or directly on a triangular mesh embedded in 3d
 (where the fields are assumed to be expressed in cartesian
 components, defined per vertex). the rectangular grid does not need
 to be equally spaced.

The formulae for the Lie derivative can be found on wikipedia
("coordinate expressions"): https://en.wikipedia.org/wiki/Lie_derivative

In charts, the computation of the Lie derivative is fairly simple and
 can be accomplished using finite differences. On a triangular mesh,
 I use finite-element methods to compute gradients, implemented by the
 highly useful geometry library igl:
     https://libigl.github.io/libigl-python-bindings/
 igl can be easily installed via conda:
     conda install -c conda-forge igl

FAQ:
    - It's not working/giving wrong results
        Try to verify whether the input data conforms to my conventions,
        desribed in the docstring. In particular, check the x-y
        coordinate to row-column index convention. If your data were originally
        using a different coordinate convention, check that you correctly
        transformed them (e.g., if you are mirroring a vector field across
        the y-axis to change conventions, don't forget to also change the sign
        of the y-component.) If that does not help, check whether all the
        tests in test_lie.py are ok.
    - I am a MATLAB user. What do I do?
        0) Install python, if you haven't already. Use anaconda:
            https://docs.anaconda.com/anaconda/install/
        1) Save your iput data (u, t) as .mat files
        2) Open a python interpreter/jupyter notebook (latter is recommended)
        3) Use scipy.io.loadmat to load the .mat files into python:
            https://docs.scipy.org/doc/scipy/reference/tutorial/io.html
        4) Verify your data have the correct numpy.ndarray form. Useful
           commands for reformating arrays include np.reshape, np.transpose,
           np.stack. For a comparision of python to MATLAB arrays, see:
            https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
        5) Import the lie_derivative module and apply the desired function to
           your data. Read the documentation via help(function)
        6) Save your data as .mat file with scipy.io.savemat.
    - It is 20XX and the igl library has changed/doesn't install.
        lie_chart is unaffected. You can use any other library capable of
        computing a gradient of a scalar function on a triangular mesh, and
        need only update the "my_grad = ..." line in lie(...).
"""

import string
import numpy as np
import igl


def tri_grad(t, v, f, grad_matrix=None):
    """
    Calculate gradient of function defined on vertices of triangular mesh.

    Assumes a time dependent field, time-independent case can be handled by
    passing an array with only one time point. If a tensore is passed,
    each tensor component is interpreted as an individual scalar function.

    Parameters
    ----------
    t : np.array of shape (#timepoints, #vertices,...)
        scalar function or tensor
    v : np.array of shape (#vertices, dim)
        vertices.
    f : np.array of shape (#faces, 3)
        faces.
    grad_matrix : scipy.sparse, optional
        Gradient operator. The default is None (calculate g from v, f).

    Returns
    -------
    np.array of shape (dim, #timepoints, #vertices, ...)
        gradient of scalar function/tensor defined on vertices.

    """
    # fix these dimension variables before reshaping
    n_t = t.shape[0]
    n_ind = t.shape[2:]

    if grad_matrix is None:  # calculate gradient operator
        grad_matrix = igl.grad(v, f)

    # swap axes and reshape so that the sparse gradient op can be applied
    t = t.swapaxes(0, 1)
    t = t.reshape((t.shape[0], np.prod(t.shape[1:])))
    # calculate the gradient of t by matrix multiplication, then reshape
    grad_t = grad_matrix.dot(t)
    # check this 'F' - copied from tutorial
    grad_t = grad_t.reshape((f.shape[0], v.shape[1], grad_t.shape[1]),
                            order='F')
    # now, average onto vertices. Need to iterate over all other axes
    grad_t = np.stack([igl.average_onto_vertices(v, f, grad_t[:, :, i])
                       for i in range(grad_t.shape[2])], axis=2)
    # finally, reshape into original shape
    grad_t = grad_t.reshape((v.shape[0], v.shape[1], n_t)+n_ind)
    # shape is now  (#vertices, dim, #timepoints, ...)
    grad_t = grad_t.swapaxes(0, 1).swapaxes(1, 2)

    return grad_t


def lie_derivative(u, tens,
                   covariant=False, density=False,
                   v=None, f=None,
                   delta_t=1, grid_coords=None):
    """
    Compute Lie derivative of an arbitrary rank tensor field in 2d or 3d.

    Can calculate the Lie derivative either in a chart or on an embedded
    triangular mesh. If #time points == 1, the autonomous Lie derivative is
    calculated.

    Chart: input the values of t, u in that chart (assumed to be a 2d/3d
    rectangular  grid, not necessarily evenly spaced, with
    n_x/n_y/n_z = #points along x-axis/y-axis/z-axis).

    Mesh: input u, t in cartesian components, defined on vertices v of a mesh.
    Assumes that u and t are expressed in cartesian/embedding components,
    i.e. NOT in a local attached to mesh vertices or faces.

    By default, the code assumes the 'chart' case. To , pass the vertices and
    faces via the arguments v, f.

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

    The Lie derivative can be calculated for tensor fields of any rank.
    The number of tensor components is inferred  automatically from the input,
    but the user must specify whether the tensor is co-, contravariant or mixed
    (i.e. which indices are "upper"/"lower") via the "covariant" keyword.

    Lie derivatives of tensor densities are computed via the "density" keyword.

    Parameters
    ----------
    u : np.array of shape (#time points, n_y, n_x, 2),
                          (#time points, n_y, n_x, n_z, 3)
                       or (#time points, #vertices, 3)
        Time dependent vector field.
    tens : np.array of shape (#time points, n_y, n_x, ...),
                             (#time points, n_y, n_x, n_z, ...)
                          or (#time points, #vertices, ...)
        Time dependent tensor field, function, vector, or rank(r,s) tensor,
        depending on the shape. ... are the tensor component indices.
        No indices: tens is a scalar function, index: tens is a (co)vector ...
    covariant : bool or list of bool, optional
        Which tensor indices are covariant. True: index is covariant,
        False: index is contravariant. If a single True/False is supplied,
        then all/no indices are covariant. The default is False.
    density : bool, optional
        Whether the input transforms as a tensor density. The default is False.
    v : np.array of shape (#vertices, dim) or None
        Vertices of triangular mesh. If None is passed, the code assumes
        that fields are defined on a rectangular grid ('chart' case).
    f : np.array of shape (#faces, dim) or None
        Faces of triangular mesh. Each row is a triple of vertex indices
        belonging to a face. If None is passed, the code assumes
        that fields are defined on a rectangular grid ('chart' case).
    delta_t : float, optional
        Spacing in time. Choose so that the output is compatible with the units
        of u, e.g. if u is in pixels/minute and the sample rate is 30s, take
        delta_t=0.5. The default is 1.
    grid_coords : list of np.arrays, optional
        Coordinates of rectangular grid, grid_coords[0] = y-axis coords,
        grid_coords[1] = x-axis, ... (e.g. grid_coords[0][i] == y-coordinate
        of tens[#time point,i,:]). Only used if v is None and f is None.
        The default is unitary spacing in all dimensions, with x-coords are
        ascending and y-coordinates descending (see above).

    Returns
    -------
    lie_tens : np.array of same shape as tens.
        Lie derivative L_u tens.

    """
    # preliminary argument parsing
    dim = u.shape[-1]
    tensor_rank = (len(tens.shape)-dim-1 if (v is None and f is None)
                   else len(tens.shape)-2)
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
    if (v is None and f is None):
        assert len(u.shape) == 2+dim and len(tens.shape) >= 1+dim, \
               "shapes incorrect"
        assert u.shape[:1+dim] == tens.shape[:1+dim], \
               "u, tens shapes incompatible"

        def my_grad(x):
            # axis order is due to coordinate convention
            return np.stack(np.gradient(x, *grid_coords,
                                        axis=(2, 1)+tuple(range(2+1, dim+1))))
        s_ind = 'xyz'[:dim]
    else:
        assert len(u.shape) == 3 and len(tens.shape) >= 2, "shapes incorrect"
        assert u.shape[:2] == tens.shape[:2], "u,t shapes incompatible"
        grad_matrix = igl.grad(v, f)

        def my_grad(x):
            return tri_grad(x, v, f, grad_matrix=grad_matrix)
        s_ind = 'v'

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
