# -*- coding: utf-8 -*-

"""
Created on Thu May 28 17:50:52 2020

@author: nikolas

This file contains a few, somewhat unrelated, convenience functions.

"""

import numpy as np
from scipy.ndimage.filters import convolve1d
from collections import Iterable

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.patches as mpatches

from skimage import transform
from PIL import Image


### Array and list manipulation


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

def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


### Finite differences


def midpoint(x): return np.vstack([x[:1], (x[1:] + x[:-1])/2, x[-1:]])


def hessian(x):
    """
    Calculate the hessian matrix with finite differences.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for j, grad_kj in enumerate(tmp_grad):
            hessian[k, j, :, :] = grad_kj
    return hessian.transpose((2, 3, 0, 1))


### Smoothing & Interpolation


def smooth(x, window_len=11, window='hanning', axis=-1):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x : ndarray
        the input signal
    axis : int
        which axis to smooth along
    window_len: int
        the dimension of the smoothing window; should be an odd integer
    window : str
        the type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman'. flat window will produce a moving average smoothing.

    Returns
    -------
        y : ndarry
            the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    See Also
    --------
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    """
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window not recognized")
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = convolve1d(x, w/w.sum(), mode='nearest', axis=axis)
    return y

def interpolate_function_list(function_list, times=None):
    """
    1d interpolation between a list of _functions_, not function values.

    Function does simple linear equidistant interpolation in 1d (in "time").
    However, instead of interpolating between a list of function _values_,
    it interpolates between a list of _functions_ (e.g. functions of
    spatial coordinate x). Use case is: time interpolation between different
    spatial interpolator instances.

    Parameters
    ----------
    function_list : ndarray of dtype object
        List of functions to be interpolated, with equal return type,
        passed as  1d np.array.
    times : np.array of dtype float, optional
        Time points corresponding to function_list entries, defaults to
        np.arange(len(function_list)). Must be sorted ascendingly.

    Returns
    -------
    function, call signature (t,x)
        Time-interpolated function.

    """
    if times is None:
        times = np.arange(len(function_list))
    max_i = len(times)-1

    def f(t, x):
        # search for matching entries in time array
        i = np.searchsorted(times, t, side='right')
        i = i if i < max_i else max_i  # to avoid errors at domain edge
        delta = times[i] - times[i-1]
        a = (times[i]-t)/delta   # calculate interp weights
        b = (t - times[i-1])/delta
        return a*function_list[i-1](x) + b*function_list[i](x)
    return f


### Image & Tensor Field Processing


def get_cytosolic_intensity(selection, size=8):
    """Get cytosolic background intensity for image of membrane bound marker"""
    wht = morphology.white_tophat(selection, selem=morphology.disk(size))
    seed = np.copy(selection-wht)
    seed[1:-1, 1:-1] = (selection-wht).max()
    filled = morphology.reconstruction(seed, selection-wht, method='erosion')
    
    return np.median(filled)

def resize_tensor(m, shape):
    """Resize tensor of shape (n_y, n_x, 2, 2)."""
    m_new = np.zeros(shape+(2, 2))
    kwarg = {'anti_aliasing': True}
    m_new[:, :, 0, 0] = transform.resize(m[:, :, 0, 0], shape, **kwarg)
    m_new[:, :, 0, 1] = transform.resize(m[:, :, 0, 1], shape, **kwarg)
    m_new[:, :, 1, 0] = transform.resize(m[:, :, 1, 0], shape, **kwarg)
    m_new[:, :, 1, 1] = transform.resize(m[:, :, 1, 1], shape, **kwarg)
    return m_new

def filter_field(field, image_filter, kwargs={}):
    """Apply a 2d image filter to all components of a vector or matrix field of shape (n_y, n_x, ...)"""
    is_scalar = (len(field.shape) == 2)
    is_vector = (len(field.shape) == 3)
    is_tensor = (len(field.shape) == 4)

    if is_scalar:
        output = image_filter(field, **kwargs)
    
    if is_vector:
        d1 = range(field.shape[-1])
        output = np.stack([image_filter(field[...,i], **kwargs) for i in d1], axis=-1)

    if is_tensor:
        d1 = range(field.shape[-2])
        d2 = range(field.shape[-1])
        output = np.stack([np.stack([image_filter(field[...,i, j], **kwargs) for i in d1], axis=-1)
                           for j in d2], axis=-1)
    return output

def functional_calc(a, f, is_vectorized=True):
    """
    Apply a function f to field of hermitian matrices by applying it to eigenvalues.

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


### Plotting


def axisEqual3D(ax):
    """
    Set the aspect ratio of a matplotlib 3 plot to 'equal'.

    Parameters
    ----------
    ax : matplotlib Axes3D axis object

    Returns
    -------
    None.

    """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in
                        'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def curly_arrow(ax, start, end, arr_size=1, n=5, linew=1., width=0.1,
                color='gray',):
    """
    Create curly/wiggly arrow patch which can be added to to matplotlib axis.

    Example of usage:
    fig, ax = plt.subplots()
    ax.add_patch(curly_arrow(ax, (20, 20), (2, 10), n=10, arr_size=2))

    Parameters
    ----------
    ax : matplotlib axis object
        axis to which to add a curly arrow
    start : 2-tuple
        x, y coordinates of arrow start
    end : 2-tuple
        x, y coordinates of arrow end
    arr_size : float, optional
        size of the arrow head
    n : int, optional
        number of wiggles
    linew : float, optional
        linewidth of the arrow
    width : float
        "Amplitude of wiggles"
    color : str
        Color of arrow.

    Returns
    -------
    patch : matplotlib.patches.PathPatch
        Arrow patch
    """
    xmin, ymin = start
    xmax, ymax = end
    dist = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2)
    n0 = dist / (2 * np.pi)

    x = np.linspace(0, dist, 151) + xmin
    y = width * np.sin(n * x / n0) + ymin
    line = plt.Line2D(x, y, color=color, lw=linew)

    del_x = xmax - xmin
    del_y = ymax - ymin
    ang = np.arctan2(del_y, del_x)

    line.set_transform(mpl.transforms.Affine2D().rotate_around(xmin, ymin, ang)
                       + ax.transData)
    ax.add_line(line)
    verts = np.array([[0, 1], [0, -1], [2, 0], [0, 1]]).astype(float)*arr_size
    verts[:, 1] += ymax
    verts[:, 0] += xmax
    path = mpath.Path(verts)
    patch = mpatches.PathPatch(path, fc=color, ec=color)
    patch.set_transform(mpl.transforms.Affine2D().rotate_around(xmax, ymax, ang) + ax.transData)
    return patch

def blender_format(load_path, save_path, fact=4):
    """Format image so that it is compatible with my blender model of the Drosophila egg"""
    img = Image.open(load_path)
    img = img.resize((fact*217, fact*256),Image.ANTIALIAS)
    img = np.array(img)
    square = np.zeros((fact*256, fact*256, 4), dtype=np.uint8)
    square[:,:fact*217,:] = img
    square = Image.fromarray(square)
    square.save(save_path)


### Geometry


def is_in_triangle(p, p0, p1, p2):
    """
    Check if point p is in triangle p0, p1, p2.

    Works by computing the barycentric coordinates of p w.r.t. triangle p0, p1, p2
    """
    Area = 0.5 *(-p1[1]*p2[0] + p0[1]*(-p1[0] + p2[0]) + p0[0]*(p1[1] - p2[1]) + p1[0]*p2[1])
    s = 1/(2*Area)*(p0[1]*p2[0] - p0[0]*p2[1] + (p2[1] - p0[1])*p[0] + (p0[0] - p2[0])*p[1])
    t = 1/(2*Area)*(p0[0]*p1[1] - p0[1]*p1[0] + (p0[1] - p1[1])*p[0] + (p1[0] - p0[0])*p[1])
    return (s < 1) & (t < 1) & ((1-s-t) < 1)


def create_regular_grid_mesh(n_x, n_y, glue=None, return_vertices=True):
    """
    Create a triangulation for a regular, rectangular grid.
    
    Contains n_x pointsin x- and n_y point in y-direction. Vertices of the 
    triangulation are returned, too (but can of course be ignored). 
    The vertices are: [(0,0), (1,0), ..., (0,1), ... (n_x, n_y)], 
    in that order. The triangulation looks like this:
        
     X.........X.........X
     .       - .       - .
     .     -   .     -   .
     .   -     .   -     .  and so on.
     . -       . -       .
     X.........X.........X
    
    Optionally, the top/bottom or left/right edges can be glued together
    to obtain the faces of a cylinder (this is useful for the embryo 
    cylindrical maps). 

    This function is equivalent to Noah's MATLAB functions  
    defineFacesRectilinearMesh.m & closeRectilinearCylMesh.m
    
    Parameters
    ----------
    n_x : int
        number of points in x-direction.
    n_y : int
        number of points in x-direction.
    glue : str or None, optional
        Which edges to glue together Possibibilies are
        'top-bottom', 'left-right', and None.
        The default is None.
    return_vertices : bool
        Whether to return vertices. The default is True

    Returns
    -------
    vertices : np.array of shape (2,n_x*n_y), optional
        Vertices of the triangulation
    faces : np.array of ints, shape (3, n_faces)
        Faces of the triangulation.
    """
    # generate vertices
    vertices = np.vstack(chain(*[[np.array([i, j]) for i in range(n_x)]
                                 for j in range(n_y)]))
    
    # Generate faces. 2 types, triangles pointing left/down or righ/up.
    left_down = [np.array([k, k+1, k + n_x]) for k in 
                 list(chain(*[[i+n_x*j for i in range(n_x-1) ]
                              for j in range(n_y-1)]))]
    right_up = [np.array([k, k+1-n_x, k+1,]) for k in 
                list(chain(*[[i+n_x*j for i in range(n_x-1) ]
                             for j in range(1,n_y)])) ]
    
    if glue is None:
        faces = np.vstack([left_down, right_up])
        if return_vertices:
            return vertices, faces
        else:
            return faces
    
    if glue == 'top-bottom':  # glue top and bottom edges of the mesh 
        left_down_seam = [np.array([i+n_x*(n_y-1), i+n_x*(n_y-1)+1, i])
                          for i in range(n_x-1)]
        right_up_seam = [np.array([i, i+n_x*(n_y-1)+1, i+1])
                         for i in range(n_x-1)]
    
    if glue == 'left-right': # glue left and right edges of the mesh
        left_down_seam = [np.array([(i+1)*n_x-1, i*n_x, (i+2)*n_x-1])
                          for i in range(n_y-1)]
        right_up_seam = [np.array([(i+2)*n_x-1, i*n_x, (i+1)*n_x])
                         for i in range(n_y-1)]

    faces = np.vstack([left_down, right_up, left_down_seam, right_up_seam])
    if return_vertices:
        return vertices, faces
    else:
        return faces