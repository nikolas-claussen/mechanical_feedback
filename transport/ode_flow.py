#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Sep 21 15:10:09 2020

@author: nikolas

Given a 2d vector field u(x,t), sampled on a rectangular grid Omega, obtain
the flow phi_t: Omega -> Omega, a family of self-mappings of Omega.
The flow is obtained by integrating the ODE dx/dt = u(x,t) for initial
conditions x_0 ranging over the rectangular grid. Periodic boundary conditions
are possible, so that Omega can be a rectangle, cylinder or torus.
Once the phi is obtained, it can be used to pull back scalar, vector or
tensor fields along the path lines of the vector field.

When using this code, it is advisable to use dictionaries to bundle
coordinates (x, y, t...) and fields together.

"""

import string
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from joblib import Parallel, delayed


def rect_flow(u, x, y, t, t_span, t_eval=None,
              initial_cond_x=None, initial_cond_y=None,
              pbc=(False, False),
              interp_kwargs=None, solver_kwargs=None,
              dense_output=False,
              n_jobs=4):
    """
    Obtain flow of a time-dependent vector field u on a 2d rectangular grid.

    This function is a wrapper around scipy.integrate.solve_ivp (see scipy
    docs). For the array u of vector field values on grid points, it uses the
    convention col == y-coordinate, row == x-coordinate.

    Can deal with periodic boundary conditions using the pbc argument
    (see below for use), so the domain can be a rectangle, cylinder or torus.
    In case of pbc's, 2 objects are returned - an array of flow lines on the
    covering space of the domain (for a cylinder, an infinite strip) and a
    function which can fold back flow lines onto the periodic domain
    (the 1st brillouin zone, so to speak). This way, partial derivatives, i.e.
    the jacobian of phi, can be properly calculated with finite differences.

    In case of ValueError, a trajectory might have left the domain. Try
    decreasing the ODE solver max_step, enlarging the domain or shrinking
    the range of initial conditions. By default, the interpolator used,
    'RectBivariateSpline' extrapolates outside of its domain, so caution
    is advised.

    Parameters
    ----------
    u :  ndarray of shape (n_timepoints, n_rows, n_cols, 2)
        Vector field values at grid points. u[:,:,:,0] is the x-component,
        u[:,:,:,1] the y-component. If u is a 3d array (len(u.shape) == 3),
        the vector field is assumed to be constant in time.
    x : ndarray of shape (n_cols,)
        Array of x-coordinate values. x[i] == x-coord for u[:,:,i].
        Coordinates must be strictly ascending.
    y : ndarray (n_rows,)
        Array of y-coordinate values. y[i] == y-coord for u[:,i,:].
        Coordinates must be strictly ascending.
    t : ndarry of shape (n_timepoints,)
        Array of times at which vector field is sampled, in strictly
        ascending order.  If the vector field is time independent
        (len(u.shape) == 3), this argument is ignored (just pass None).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf).
    t_eval : np.array, otional.
        Times at which to evaluate the flow. Defaults to
        np.arange(t_span[0], t_span[1]+1), so phi[0] == initial conditions.
    initial_cond_x : ndarrray of dimension 1, optional
        Array of x-coordinate values of ODE initial conditions. If None,
        defaults to the same value as 'x'.
    initial_cond_y : ndarrray of dimension 1,, optional
        Array of y-coordinate values of ODE initial conditions. If None,
        defaults to the same value as 'y'.
    pbc : 2-tuple of bools, optional
        Whether domain is periodic in x- or y-direction (pbc[0] resp. pbc[1]).
        Assumes that the periodic coordinate starts or ends at 0, e.g.
        periodicity in the x-direction requires x[0]==0 or x[-1]==1.
    interp_kwargs  : dict, optional
        Parameters for the scipy.interpolate.RectBivariateSpline interpolator,
        see scipy docs. In particular, one can choose the order
        (linear, cubic, ...) and the smoothing paramter.
    solver_kwargs : dict, optional
        Parameters for the scipy ODE solver solve_ivp, see scipy docs.
        The default is {'method'='LSODA', 'rtol'=1e-05}.
    dense_output : bool, optional
        Whether to return dense-in-time output, i.e. for each point in the
        initial coniditon grid, a function x(t), y(t), in the form of a
        scipy.interpolate.OdeSolution object.
    n_jobs : int, optional
        Number of jobs in parallel solution of ODEs.

    Returns
    -------
    phi : ndarray of shape (#timepoints, n_rows, n_cols, 2)
        Flow of vector field. phi[:,i,j,:] is the trajectory of
        point y[i], y[j], with phi[:,i,j,0] being the x-coordinate
        and phi[:,i,j,1] the y-coordinate. If dense_output is True, return
        array (n_rows, n_cols, 2) of OdeSolution objects.
    fold_pbc : callable,
        Returned only if any(pbc), function which maps array of flow lines phi
        back into the fundamental periodic domain ("1st Brillouin zone").

    """
    # preliminary argument parsing
    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)
    if initial_cond_x is None:
        initial_cond_x = x
    if initial_cond_y is None:
        initial_cond_y = y
    initial_cond_x = initial_cond_x.astype(float, copy=False)
    initial_cond_y = initial_cond_y.astype(float, copy=False)
    # ensure correct dtype - if coords are ints, errors can occur
    shift = np.array([np.min(x), np.min(y)])
    x = x - shift[0]
    y = y - shift[1]
    initial_cond_x = initial_cond_x - shift[0]
    initial_cond_y = initial_cond_y - shift[1]
    # shift coordinates x, y so that they are all positive and start (or end)
    # at 0. Necessary for the modulo operation which enforces periodicity.
    if t_eval is None:
        t_eval = np.arange(t_span[0], t_span[1]+1)
    if interp_kwargs is None:
        interp_kwargs = {}
    if solver_kwargs is None:
        solver_kwargs = {'method': 'RK45', 'rtol': 1e-05}
    solver_kwargs['t_eval'] = t_eval
    solver_kwargs['vectorized'] = True
    if dense_output:
        solver_kwargs['solver_kwargs'] = True
    period = np.array([np.abs(y[0]-y[-1]) if pbc[1] else np.inf,
                       np.abs(x[0]-x[-1]) if pbc[0] else np.inf])
    # trajectories will be computed modulo these periods

    # set up interpolator
    if len(u.shape) == 3:  # time-independent vector field
        interp_x = RectBivariateSpline(y, x, u[:, :, 0], **interp_kwargs)
        interp_y = RectBivariateSpline(y, x, u[:, :, 1], **interp_kwargs)

        def rhs(t, pts):
            pts = np.fmod(pts.T, period)  # map point back into cylinder
            return np.stack([interp_y(pts[:, 0], pts[:, 1]),
                             interp_x(pts[:, 0], pts[:, 1])])
    elif len(u.shape) == 4:
        # RectBivariateSpline can only deal with 2d interpolation.
        # Use a different interpolator for each time point,
        # then interpolate linearly in time between those functions.
        # using convenience.interpolate_function_list does not work with
        # joblib, so I need to copy the code here.
        interp_x = [RectBivariateSpline(y, x, vf[:, :, 0], **interp_kwargs)
                    for vf in u]
        interp_y = [RectBivariateSpline(y, x, vf[:, :, 1], **interp_kwargs)
                    for vf in u]
        max_i = len(t)-1

        def rhs(tpt, pts):
            pts = np.fmod(pts.T, period)
            i = np.searchsorted(t, tpt, side='right')
            i = i if i < max_i else max_i
            delta = t[i] - t[i-1]
            val_a = np.stack([interp_y[i-1](pts[:, 0], pts[:, 1]),
                              interp_x[i-1](pts[:, 0], pts[:, 1])])
            val_b = np.stack([interp_y[i](pts[:, 0], pts[:, 1]),
                              interp_x[i](pts[:, 0], pts[:, 1])])
            return ((t[i]-tpt)*val_a + (tpt - t[i-1])*val_b)/delta

    # iterate over initial conditions.
    phi = []

    def loop_op(x_0, y_0):
        out = solve_ivp(rhs, t_span, np.array([y_0, x_0]), **solver_kwargs)
        assert out['status'] != -1, "integration failed"
        if dense_output:
            return out['sol']
        return out['y']
    for y_0 in initial_cond_y:
        if n_jobs > 1:
            phi.append(Parallel(n_jobs=n_jobs)(delayed(loop_op)(x_0, y_0)
                                               for x_0 in initial_cond_x))
        elif n_jobs == 1:
            phi.append([loop_op(x_0, y_0) for x_0 in initial_cond_x])
    phi = np.stack(phi).transpose(3, 0, 1, 2)
    phi = np.roll(phi, 1, axis=-1) + shift
    # ensures that x-coord is 1st & undo shift
    if any(pbc):
        period = np.roll(period, 1)  # to compensate roll of phi

        def fold_pbc(flow):
            return np.fmod(np.around(flow, decimals=5)-shift, period)+shift
        return phi, fold_pbc
    return phi


def pull_back(phi, t_eval, field, x, y, t_field=None, fold_pbc=None,
              covariant=False, density=False, interp_kwargs=None):
    """
    Pull back 2d scalar/vector/tensor field along flow lines phi.

    The field must be defined on a rectangular grid which contains the flow
    lines collected in phi. Assumes that phi[0], the array of flow line initial
    conditions forms a rectangular grid, which covers the region of interest
    into which one wishes to back-transport.

    The tensor field can be both co- and contravariant, see
    https://en.wikipedia.org/wiki/Tensor_calculus.

    Parameters
    ----------
    phi : np.array of shape (#timepoints, n_rows_phi, n_cols_phi, 2)
        Flow of a vector field, as returned by rect_flow(). phi[0] must form
        a rectangular grid. If the underlying domain is periodic, supply phi
        as returned by rect_flow() (i.e. the flow lifted to covering space),
        and not the flow lines mapped back into the periodic domain. This is
        important for the numerical calculation of jacobians.
    t_eval : np.array of shape (#timepoints,)
        Time points at which flow lines phi were evaluated.
    field : np.array of shape (n_rows, n_cols, ...)
        Values of scalar, vector or tensor field on rectangular grid.
    x : np.array of shape (n_cols,)
        Array of x-coordinate values, with x[i] == x-coordinate of field[:,i].
        Must be strictly increasing.
    y : np.array of shape (n_rows,)
        Array of y-coordinate values, with y[i] == y-coordinate of field[i,:].
        Must be strictly increasing.
    t_field : float, optional
        Time at which field values were evaluated. t_field must be in t_eval.
        Defaults to t_eval[-1] (last point on flow lines).
    fold_pbc : callable, optional
        If the flow is on a periodic domain, provide function which maps flow
        lines back into periodic domain, as returned by rect_flow. The default
        is None.
    covariant : bool or list of bools, optional
        Which tensor indices are covariant. True: index is covariant,
        False: index is contravariant. If a single True/False is supplied,
        then all/no indices are covariant.
    density : bool, optional
        Whether field transforms as a tensor density
    interp_kwargs : dict, optional
        Dictionary of keyword arguments for RectBivariatSpline interpolator
        used to interpolate field.

    Returns
    -------
    field_back : np.array of shape (n_rows, n_cols, ...)
        Field pulled back along flowlines, at x/y-coordinates phi[0,:,:,0/1].
    """
    # preliminaries
    interp_kwargs = {} if interp_kwargs is None else interp_kwargs
    t_ind = np.nonzero(t_eval == t_field)[0][0] if t_field is not None else -1
    n_comp = np.prod(field.shape[2:]) if len(field.shape) > 2 else 1
    if not isinstance(covariant, list):
        covariant = (covariant,)*(len(field.shape)-2)
    # calculate the jacobian and map flow lines back into periodic domain
    jac = np.stack(np.gradient(phi[t_ind], phi[0, 0, :, 0], phi[0, :, 0, 1],
                               axis=(1, 0)), axis=-2)
    jac_inv = np.linalg.pinv(jac)
    phi = phi if fold_pbc is None else fold_pbc(phi)

    in_bounds = np.all((phi[t_ind, :, :, 0] >= np.min(x))
                       & (phi[t_ind, :, :, 0] <= np.max(x))
                       & (phi[t_ind, :, :, 1] >= np.min(y))
                       & (phi[t_ind, :, :, 1] <= np.max(y)))
    assert in_bounds, "Stream lines out of field domain"

    # interpolate field components onto stream line end points - need to reshp
    field_comps = field.reshape(field.shape[:2]+(n_comp,)
                                ).transpose((-1,)+(0, 1))
    comp_interp = np.array([RectBivariateSpline(y, x, component,
                                                **interp_kwargs)
                            for component in field_comps])

    endpoint_vals = np.array([f(phi[t_ind, :, :, 1], phi[t_ind, :, :, 0],
                                grid=False) for f in comp_interp])
    if len(field.shape) > 2:
        endpoint_vals = endpoint_vals.transpose(1, 2, 0).reshape(
            phi.shape[1:3]+field.shape[2:])
    else:  # special case: scalar function
        endpoint_vals = endpoint_vals[0]

    # contract tensor field indices with jacobian
    if density:
        endpoint_vals = (endpoint_vals.T * np.linalg.det(jac).T).T
    tensor_ind = string.ascii_lowercase[10:10+len(field.shape)-2]
    base_str = 'abij,ab'+tensor_ind
    for index, is_cov in enumerate(covariant):
        if is_cov:
            contr_str = base_str.replace(tensor_ind[index], 'j')
            target_str = '->ab'+tensor_ind.replace(tensor_ind[index], 'i')
            endpoint_vals = np.einsum(contr_str+target_str, jac,
                                      endpoint_vals)
        else:
            contr_str = base_str.replace(tensor_ind[index], 'i')
            target_str = '->ab'+tensor_ind.replace(tensor_ind[index], 'j')
            endpoint_vals = np.einsum(contr_str+target_str, jac_inv,
                                      endpoint_vals)

    return endpoint_vals


def advect_points(u, x, y, t, t_span, t_eval=None,
                  initial_conds=None,
                  interp_kwargs=None, solver_kwargs=None):
    """
    Advect points (initial conditions) by flow field.

    A simplified version of rect_flow. Can't deal with periodic
    bc's. Seeds streamlines at arbitrary initial conditions
    instead of grid, thus not suitable to pulling back
    fields. To be used to advect points.
    
    Parameters
    ----------
    u : ndarray of shape (n_timepoints, n_rows, n_cols, 2)
        Vector field values at grid points. u[:,:,:,0] is the x-component,
        u[:,:,:,1] the y-component. If u is a 3d array (len(u.shape) == 3),
        the vector field is assumed to be constant in time.
    x : ndarray of shape (n_cols,)
        Array of x-coordinate values. x[i] == x-coord for u[:,:,i].
        Coordinates must be strictly ascending.
    y : ndarray (n_rows,)
        Array of y-coordinate values. y[i] == y-coord for u[:,i,:].
        Coordinates must be strictly ascending.
    t : ndarry of shape (n_timepoints,)
        Array of times at which vector field is sampled, in strictly
        ascending order.  If the vector field is time independent
        (len(u.shape) == 3), this argument is ignored (just pass None).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf).
    t_eval : np.array, otional.
        Times at which to evaluate the flow. Defaults to
        np.arange(t_span[0], t_span[1]+1), so phi[0] == initial conditions.
    initial_conds : ndarrray of dimension (N, 2)
        Array of initial conditions
    interp_kwargs  : dict, optional
        Parameters for the scipy.interpolate.RectBivariateSpline interpolator,
        see scipy docs. In particular, one can choose the order
        (linear, cubic, ...) and the smoothing paramter.
    solver_kwargs : dict, optional
        Parameters for the scipy ODE solver solve_ivp, see scipy docs.
        The default is {'method'='LSODA', 'rtol'=1e-05}.

    Returns
    -------
    phi : ndarray of shape (#timepoints, n_rows, n_cols, 2)
        Flow of vector field. phi[:,i,j,:] is the trajectory of
        point y[i], y[j], with phi[:,i,j,0] being the x-coordinate
        and phi[:,i,j,1] the y-coordinate. If dense_output is True, return
        array (n_rows, n_cols, 2) of OdeSolution objects.

    """
    ## preliminary argument parsing
    # ensure correct dtype - if coords are ints, errors can occur
    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)
    initial_conds = initial_conds.astype(float, copy=False)
    if t_eval is None:
        t_eval = np.arange(t_span[0], t_span[1]+1)
    if interp_kwargs is None:
        interp_kwargs = {}
    if solver_kwargs is None:
        solver_kwargs = {'method': 'RK45', 'rtol': 1e-05}
    solver_kwargs['t_eval'] = t_eval
    solver_kwargs['vectorized'] = True

    ## set up interpolator
    if len(u.shape) == 3:  # time-independent vector field
        interp_x = RectBivariateSpline(y, x, u[:, :, 0], **interp_kwargs)
        interp_y = RectBivariateSpline(y, x, u[:, :, 1], **interp_kwargs)

        def rhs(tpt, pts):
            return np.stack([interp_y(pts[0], pts[1]),
                             interp_x(pts[0], pts[1])])
    elif len(u.shape) == 4:
        # RectBivariateSpline can only deal with 2d interpolation.
        # Use a different interpolator for each time point,
        # then interpolate linearly in time between those functions.
        interp_x = [RectBivariateSpline(y, x, vf[:, :, 0], **interp_kwargs)
                    for vf in u]
        interp_y = [RectBivariateSpline(y, x, vf[:, :, 1], **interp_kwargs)
                    for vf in u]
        max_i = len(t)-1

        def rhs(tpt, pts):
            i = np.searchsorted(t, tpt, side='right')
            i = i if i < max_i else max_i
            delta = t[i] - t[i-1]
            val_a = np.stack([interp_y[i-1](pts[0], pts[1]),
                              interp_x[i-1](pts[0], pts[1])])
            val_b = np.stack([interp_y[i](pts[0], pts[1]),
                              interp_x[i](pts[0], pts[1])])
            return ((t[i]-tpt)*val_a + (tpt - t[i-1])*val_b)/delta

    ## iterate over initial conditions.
    phi = []
    for x_0 in initial_conds:
        out = solve_ivp(rhs, t_span, x_0, **solver_kwargs)
        assert out['status'] != -1, "integration failed"
        phi.append(out['y'])
        
    return np.stack(phi).transpose(2,0,1)
