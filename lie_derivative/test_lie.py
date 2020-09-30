#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:05:57 2020

@author: nikolas

Unit tests for lie_derivative.py.

Most vector/tensor fields will be randomly generated, but I will also include
some cases where the Lie derivative can be calculated analytically.

Careful: the partial derivatives have to conform to the coordinate convetions
of lie_derivative.py. row = y-axis, in inverted order, column = x-axis,
regular order.

TO DO: - add non-equally spaced grid test, maybe just simple function.
       - add tri mesh tests, on sphere and rectangular mesh
"""

import numpy as np
import lie
from scipy.ndimage import gaussian_filter1d
import os
import igl

rng = np.random.default_rng()

# Helper functions


def get_random_tensor_rect(t_steps=1, n_x=100, n_y=200, t_rank=0, scale=40,
                           smooth=20, smooth_t=5):
    """
    Generate a random, smooth tensor field on a 2d rect mesh for test purposes.
    The shapes are consistent with the x-y to row-column convention
    in lie_derivative.py

    Parameters
    ----------
    t_steps : int, optional
        Number of time steps. The default is 1.
    n_x : int, optional
        DESCRIPTION. The default is 200.
    n_y : int, optional
        DESCRIPTION. The default is 100.
    t_rank : TYPE, optional
        DESCRIPTION. The default is 0.
    scale : float, optional
        Std deviation of Gaussian random number generator. The default is 10.
    smooth : float, optional
        Spatial smoothing parameter (gaussian filter). The default is 10.
    smooth_t : float, optional
        Temporal smoothing parameter (gaussian filter). The default is 10

    Returns
    -------
    None.

    """
    rand_t = rng.normal(scale=scale, size=(t_steps, n_y, n_x,)+(2,)*t_rank)
    rand_t = gaussian_filter1d(gaussian_filter1d(rand_t, smooth, axis=1),
                               smooth, axis=2)
    rand_t = gaussian_filter1d(rand_t, smooth_t, axis=0)
    return rand_t


def rel_abs_err(a, b):
    """
    Returns the relative absolute error between two arrays a,b.
    a,b need to have the same shape.

    Parameters
    ----------
    a : np.array
    b : np.array

    Returns
    -------
    err : np.array
        abs(a-b)/abs(b), flattened

    """
    return (np.abs(a-b)/np.abs(b)).flatten()


rtol = 1e-3

# mathematical property checks


def test_constant_u_function():
    # For a constant tensor & vector field, the lie derivative is 0
    u = 10 * np.ones((10, 200, 100, 2))
    t = 15 * np.ones((10, 200, 100,))

    lie_t = lie.lie_derivative(u, t)

    assert np.allclose(lie_t, np.zeros(t.shape), rtol=rtol)


def test_constant_u_tensor():
    # For a constant tensor & vector field, the lie derivative is 0
    u = 10 * np.ones((10, 200, 100, 2))
    t = 15 * np.ones((10, 200, 100, 2, 2, 2))

    lie_t = lie.lie_derivative(u, t)

    assert np.allclose(lie_t, np.zeros(t.shape), rtol=rtol)


def test_u_0_time_derivative():  # why does this take so long ???
    # For a vectorfield that is 0, lie derivative is just the time derivative
    u = np.zeros((10, 200, 100, 2))
    t = get_random_tensor_rect(t_steps=10, t_rank=5)

    lie_t = lie.lie_derivative(u, t)

    # partial time derivative
    dt_t = np.gradient(t, axis=0)

    assert np.allclose(lie_t, dt_t, rtol=rtol)


def test_directional_derivative():
    # for a time-independent function, the lie derivative is the usual
    # directional derivative

    u = get_random_tensor_rect(t_steps=1, t_rank=1)
    t = get_random_tensor_rect(t_steps=1, t_rank=0)

    lie_t = lie.lie_derivative(u, t)

    # directional derivative - careful to use consistent coord convention
    grad_t = np.stack([np.gradient(t, axis=2), -np.gradient(t, axis=1)])

    u_dot_grad_t = np.einsum('tyxi,ityx->tyx', u, grad_t)

    assert np.allclose(lie_t, u_dot_grad_t, rtol=rtol)


def test_density_divergence():
    # for a constant density, the lie derivative is div u
    u = get_random_tensor_rect(t_steps=1, t_rank=1)
    t = np.ones((1, 200, 100,))

    lie_t = lie.lie_derivative(u, t, density=True)

    du = np.stack([np.gradient(u, axis=2), -np.gradient(u, axis=1)])

    div_u = np.einsum('ityxi->tyx', du)

    assert np.allclose(lie_t, div_u, rtol=rtol)


def test_linearity_t():
    # the lie derivative lie_u is a linear oerator
    # try covariant tensors
    u = get_random_tensor_rect(t_steps=10, t_rank=1)
    s = get_random_tensor_rect(t_steps=10, t_rank=2)
    t = get_random_tensor_rect(t_steps=10, t_rank=2)

    lie_ts = lie.lie_derivative(u, t+s, covariant=True)

    lie_t = lie.lie_derivative(u, t, covariant=True)
    lie_s = lie.lie_derivative(u, s, covariant=True)

    assert np.allclose(lie_t+lie_s, lie_ts, rtol=1e-3)


def test_anti_symmetry():
    # on vector fields, the Lie derivative is antisymmetric
    u = get_random_tensor_rect(t_steps=1, t_rank=1)
    v = get_random_tensor_rect(t_steps=1, t_rank=1)

    lie_u_v = lie.lie_derivative(u, v, covariant=False)
    lie_v_u = lie.lie_derivative(v, u, covariant=False)

    assert np.allclose(lie_u_v, -lie_v_u, rtol=rtol)


def test_product_rule_functions():
    # the Lie derivative obeys the product rule (using for functions)
    u = get_random_tensor_rect(t_steps=1, t_rank=1)
    f = get_random_tensor_rect(t_steps=1, t_rank=0)
    g = get_random_tensor_rect(t_steps=1, t_rank=0)

    lie_fg = lie.lie_derivative(u, f*g)  # pointwise product = TP for fcts
    lie_f = lie.lie_derivative(u, f)
    lie_g = lie.lie_derivative(u, g)
    prod = f * lie_g + lie_f * g

    err = rel_abs_err(lie_fg, prod)

    # 90% relative error is at 2 %, but 47/20000 entries are truly different
    # perfect fullfillement, i.e.  np.allclose(lie_fg, prod, rtol=rtol)
    # is not realistic, numerically
    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_product_rule():
    # the Lie derivative obeys the product rule (using tensor products)
    u = get_random_tensor_rect(t_steps=1, t_rank=1)
    s = get_random_tensor_rect(t_steps=1, t_rank=1)
    t = get_random_tensor_rect(t_steps=1, t_rank=1)

    lie_st = lie.lie_derivative(u, np.einsum('txyi,txyj->txyij', s, t))
    lie_t = lie.lie_derivative(u, t)
    lie_s = lie.lie_derivative(u, s)
    prod = (np.einsum('txyi,txyj->txyij', s, lie_t)
            + np.einsum('txyi,txyj->txyij', lie_s, t))

    err = rel_abs_err(lie_st, prod)

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_lie_bracket_time_independent():
    # for vector fields, the lie derivative is the lie bracket
    u = get_random_tensor_rect(t_steps=1, t_rank=1)
    v = get_random_tensor_rect(t_steps=1, t_rank=1)
    lie_v = lie.lie_derivative(u, v, covariant=False)

    du = np.stack([np.gradient(u, axis=2), -np.gradient(u, axis=1)])
    dv = np.stack([np.gradient(v, axis=2), -np.gradient(v, axis=1)])
    bracket = (np.einsum('txyi,itxyj->txyj', u, dv)
               - np.einsum('txyi,itxyj->txyj', v, du))

    assert np.allclose(bracket, lie_v, rtol=rtol)


def test_one_form():
    # check formula for one forms
    u = get_random_tensor_rect(t_steps=10, t_rank=1)
    alpha = get_random_tensor_rect(t_steps=10, t_rank=1)

    lie_alpha = lie.lie_derivative(u, alpha, covariant=True)

    dt_alpha = np.gradient(alpha, axis=0)
    du = np.stack([np.gradient(u, axis=2), -np.gradient(u, axis=1)])
    dalpha = np.stack([np.gradient(alpha, axis=2),
                       -np.gradient(alpha, axis=1)])

    lie_manual = (dt_alpha
                  + np.einsum('txyi,itxyj->txyj', u, dalpha)
                  + np.einsum('txyi,jtxyi->txyj', alpha, du))

    assert np.allclose(lie_alpha, lie_manual, rtol=rtol)


def test_zero_two_tensor():
    # check the formula for (0,2)-tensors
    u = get_random_tensor_rect(t_steps=1, t_rank=1)
    t = get_random_tensor_rect(t_steps=1, t_rank=2)

    lie_t = lie.lie_derivative(u, t, covariant=True)

    du = np.stack([np.gradient(u, axis=2), -np.gradient(u, axis=1)])
    dt = np.stack([np.gradient(t, axis=2), -np.gradient(t, axis=1)])

    lie_manual = (np.einsum('txyi,itxyjk->txyjk', u, dt)
                  + np.einsum('jtxyi,txyik->txyjk', du, t)
                  + np.einsum('ktxyi,txyji->txyjk', du, t))

    assert np.allclose(lie_t, lie_manual, rtol=rtol)


# analytic checks :

def test_analytic_directional_derivative_0():
    # very simple example: a linear function, constant vector field
    X, Y = np.meshgrid(np.arange(100), np.arange(200)[::-1])
    t = X[np.newaxis, :, :]
    u = np.stack([np.ones(X.shape),
                  np.zeros(Y.shape)], axis=-1)[np.newaxis, :, :, :]

    lie_t = lie.lie_derivative(u, t)

    assert np.allclose(lie_t[0, :, :], np.ones(X.shape), rtol=rtol)


def test_analytic_directional_derivative_1():
    # another simple example: a linear function, radial vector field
    # start at 1 to avoid division by 0 in relative error calc
    X, Y = np.meshgrid(np.arange(1, 101), np.arange(1, 201)[::-1])
    t = X[np.newaxis, :, :]
    u = np.stack([X, Y], axis=-1)[np.newaxis, :, :, :]

    lie_t = lie.lie_derivative(u, t)

    err = rel_abs_err(lie_t, X)

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_wikipedia_example_1():
    # 1st example of Lie derivative from wikipedia
    X, Y = np.meshgrid(np.arange(1, 101), np.arange(1, 201)[::-1])
    # stack *10 creates a constant in time field
    t = np.stack([X**2 + np.sin(Y)]*10)
    u = np.stack([np.stack([-Y**2, np.sin(X)], axis=-1)]*10)

    lie_t = lie.lie_derivative(u, t)

    analytic_res = np.stack([-np.sin(X)*np.cos(Y)-2*X*Y**2]*10)

    err = rel_abs_err(lie_t, analytic_res)

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_wikipedia_example_2():
    # wikipedia example no. 2. In 3d!
    X, Y, Z = np.meshgrid(np.arange(1, 101),
                          np.arange(1, 201)[::-1],
                          np.arange(1, 3))

    u = np.stack([np.stack([-Y**2, np.sin(X), np.zeros(X.shape)], axis=-1)])
    # x**2 + Y**2 dx wedge dz
    t = np.zeros((1,)+X.shape+(3, 3,))
    t[0, :, :, :, 0, 2] = X**2+Y**2
    t[0, :, :, :, 2, 0] = -t[0, :, :, :, 0, 2]

    lie_t = lie.lie_derivative(u, t, covariant=True)

    analytic_res = np.zeros(t.shape)
    analytic_res[0, :, :, :, 0, 2] = -2*X*Y**2 + 2*Y*np.sin(X)
    analytic_res[0, :, :, :, 2, 0] = -analytic_res[0, :, :, :, 0, 2]
    analytic_res[0, :, :, :, 1, 2] = -2*Y*X**2 - 2*Y**3
    analytic_res[0, :, :, :, 2, 1] = -analytic_res[0, :, :, :, 1, 2]

    err = np.abs(lie_t - analytic_res)/np.mean(np.abs(analytic_res))
    # relative error does not work, there are 0's.

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_analytic_lie_bracket():
    # simple example of Lie bracket: X, Y vector fields commute
    X, Y = np.meshgrid(np.arange(100), np.arange(200)[::-1])
    u = np.stack([X, np.zeros(Y.shape)], axis=-1)[np.newaxis, :, :, :]
    v= np.stack([np.zeros(X.shape), Y], axis=-1)[np.newaxis, :, :, :]

    lie_u = lie.lie_derivative(u, v)

    assert np.allclose(lie_u, np.zeros(u.shape), rtol=rtol)


def test_analytic_2_0_tensor():
    # simple tensor t = x * (r x r), u = r * d_phi
    X, Y = np.meshgrid(np.arange(1, 101), np.arange(1, 201)[::-1])
    u = np.stack([Y, -X], axis=-1)[np.newaxis, :, :, :]
    t = np.stack([[[X * X**2, X * X*Y], [X * X*Y, X * Y**2]]]
                 ).transpose(0, 3, 4, 1, 2)
    lie_t = lie.lie_derivative(u, t, covariant=False)

    analytic_res = np.stack([[[Y*X**2, Y*X*Y], [Y*X*Y, Y*Y**2]]]
                            ).transpose(0, 3, 4, 1, 2)
    err = rel_abs_err(lie_t, analytic_res)

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


# load test meshes: a sphere, a cylinder, and a flat rect mesh
root_folder = os.getcwd()
v_sphere, f_sphere = igl.read_triangle_mesh(os.path.join(root_folder,
                                                         'sphere.ply'))
v_sphere, f_sphere = igl.loop(v_sphere, f_sphere, 3)
v_sphere = (v_sphere.T / np.linalg.norm(v_sphere, axis=1)).T
v_rect, f_rect = igl.read_triangle_mesh(os.path.join(root_folder, 'rect.ply'))
v_cyl, f_cyl = igl.cylinder(100, 500)


def test_tri_grad_const():
    # check that the tri_grad function returns 0 on constant functions
    # and shape is correct
    t = np.ones((10, v_sphere.shape[0], 3, 3))
    grad_t = lie.tri_grad(t, v_sphere, f_sphere)

    assert np.allclose(grad_t, np.zeros(grad_t.shape), atol=0.01)


def test_tri_grad_product_rule():
    # check tri_grad product rule
    f = v_sphere[np.newaxis, :, 0]
    g = v_sphere[np.newaxis, :, 1]

    grad_f = lie.tri_grad(f, v_sphere, f_sphere)
    grad_g = lie.tri_grad(g, v_sphere, f_sphere)
    grad_fg = lie.tri_grad(f*g, v_sphere, f_sphere)

    err = np.mean(np.linalg.norm(grad_fg-f*grad_g-g*grad_f, axis=0))
    err = err/np.mean(np.linalg.norm(grad_fg, axis=0))

    assert (np.median(err) < 0.05) & (np.quantile(err, 0.99) < 0.05)


def test_tri_grad_cyl():
    # simple test of gradient on cylinder
    grad_t = lie.tri_grad(v_cyl[np.newaxis, :, -1], v_cyl, f_cyl)

    assert np.allclose(grad_t[0, 0, :], np.zeros(v_cyl.shape[0]), atol=0.01)


def test_tri_grad_rect_mesh():
    # test whether tri grad coincides with np.grad on rect mesh
    X, _ = v_rect[:, 0].reshape(257, 218), v_rect[:, 1].reshape(257, 218)
    # X = X[:,::-1] # ensure correct convetions

    f_tri = np.sin(v_rect[:, 0]/10)[np.newaxis, :]
    f_grid = np.sin(X/10)[np.newaxis, :]

    grad_f_tri = lie.tri_grad(f_tri, v_rect, f_rect)
    grad_f_grid = -np.stack([np.gradient(f_grid, axis=2),
                             np.gradient(f_grid, axis=1)])
    # minus sign to match coordinate convetions

    err = grad_f_tri[:2, 0, :].reshape(2, 257, 218) - grad_f_grid[:, 0, :, :]
    err = err / np.mean(np.abs(grad_f_grid))

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_tri_grad_tangent():
    # check that the gradient is in the tangent plance
    grd_x = lie.tri_grad(v_sphere[np.newaxis, :, 0], v_sphere, f_sphere)
    err = np.abs(np.einsum('iv,vi->v', grd_x[:, 0, :], v_sphere))
    err /= np.mean(np.linalg.norm(grd_x, axis=0))

    print(np.median(err), np.quantile(err, 0.9))

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_lie_bracket_sphere_tangent():
    # lie derivative of vector field is still tangent
    phi = np.arctan2(v_sphere[:, 1], v_sphere[:, 0])
    theta = np.arccos(v_sphere[:, 2])

    d_phi = np.stack([-np.sin(phi),
                      np.cos(phi),
                      np.zeros(v_sphere.shape[0])], axis=-1)
    d_theta = np.stack([np.cos(phi) * np.cos(theta),
                        np.sin(phi) * np.cos(theta),
                        -np.sin(theta)], axis=-1)

    lie_t = lie.lie_derivative(d_phi[np.newaxis, :, :],
                               d_theta[np.newaxis, :, :],
                               v=v_sphere, f=f_sphere,
                               covariant=False)

    err = np.abs(np.einsum('vi,vi->v', lie_t[0, :, :], v_sphere))
    err /= np.mean(np.linalg.norm(lie_t, axis=2))

    print(np.median(err), np.quantile(err, 0.9))

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_lie_bracket_sphere_antisymmetry():
    phi = np.arctan2(v_sphere[:, 1], v_sphere[:, 0])
    theta = np.arccos(v_sphere[:, 2])

    d_phi = np.stack([-np.sin(phi),
                      np.cos(phi),
                      np.zeros(v_sphere.shape[0])], axis=-1)
    d_theta = np.stack([np.cos(phi) * np.cos(theta),
                        np.sin(phi) * np.cos(theta),
                        -np.sin(theta)], axis=-1)

    lie_theta = lie.lie_derivative(d_phi[np.newaxis, :, :],
                                   d_theta[np.newaxis, :, :],
                                   v=v_sphere, f=f_sphere,
                                   covariant=False)

    lie_phi = lie.lie_derivative(d_theta[np.newaxis, :, :],
                                 d_phi[np.newaxis, :, :],
                                 v=v_sphere, f=f_sphere,
                                 covariant=False)

    err = np.linalg.norm(lie_theta+lie_phi, axis=2)
    err /= np.mean(np.linalg.norm(lie_theta, axis=2))

    print(np.median(err), np.quantile(err, 0.9))

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


def test_lie_sphere_analytic_vector():
    # lie derivative on sphere. commutation relation of d_phi, d_theta
    # one should get [d_phi, d_theta] = z * d_theta

    z = v_sphere[:, 2]
    phi = np.arctan2(v_sphere[:, 1], v_sphere[:, 0])
    theta = np.arccos(z)

    d_phi = np.stack([-np.sin(phi),
                      np.cos(phi),
                      np.zeros(v_sphere.shape[0])], axis=-1)
    d_theta = np.stack([np.cos(phi) * np.cos(theta),
                        np.sin(phi) * np.cos(theta),
                        -np.sin(theta)], axis=-1)
    rho = np.sqrt(v_sphere[:, 0]**2+v_sphere[:, 1]**2)

    commutator = lie.lie_derivative(d_phi[np.newaxis, :, :],
                                    d_theta[np.newaxis, :, :],
                                    v=v_sphere, f=f_sphere,
                                    covariant=False)

    analytic_res = ((z/rho * d_phi.T).T)
    err = np.linalg.norm(analytic_res-commutator[0, :, :], axis=1)
    err /= np.mean(np.linalg.norm(analytic_res, axis=1))

    print(np.median(err), np.quantile(err, 0.9))

    assert (np.median(err) < 0.01) & (np.quantile(err, 0.9) < 0.05)


## Tests to be implemented

#def test_tri_reshaping():
    # check whether the reshaping works correctly

#    assert False


#def test_lie_rect_mesh_equals_lie_chart():
    # check that the rect mesh


#    assert False
#"""
