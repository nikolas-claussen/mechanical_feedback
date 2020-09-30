# -*- coding: utf-8 -*-

"""
Created on Tue Sep 22 10:29:09 2020

@author: nikolas

Tests for the module get_ode_flow.py. The rect_flow function is tested,
later functions test the rect_flow function in combination with the pull_back()
function. The tests can serve as examples on how to use both functions.

"""

import numpy as np
from scipy import ndimage
import ode_flow as flow


def test_autonomous_runs_correctly():
    # trivial example. 0 vector field should give identity flow
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 5, 10)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((len(y), len(x), 2))

    Phi = flow.rect_flow(u, x, y, None, (0, 4))

    assert np.allclose(Phi[-1, :, :, 0], X) and np.allclose(Phi[-1, :, :, 1], Y)


def test_autonomous_x_const():
    # constant flow in x direction
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 5, 10)
    X, Y = np.meshgrid(x, y)
    u = np.stack([np.ones((len(y), len(x))),
                  np.zeros((len(y), len(x)))], axis=-1)
    u[:, -2:] = 0  # brd to zero to avoid errors

    Phi = flow.rect_flow(u, x, y, None, (0, 3))

    assert (np.allclose(Phi[-1, :, :, 1], Y)
            and np.allclose(Phi[-1, :, :10, 0], X[:, :10]+3, atol=1e-2))


def test_flow_autonomous_x_accel():
    # accelerating flow in x direction
    x = np.linspace(0, 10, 21)
    y = np.linspace(0, 5, 11)
    X, Y = np.meshgrid(x, y)
    u = np.stack([X/3, np.zeros((len(y), len(x)))], axis=-1)
    u[:, -2:] = 0  # brd to zero to avoid errors

    Phi = flow.rect_flow(u, x, y, None, (0, 3))

    assert (np.allclose(Phi[-1, :, :, 1], Y)
            and np.allclose(Phi[-1, :, :5, 0], X[:, :5]*np.exp(3/3),
                            atol=1e-2))


def test_flow_rotation_vectorfield():
    # rotation
    x = np.linspace(-2, 2, 41)
    y = np.linspace(-2, 2, 41)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2+Y**2)
    mask = (r <= 2).astype(int)
    mask2 = (r <= 1.7).astype(int)
    angle = np.arctan2(Y, X)
    u = np.stack([mask*r*np.sin(angle), -mask*r*np.cos(angle)], axis=-1)

    Phi = flow.rect_flow(u, x, y, None, (0, np.pi),
                         t_eval=np.linspace(0, np.pi, 11))

    assert (np.allclose(mask2*Phi[-1, :, :, 0], -mask2*X, atol=1e-2)
            and np.allclose(mask2*Phi[-1, :, :, 1], -mask2*Y, atol=1e-2))


def test_flow_timedepent_const_vectorfield():
    # check whether a time-dependent, but constant in time vf
    # gives the same result as the time-independent calculation
    x = np.linspace(0, 10, 40)
    y = np.linspace(0, 5, 20)
    t = np.arange(10)
    X, Y = np.meshgrid(x, y)
    u1 = np.stack([np.ones((len(y), len(x))),
                  np.zeros((len(y), len(x)))], axis=-1)
    u1[:, -2:] = 0  # brd to zero to avoid errors

    Phi1 = flow.rect_flow(u1, x, y, None, (0, 9),
                          initial_cond_x=x,
                          initial_cond_y=y,
                          solver_kwargs={'method': 'RK45',
                                         'rtol': 1e-5,
                                         'max_step': 1})

    u2 = np.stack([u1]*len(t))
    Phi2 = flow.rect_flow(u2, x, y, t, (0, 9),
                          initial_cond_x=x,
                          initial_cond_y=y,
                          solver_kwargs={'method': 'RK45',
                                         'rtol': 1e-5,
                                         'max_step': 1})

    assert np.allclose(Phi1, Phi2, atol=1e-2)


def test_flow_pbc_cylinder():
    # check pbc for simple translation in periodic direction
    x = np.linspace(0, 10, 41)
    y = np.linspace(0, 5, 21)
    X, Y = np.meshgrid(x, y)
    u = np.stack([np.ones((len(y), len(x))),
                  np.zeros((len(y), len(x)))], axis=-1)

    Phi, fold_pbc = flow.rect_flow(u, x, y, None, (0, 10), pbc=(True, False),
                                   interp_kwargs={'kx': 1, 'ky':1})
    Phi = fold_pbc(Phi)
    # with Spline interpolation, there can be some small issues

    assert all([np.allclose(Phi[t, :, :, 0], (X+t) % 10, atol=1e-3)
                for t in range(10)])


def test_flow_timedependent_space_independent_vectorfield():
    # test a time-dependent, but spatially constant vector field.
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    t = np.linspace(0, 4*np.pi, 41)
    X, Y = np.meshgrid(x[:7], y[:7])

    u = np.stack([np.ones((len(y), len(x))),
                  np.zeros((len(y), len(x)))], axis=-1)
    u = (np.stack([u]*len(t), axis=-1) * np.sin(t) * np.cos(t))
    u = np.moveaxis(u, -1, 0)
    res = X + (np.sin(t)**2/2)[:, None, None]

    Phi = flow.rect_flow(u, x, y, t, (t[0], t[-1]), t_eval=t,
                         initial_cond_x=x[:7], initial_cond_y=y[:7])
    err = np.abs(Phi[:, 1:, 1:, 0] - res[:, 1:, 1:])/res[:, 1:, 1:]

    assert np.quantile(err, 0.9) < 0.01


def test_flow_time_and_space_dependent_vectorfield():
    # example of time and space dependent vector field
    # consider accelerating rotation.
    x = np.linspace(-2, 2, 41)
    y = np.linspace(-2, 2, 41)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2+Y**2)
    mask = (r <= 2).astype(int)
    mask2 = (r <= 1.7).astype(int)
    angle = np.arctan2(Y, X)
    u = (np.pi/2) * np.stack([mask*r*np.sin(angle), -mask*r*np.cos(angle)],
                             axis=-1)
    t = np.linspace(0, 2*np.pi, 21)
    u = np.tensordot(np.sin(t), u, axes=0)

    Phi, _ = flow.rect_flow(u, x, y, t, (0, 2*np.pi), t_eval=t,
                            pbc=(True, True))
    rot_angle = (-np.pi/2 * np.cos(t) + np.pi/2 * np.cos(t[0])) * 360/(2*np.pi)
    errs = [np.abs(ndimage.rotate(X, -rot_angle[i], reshape=False)
                   - Phi[i, :, :, 0]) * mask2 for i in range(11)]
    assert all([np.quantile(x, 0.95) < 0.05 for x in errs])


def test_jacobian():
    # simple test calculation of jacobian, same code as in ode_flow
    x = np.linspace(-2, 2, 41)
    y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(x, y)

    jac_id = np.stack(np.gradient(np.stack([X, Y], axis=-1), x, y,
                                  axis=(1, 0)), axis=-2)

    assert np.allclose(jac_id,
                       np.tensordot(np.ones(X.shape), np.eye(2), axes=0))


def test_pull_back_trivial():
    # simple translation, but with vectors
    x = np.linspace(-2, 2, 41)
    y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(x, y)
    t_eval = np.arange(10)
    phi = np.stack([np.stack([X+step, Y], axis=-1) for step in t_eval])
    f = np.stack([X**2, np.zeros(X.shape)], axis=-1)
    f_pullback = flow.pull_back(phi, t_eval, f, x+9, 2*y)
    # y+1/2 raises assertion error as it should
    assert np.allclose(f, f_pullback)


def test_pull_back_radial_function():
    # radial flow with ring-like function, check also density
    x = np.linspace(-2, 2, 101)
    y = np.linspace(-2, 2, 101)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2+Y**2)
    def ring(r1, r2): return ((r >= r1) & (r <= r2)).astype(float)
    radial_vf = -np.stack([X, Y], axis=-1)
    t_eval = np.linspace(0, 1, 51)
    phi = flow.rect_flow(radial_vf, x, y, None, (0, 1), t_eval=t_eval)
    f = ring(0.5, 1)
    f_pullbacks = [flow.pull_back(phi, t_eval, f, x, y, t_field=t,
                                  density=True) for t in t_eval]
    sol = [ring(np.exp(t)*0.5, np.exp(t)*1)*np.exp(-2*t) for t in t_eval]
    # ring moves and is dilutes
    err = [np.mean(np.abs(a-b))/np.mean(b) for a, b, in zip(f_pullbacks, sol)]
    # error is due to numerical issues at ring boundaries.
    assert np.all(np.array(err) < 0.1)


def test_pull_back_vector_field_rotation():
    # pull back vector field under rotation
    x = np.linspace(-3, 3, 101)
    y = np.linspace(-3, 3, 101)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2+Y**2)
    mask = (r <= 2.75).astype(int)
    mask2 = (r <= 2.5).astype(int)
    angle = np.arctan2(Y, X)
    u = np.stack([mask*r*np.sin(angle), -mask*r*np.cos(angle)], axis=-1)
    # points in clockwise direction
    t_eval = np.linspace(0, np.pi/2, 21)
    phi = flow.rect_flow(u, x, y, None, (t_eval[0], t_eval[-1]), t_eval=t_eval)

    vf = np.stack([np.ones(X.shape), np.zeros(Y.shape)], axis=-1)
    vf_pullback = flow.pull_back(phi, t_eval, vf, x, y, t_field=t_eval[-1])
    sol = np.stack([np.zeros(X.shape), np.ones(Y.shape)], axis=-1)
    err = (np.linalg.norm(vf_pullback-sol, axis=-1) * mask2)
    # one form pullback should give the same for a rotation
    form_pullback = flow.pull_back(phi, t_eval, vf, x, y, t_field=t_eval[-1],
                                   covariant=True)
    err_form = (np.linalg.norm(form_pullback-sol, axis=-1) * mask2)

    assert (np.quantile(err, 0.95) < 0.005) and (np.quantile(err_form, 0.95) < 0.005)


def test_pull_back_covector():
    # pull back covector field under hyperbolic flow
    x = np.linspace(-3, 3, 151)
    y = np.linspace(-3, 3, 151)
    X, Y = np.meshgrid(x, y)
    u = np.stack([X, -Y], axis=-1)
    one_form = np.stack([np.ones(X.shape), np.ones(Y.shape)], axis=-1)
    # keep it simple.
    t_eval = np.linspace(0, 1, 21)
    phi = flow.rect_flow(u, x, y, None, (t_eval[0], t_eval[-1]),
                         initial_cond_x=x[50:-50], initial_cond_y=y[50:-50],
                         t_eval=t_eval)
    vf_pullback = flow.pull_back(phi, t_eval, one_form, x, y,
                                 t_field=t_eval[-1], covariant=False)
    one_form_pullback = flow.pull_back(phi, t_eval, one_form, x, y,
                                       t_field=t_eval[-1], covariant=True)
    # should be a pure shear transformation.

    assert (np.all(np.abs(one_form_pullback[:, :, 1]-np.exp(-1)) <= 0.005)
            and np.all(np.abs(vf_pullback[:, :, 1]-np.exp(1)) <= 0.005))


def test_pull_back_vector_pbc():
    # pull back vector field with periodic boundary conditions
    x = np.linspace(0, 2*np.pi, 151)
    y = np.linspace(0, 2*np.pi, 151)
    X, Y = np.meshgrid(x, y)
    # take a torus and simple flow wrapping around twice
    u = np.stack([np.ones(X.shape), np.ones(Y.shape)], axis=-1)
    t_eval = np.linspace(0, 2*np.pi, 21)
    phi, fold_pbc = flow.rect_flow(u, x, y, None, (t_eval[0], t_eval[-1]),
                                   pbc=(True, True),
                                   t_eval=t_eval)
    vf = np.stack([np.sin(X), np.zeros(Y.shape)], axis=-1)
    vf_pullback_2pi = flow.pull_back(phi, t_eval, vf, x, y, fold_pbc=fold_pbc,
                                     t_field=t_eval[20])
    vf_pullback_pi = flow.pull_back(phi, t_eval, vf, x, y, fold_pbc=fold_pbc,
                                    t_field=t_eval[10])
    err_2pi = np.linalg.norm(vf_pullback_2pi-vf, axis=-1).flatten()
    err_pi = np.linalg.norm(vf_pullback_pi+vf, axis=-1).flatten()

    assert np.allclose(err_2pi, 0, atol=1e-4) and np.allclose(err_pi, 0, atol=1e-4)


def test_pull_back_matrix():
    # pulling back (2, 0) tensor
    # let's take the example of a disclination  of type -1/2 and a rotation
    # vector field
    x = np.linspace(-3, 3, 101)
    y = np.linspace(-3, 3, 101)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2+Y**2)
    mask = (r <= 2.75).astype(int)
    mask2 = (r <= 2.5).astype(int)
    u = np.stack([mask*Y, -mask*X], axis=-1)
    t_eval = np.linspace(0, 2*np.pi/3, 21)
    phi = flow.rect_flow(u, x, y, None, (t_eval[0], t_eval[-1]), t_eval=t_eval)
    m = np.stack([[X, -Y], [-Y, -X]]).transpose(2, 3, 0, 1)
    m_pullback = flow.pull_back(phi, t_eval, m, x, y, t_field=t_eval[-1])
    err = np.linalg.norm(m-m_pullback, axis=(2,3))*mask2

    assert np.quantile(err, 0.9) < 0.01
