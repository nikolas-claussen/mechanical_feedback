#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep  15:10:36 2020

@author: nikolas, nclaussen@ucsb.edu

Tests for the module windowed_radon. 

"""

import numpy as np
import windowed_radon as wr

rng = np.random.default_rng()

def test_runs_errorfree():
    # check whether there are any runtime errors and shapes are ok
    im = np.ones((20,20))
    m = wr.windowed_radon(im, window_len=5, step=2)
    
    assert (len(m.shape) == 4) and (m.shape[2] == m.shape[3] == 2)
    
def test_linearity():
    # radon tf should be linear
    im1 = rng.normal(size=(20,20))
    im2 = rng.normal(size=(20,20))
    
    m1 = wr.windowed_radon(im1, window_len=5, step=2)
    m2 = wr.windowed_radon(im2, window_len=5, step=2)
    m12 = wr.windowed_radon(im1+im2, window_len=5, step=2)

    assert np.allclose(m1+m2, m12)
    
def test_symmetric():
    # resulting tensor should be symmetric
    im = rng.normal(size=(50,50))
    m = wr.windowed_radon(im, window_len=10, step=5)

    assert np.allclose(m, m.transpose(0,1,3,2))
    
def test_constant_im():
    # a constant image should correspond to the identity matrix
    im = np.ones((50,50))
    m = wr.windowed_radon(im, window_len=10, step=5)
    m = m / np.mean(m[:,:,0,0])
    
    m_id = np.repeat(np.repeat(np.eye(2)[np.newaxis,:,:],6,axis=0
                             )[np.newaxis,:,:,:],6, axis=0)
    
    assert np.allclose(m, m_id, atol=0.001)
    
    
def test_xy_stripes():
    # simple constant stripes in x or y direction
    im_x, im_y = (np.zeros((50,50)), np.zeros((50,50)))
    im_x[::5,:] = 1
    im_y[:,::5] = 1
    m_x = wr.windowed_radon(im_x, window_len=10, step=5, sigma=5)
    n_x = np.linalg.eigh(m_x)[1][:,:,1,:] # get highest eigen vector
    m_y = wr.windowed_radon(im_y, window_len=10, step=5, sigma=5)
    n_y = np.linalg.eigh(m_y)[1][:,:,1,:]

    assert (np.allclose(np.abs(n_x[:,:,0]), np.ones(m_x.shape[:2]), atol=0.01) 
            and np.allclose(np.abs(n_y[:,:,1]), np.ones(m_y.shape[:2]), atol=0.01))
    
def test_trace_intensity():
    # the trace should be proportional to the local intensity
    im, _ = np.meshgrid(np.arange(50), np.arange(50))
    m = wr.windowed_radon(im, window_len=5, step=1, sigma=0.5) 
    # low sigma so no smoothing
    tr = np.einsum('abii', m)
    
    assert np.allclose(tr/np.mean(tr), (im/np.mean(im))[5:-5,5:-5], atol=0.01)
    
    
def test_crossing_lines():
    # test a cross-shaped input
    im = np.zeros((50,50))
    im[23:27,:] = 1
    im[:,23:27] = 1
    
    m = wr.windowed_radon(im, window_len=5, step=2, sigma=2.5)
    
    # check that m is 0 in corners, prop identity in center
    # and correctly oriented on sides
    corner = np.allclose(m[0,0,:,:], np.zeros((2,2)), atol=0.001)
    center = np.allclose(m[10,10,:,:]/m[10,10,0,0], np.eye(2), atol=0.01)
    top_n = np.linalg.eigh(m[0,10,:,:])[1][1,:]
    top = np.allclose(np.abs(top_n[1]), 1, atol=0.01)
    left_n = np.linalg.eigh(m[10,0,:,:])[1][1,:]
    left = np.allclose(np.abs(left_n[0]), 1, atol=0.01)
    
    assert corner and center and top and left
    
    
def test_sigma():
    # test influence of smoothing parameter
    # take a criss-cross pattern and check that at high sigma, anisotropy is 0
    
    im = np.zeros((50,50))
    im[::2, ::2] = 1
    m = wr.windowed_radon(im, window_len=20, step=5, sigma=10)
    # make traceless
    m_tr = m - np.einsum('abii,jk',m,np.eye(2))/2
        
    assert np.allclose(np.linalg.norm(m_tr, axis=(2,3))/ np.linalg.norm(m, axis=(2,3)),
                       np.zeros(m.shape[:2]), atol=0.01)
