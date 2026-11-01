# -*- coding: utf-8 -*-
"""
This file contains the script for defining characteristic functions
and using them as a way to embed distributional information
in Euclidean space
"""
import math
import numpy as np


def charac_function(time_points, temp):
    temp2 = temp.T.tolil()
    d = temp2.data
    n_timepnts = len(time_points)
    n_nodes = temp.shape[1]
    final_sig = np.zeros((2 * n_timepnts, n_nodes))
    zeros_vec = np.array([1.0 / n_nodes*(n_nodes - len(d[i])) for i in range(n_nodes)])
    for i in range(n_nodes):
        final_sig[::2, i] = zeros_vec[i] +\
                            1.0 / n_nodes *\
                            np.cos(np.einsum("i,j-> ij",
                                             time_points,
                                             np.array(d[i]))).sum(1)
    for it_t, t in enumerate(time_points):
        final_sig[it_t * 2 + 1, :] = 1.0 / n_nodes * ((t*temp).sin().sum(0))

    return final_sig


def charac_function_multiscale(heat, time_points):
    final_sig = []
    for i in heat.keys():
        final_sig.append(charac_function(time_points, heat[i]))
    return np.vstack(final_sig).T
