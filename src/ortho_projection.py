"""Orthogonal projection of a 3d table to the
space of tables with specific marginals."""

import numpy as np


def consistency_check_3d(marginal_01, marginal_12, marginal_02):
    "chekc if marginals are consistent"
    is_consistent = marginal_01.shape[0] == marginal_02.shape[0] \
    and marginal_01.shape[1] == marginal_12.shape[0] \
    and marginal_12.shape[1] == marginal_02.shape[1] \
    and np.allclose(np.sum(marginal_01, axis=1), np.sum(marginal_02, axis=1)) \
    and np.allclose(np.sum(marginal_01, axis=0), np.sum(marginal_12, axis=1)) \
    and np.allclose(np.sum(marginal_12, axis=0), np.sum(marginal_02, axis=0))

    return is_consistent

def check_marginals_3d(table, marginal_01, marginal_12, marginal_02):
    "check whether table fulfills the marginals"
    fulfills_marginals = np.allclose(marginal_01, np.sum(table, axis=2)) \
    and np.allclose(marginal_12, np.sum(table, axis=0)) \
    and np.allclose(marginal_02, np.sum(table, axis=1))

    return fulfills_marginals

def particular_table_3d(marginal_01, marginal_12, marginal_02):
    "generate a 3d table with the given marginals"

    assert consistency_check_3d(marginal_01, marginal_12, marginal_02)

    shape = (marginal_01.shape[0], marginal_01.shape[1], marginal_12.shape[1])

    particular = np.zeros(shape)

    particular[0, : , :] += marginal_12
    particular[:, 0, :] += marginal_02
    particular[:, :, 0] += marginal_01

    particular[0, 0, :] -= np.sum(marginal_12, axis=0)
    particular[0, :, 0] -= np.sum(marginal_01, axis=0)
    particular[:, 0, 0] -= np.sum(marginal_01, axis=1)

    particular[0, 0, 0] += np.sum(marginal_01)

    assert check_marginals_3d(particular, marginal_01, marginal_12, marginal_02)

    return particular


def ortho_project_3d_table(table_to_project, marginal_01, marginal_12, marginal_02):
    "project table_to_project to the space of tables with specified marginals, in the Frobenius norm"

    particular = particular_table_3d(marginal_01, marginal_12, marginal_02)

    shape = particular.shape

    to_project_homogenous = table_to_project - particular
    projection_homogenous = to_project_homogenous

    # enforce sum(projection_homogenous, axis=0) = 0
    projection_homogenous = projection_homogenous - np.sum(projection_homogenous, axis=0, keepdims=True) / shape[0] * np.ones(shape)

    # enforce sum(projection_homogenous, axis=1) = 0
    projection_homogenous = projection_homogenous - np.sum(projection_homogenous, axis=1, keepdims=True) / shape[1] * np.ones(shape)

    # enforce sum(projection_homogenous, axis=2) = 0
    projection_homogenous = projection_homogenous - np.sum(projection_homogenous, axis=2, keepdims=True) / shape[2] * np.ones(shape)

    projection = projection_homogenous + particular

    assert check_marginals_3d(projection, marginal_01, marginal_12, marginal_02)

    return projection
