import sys
sys.path += ["../data", "../src"]

import numpy as np
import matplotlib.pyplot as plt

import data_preprocessing as data_pre
import ipf
import goodness_of_fit


marginals = [
    ipf.Marginal(2, data_pre.origin_dest),
    ipf.Marginal(1, data_pre.origin_age),
    ipf.Marginal(0, data_pre.dest_age),
]
constraints = marginals + [
    ipf.Tiles(data_pre.federal_cuts, data_pre.federal_values),
    ipf.SubtableTiles(data_pre.vienna_cuts, data_pre.vienna_values, data_pre.vienna_subtable)
]

errors = []
def callback(iteration, table, constraints):
    if iteration % 10 != 0:
        return

    errors.append([constraint.error(table) for constraint in constraints])


shape = (data_pre.origin_dest.shape[0], data_pre.origin_dest.shape[1], data_pre.origin_age.shape[2])
initial_value = np.ones(shape)
mle = ipf.ipf(initial_value, constraints, iterations=200, callback=callback)

# plotting errors
errors = np.array(errors)
for i in range(errors.shape[1]):
    plt.plot(errors[:, i], label=f"Constraint N°{i+1}")
plt.legend()
plt.show()


def df_with_federal_and_vienna(mle, marginals, tiles, vienna):
    # df for model test, with marginals and federal data

    # number of cells
    T_e = np.size(mle)
    # number of independent parameters estimated
    T_p = 1 \
    + (mle.shape[0] - 1) + (mle.shape[1] - 1) + (mle.shape[2] - 1) \
    + (mle.shape[0] - 1) * (mle.shape[1] - 1) + (mle.shape[0] - 1) * (mle.shape[2] - 1) + (mle.shape[1] - 1) * (mle.shape[2] - 1) \
    + (tiles.values.shape[0] - 1) * (tiles.values.shape[1] - 1) * (tiles.values.shape[2] - 1) \
    + (vienna.values.shape[0] - 1) * (vienna.values.shape[1] - 1) * (vienna.values.shape[2] - 1) \
    - vienna.values.shape[2] # last term compensates for tiles losing some independence in Vienna
    assert T_p != 1

    V = T_e - T_p

    # elementary cells with zero estimates
    z_e = np.sum(np.isclose(mle, 0.))
    # parameters that cannot be estimated
    # formula: zero cells from marginals minus zero cells from 1d-marginals plus zero cells from tiles minus zero cells from marginals of tiles plus zero cells from 1d-marginals of tiles plus zero cells from Vienna minus marginal zero cells from Vienna plus 1d-marginal cells from Vienna
    # interation terms between tiles and vienna should not be necessary due to a lack of zero cells
    sub_marginals = [np.sum(mle, axis=(0, 1)), np.sum(mle, axis=(0, 2)), np.sum(mle, axis=(1, 2))]
    tiles_marginals = [np.sum(tiles.values, axis=0, keepdims=True), np.sum(tiles.values, axis=1, keepdims=True), np.sum(tiles.values, axis=2, keepdims=True)]
    tiles_submarginals = [np.sum(tiles.values, axis=(0, 1)), np.sum(tiles.values, axis=(1, 2)), np.sum(tiles.values, axis=(0, 2))]
    vienna_marginals = [np.sum(vienna.values, axis=0, keepdims=True), np.sum(vienna.values, axis=1, keepdims=True), np.sum(vienna.values, axis=2, keepdims=True)]
    vienna_submarginals = [np.sum(vienna.values, axis=(0, 1)), np.sum(vienna.values, axis=(1, 2)), np.sum(vienna.values, axis=(0, 2))]
    z_p = np.sum(np.isclose(marginals[0].value, 0.)) + np.sum(np.isclose(marginals[1].value, 0.)) + np.sum(np.isclose(marginals[2].value, 0.)) \
    - np.sum(np.isclose(sub_marginals[0], 0.)) - np.sum(np.isclose(sub_marginals[1], 0.)) - np.sum(np.isclose(sub_marginals[2], 0.)) \
    \
    + np.sum(np.isclose(tiles.values, 0.)) \
    - np.sum(np.isclose(tiles_marginals[0], 0.)) - np.sum(np.isclose(tiles_marginals[1], 0.)) - np.sum(np.isclose(tiles_marginals[1], 0.)) \
    + np.sum(np.isclose(tiles_submarginals[0], 0.)) + np.sum(np.isclose(tiles_submarginals[1], 0.)) + np.sum(np.isclose(tiles_submarginals[2], 0.)) \
    \
    + np.sum(np.isclose(vienna.values, 0.)) \
    - np.sum(np.isclose(vienna_marginals[0], 0.)) - np.sum(np.isclose(vienna_marginals[1], 0.)) - np.sum(np.isclose(vienna_marginals[1], 0.)) \
    + np.sum(np.isclose(vienna_submarginals[0], 0.)) + np.sum(np.isclose(vienna_submarginals[1], 0.)) + np.sum(np.isclose(vienna_submarginals[2], 0.))
    assert not np.isclose(np.sum(mle), 0.)

    V_prime = V - z_e + z_p

    return V_prime

df = df_with_federal_and_vienna(mle, marginals, tiles=constraints[3], vienna=constraints[4])
masked_subtable = data_pre.extract_subtable("../data/sub_table_sample_1.csv", data_pre.names_places, 3, data_pre.names_places, 2, data_pre.names_age, 0, value_index=4,  left=1, upper=10, right=-1, lower=-9)
approx_dev, p_val = goodness_of_fit.approx_deviance_nan_masked_subtable(masked_subtable, mle, df=df)
print(approx_dev, p_val, df)
