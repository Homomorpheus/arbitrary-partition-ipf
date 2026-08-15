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
constraints = marginals

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


# test goodness of fit
masked_subtable = data_pre.extract_subtable("../data/sub_table_sample_1.csv", data_pre.names_places, 3, data_pre.names_places, 2, data_pre.names_age, 0, value_index=4,  left=1, upper=10, right=-1, lower=-9)
approx_dev, p_val, df = goodness_of_fit.approx_deviance_nan_masked_subtable_hom_assoc_3d(masked_subtable, mle, marginals)
print(approx_dev, p_val, df)
