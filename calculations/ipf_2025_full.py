import sys
import pathlib

sys.path += [str(pathlib.Path(__file__).parent / ".." / "data"), str(pathlib.Path(__file__).parent / ".." / "src")]

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

def ipf_estimate(plot_iteration_errors=False):
    """Perform IPF estimation, return result of IPF.
    Result is a 125 x 125 x 96 table of origin, destination, age.
    IPF is applied on a constant table and complete marginal data.
    """

    marginals = [
        ipf.Marginal(2, data_pre.origin_dest),
        ipf.Marginal(1, data_pre.origin_age),
        ipf.Marginal(0, data_pre.dest_age),
    ]
    constraints = marginals

    if plot_iteration_errors:
        modulo = 10
        errors = []
        def callback(iteration, table, constraints):
            if iteration % modulo != 0:
                return

            errors.append([constraint.error(table) for constraint in constraints])
    else:
        callback=None

    shape = (data_pre.origin_dest.shape[0], data_pre.origin_dest.shape[1], data_pre.origin_age.shape[2])
    iterations=200
    mle = ipf.ipf(np.ones(shape), constraints, iterations, callback=callback)

    if plot_iteration_errors:
        # plot errors
        errors = np.array(errors)
        fig, ax = plt.subplots()
        for i in range(errors.shape[1]):
            ax.plot(range(1, iterations + 1, modulo), errors[:, i], label=f"Constraint N°{i+1}")
        ax.set_ylabel("Error in marginal values")
        ax.set_xlabel("Amount of iterations")
        ax.legend()
        plt.show()

    return mle


if __name__=="__main__":
    mle = ipf_estimate(plot_iteration_errors=True)

    masked_subtable = data_pre.extract_subtable(pathlib.Path(__file__).parent / ".." / "data" / "sub_table_sample_2.csv", data_pre.names_places, 2, data_pre.names_places, 4, data_pre.names_age, 0, value_index=5,  left=0, upper=7, right=-1, lower=-11)
    approx_dev, p_val, df = goodness_of_fit.approx_deviance_nan_masked_subtable_hom_assoc_3d(masked_subtable, mle, marginals)
    print(
    f"Approximate deviance is G²≈{approx_dev},\n",
    f"corresponding p-value is p≈{p_val},\n",
    f"calculated for {df} degrees of freedom.\n",
    sep="",
    )


    rel_err, abs_err_where_true_data_zero = goodness_of_fit.cell_wise_error_subtable(masked_subtable, mle)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].boxplot(rel_err, label="Relative error", whis=(0, 100))
    ax[0, 1].boxplot(abs_err_where_true_data_zero, label="Absolute error where true data zero", whis=(0, 100))
    ax[1, 0].hist(rel_err, label="Relative error", bins=50)
    ax[1, 1].hist(abs_err_where_true_data_zero, label="Absolute error where true data zero", bins=50)
    fig.legend()
    plt.show()

    print("Statistics for the relative error:")
    print(f"lower whisker={np.min(rel_err)},")
    print(f"lower quartile={np.quantile(rel_err, 0.25)},")
    print(f"median={np.median(rel_err)},")
    print(f"upper quartile={np.quantile(rel_err, 0.75)},")
    print(f"upper whisker={np.max(rel_err)},")
    print(f"average={np.mean(rel_err)},")
    print()

    print("Statistics for the relative error where the true data is zero:")
    print(f"lower whisker={np.min(abs_err_where_true_data_zero)},")
    print(f"lower quartile={np.quantile(abs_err_where_true_data_zero, 0.25)},")
    print(f"median={np.median(abs_err_where_true_data_zero)},")
    print(f"upper quartile={np.quantile(abs_err_where_true_data_zero, 0.75)},")
    print(f"upper whisker={np.max(abs_err_where_true_data_zero)},")
    print(f"average={np.mean(abs_err_where_true_data_zero)},")
