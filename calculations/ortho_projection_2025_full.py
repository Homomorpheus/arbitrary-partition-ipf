import sys
import pathlib

sys.path += [str(pathlib.Path(__file__).parent / ".." / "data"), str(pathlib.Path(__file__).parent / ".." / "src")]

import numpy as np

import data_preprocessing as data_pre
import ortho_projection


marginal_01 = np.squeeze(data_pre.origin_dest)
marginal_12 = np.squeeze(data_pre.dest_age)
marginal_02 = np.squeeze(data_pre.origin_age)

shape = (marginal_01.shape[0], marginal_01.shape[1], marginal_12.shape[1])
# table_to_project is the table that will be projected to the space of all
# marginal-fulfilling tables
table_to_project = np.ones(shape)

result = ortho_projection.ortho_project_3d_table(table_to_project, marginal_01, marginal_12, marginal_02)

print(result)
print(f"""
{np.sum(result < 0)} negative cells in the resulting table.
Total cell count is {result.size},
that means that {np.sum(result < 0) / result.size * 100 :.2f}% of cells is negative.""")
