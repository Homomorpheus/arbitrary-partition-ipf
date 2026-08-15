"""
An implementation of iterative proportional fitting,
extended to arbitrary table partitions.
"""

import numpy as np


class LinConstraint:
    "Base class for a linear constraint on table values"
    def __init__(self):
        raise NotImplementedError

    def enforce(self, joint):
        "perform the iterative scaling on joint"
        raise NotImplementedError

    def error(self, joint, norm=np.inf):
        """compute by how much joint violates the constraint,
        to be measured in norm
        """
        raise NotImplementedError


class Marginal(LinConstraint):
    """Pre-determined value for a table marginal"""
    def __init__(self, index: int, value: np.ndarray):
        """The marginal is obtained by summing over
        the index-th axis, resulting in value.
        """
        self.index: int = index
        self.value: np.ndarray = value

    def enforce(self, joint):
        "enforce the marginal values in-place, in the ipf sense"
        assert self.index < len(joint.shape)
        current_marginal = np.sum(joint, axis=self.index, keepdims=True)
        assert self.value.shape == current_marginal.shape
        zero_mask = current_marginal != 0
        factor = np.ones_like(self.value)
        factor[zero_mask] = self.value[zero_mask] / current_marginal[zero_mask]
        factor[np.logical_and(self.value == 0, current_marginal == 0)] = 1
        joint *= factor
        assert np.allclose(np.sum(joint, axis=self.index, keepdims=True), self.value)

    def error(self, joint, norm=np.inf):
        current_marginal = np.sum(joint, axis=self.index, keepdims=True)
        err = np.linalg.vector_norm(self.value - current_marginal, ord=norm)
        return err



class Tiles(LinConstraint):
    "Fitting to a lower-resolution table."
    def __init__(self, cuts, values):
        """When the table is cut along axis i,
        at indices cuts[i], summing up over the
        resulting sub-tables results in the values
        stored in values.
        """
        # cuts: iterable containing iterables of each axis
        self.cuts = cuts

        self.values = values
        value_cuts = [np.arange(1, dim_size) for dim_size in values.shape]

        for cut, value_cut in zip(cuts, value_cuts, strict=True):
            assert len(cut) == len(value_cut)
        self.values_cut = self.cut_split(np.asarray(values), value_cuts)
        self.values_flat = self.recursive_flatten_list(self.values_cut)

    def cut_split(self, array, cuts, axis=0):
        # recursively split array along cuts, starting with cuts[axis]
        splits = []
        sub_arrays = np.array_split(array, cuts[axis], axis=axis)
        if axis < len(array.shape) - 1:
            for sub_array in sub_arrays:
                splits += self.cut_split(sub_array, cuts, axis + 1)
        else:
            splits = sub_arrays
        return splits

    @classmethod
    def recursive_flatten_list(cls, list_of_lists):
        # recursively turn a nested list into a flat one
        flat = []
        for entry in list_of_lists:
            if isinstance(entry, list):
                flat += cls.recursive_flatten_list(entry)
            else:
                flat.append(entry)

        return flat

    def check_consistency(self, joint):
        assert np.isclose(np.sum(joint), np.sum(self.values))

    def enforce(self, joint):

        self.check_consistency(joint)

        split_array = self.cut_split(joint, self.cuts)
        assert len(self.values_cut) == len(split_array)

        split_array_flat = self.recursive_flatten_list(split_array)
        assert len(self.values_flat) == len(split_array_flat)

        for value, block in zip(self.values_flat, split_array_flat, strict=True):
            block_sum = np.sum(block)
            if block_sum == 0:
                continue
            block[...] = block / block_sum * value

    def error(self, joint, norm=np.inf):
        split_array = self.cut_split(joint, self.cuts)
        split_array_flat = self.recursive_flatten_list(split_array)
        split_array_sums = np.array([np.sum(block) for block in split_array_flat])
        values_flat_reshaped = np.array([np.squeeze(value) for value in self.values_flat])
        err = np.linalg.vector_norm(split_array_sums - values_flat_reshaped, ord=norm)
        return err


class SubtableTiles(Tiles):
    "As Tiles, but only on a part of the table."
    def __init__(self, subtable_cuts, values, subtable_slices):
        """As Tiles; subtable_slices stores the
        slice-objects that are used to obtain
        the sub-table. subtable_cuts is like Tiles.cuts,
        but relative to the sub-table.
        """
        super().__init__(subtable_cuts, values)

        self.subtable_slices = subtable_slices

    def enforce(self, joint):
        # extract subtable
        subtable = joint[*self.subtable_slices]

        # enforce on subtable
        super().enforce(subtable)

        # enforce residual on everything else
        mask = np.ones_like(joint)
        mask[*self.subtable_slices] = 0
        factor = np.ones_like(joint)
        current_residual = np.sum(joint * mask)
        target_residual = np.sum(joint) - np.sum(self.values)
        factor[mask == 1] = target_residual / current_residual
        joint[...] = joint * factor

    def error(self, joint, norm=np.inf):
        subtable = joint[*self.subtable_slices]
        # this deliberately only measures the error on the subtable
        return super().error(subtable, norm=norm)

def ipf(initial_value: np.ndarray, constraints: list[LinConstraint], iterations=10, callback=None):
    result = np.asarray(initial_value, dtype=np.float64, copy=True)
    old_shape = initial_value.shape

    for iteration in range(iterations):
        for constraint in constraints:
            constraint.enforce(result)

        if callback is not None:
            callback(iteration, result, constraints)

    assert old_shape == result.shape
    return result
