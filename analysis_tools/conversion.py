import ROOT
import numpy as np
import awkward as ak

def to_root_tree(array_or_arrays, tree_name, branch_name_or_branch_names):
    if isinstance(array_or_arrays, list):
        assert isinstance(branch_name_or_branch_names, list)
        assert len(array_or_arrays) == len(branch_name_or_branch_names)
        assert all([len(array) == len(array_or_arrays[0]) for array in array_or_arrays])

        arrays = array_or_arrays
        branch_names = branch_name_or_branch_names
    else:
        arrays = [array_or_arrays]
        branch_names = [branch_name_or_branch_names]
    
    for array in arrays:
        if isinstance(array, ak.Array):
            array = ak.to_numpy(array)

    tree = ROOT.TTree(tree_name, tree_name)

    x = [np.zeros(1, dtype=float) for _ in arrays]
    for i, branch_name in enumerate(branch_names):
        tree.Branch(branch_name, x[i], branch_name + "/D")

    for i in range(len(arrays[0])):
        for j, array in enumerate(arrays):
            x[j][0] = array[i]
        tree.Fill()

    return tree