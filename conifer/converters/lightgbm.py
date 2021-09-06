from typing import *
import numpy as np
from .converter import addParentAndDepth, padTree
from ..model import model
from lightgbm import Booster, LGBMModel, LGBMClassifier
from rich import print
from rich.pretty import pprint
from tqdm.auto import tqdm


def convert_bdt(bdt: Booster, n_classes=1):
    tdf = bdt.trees_to_dataframe()
    print(tdf.head())
    dump_dict = bdt.dump_model(bdt.best_iteration)

    ensembleDict = {'max_depth': int(tdf.node_depth.max()), 'n_trees': bdt.num_trees(),
                    'n_features': bdt.num_feature(),
                    'n_classes': n_classes, 'trees': [],
                    'init_predict': [0.] * n_classes,
                    'norm': 1}
    # pprint(ensembleDict)
    pprint(f'{ensembleDict["max_depth"]=}')
    for tree in dump_dict['tree_info']:
        tree = treeToDict(bdt, tree)
        # pprint(tree)
        tree = padTree(ensembleDict, tree)
        # NB node values are multiplied by the learning rate here, saving work in the FPGA
        # TODO: Check if this is needed
        # pprint(tree['value'])
        # tree['value'] = (np.array(tree['value'])[:, 0, 0] * bdt.learning_rate).tolist()
        # pprint(tree['value'])
        ensembleDict['trees'].append([tree])

    return ensembleDict


def convert(bdt: Union[Booster, LGBMModel]):
    if isinstance(bdt, LGBMClassifier):
        convert_bdt(bdt.booster_, bdt.n_classes_)
    if isinstance(bdt, Booster):
        return convert_bdt(bdt)
    else:
        return convert_bdt(bdt.booster_)


def is_internal(node: dict):
    return 'split_index' in node.keys()


def node_index(node: dict, num_internal: int):
    if is_internal(node):
        return node['split_index']
    else:
        return node['leaf_index'] + num_internal


def node_value(node: dict):
    if is_internal(node):
        return node['internal_value']
    else:
        return node['leaf_value']


def node_threshold(node: dict):
    if is_internal(node):
        return node['threshold']
    else:
        return -2.0  # Use -2 instead of None


def traverse_to_list(tree: dict, result: dict, num_internal: int):
    """Preorder enumeration"""

    keys = tree.keys()
    current_index = node_index(tree, num_internal)
    # print(current_index)
    result['value'][current_index] = node_value(tree)
    result['threshold'][current_index] = node_threshold(tree)

    for side in 'right', 'left':
        if f'{side}_child' in keys:
            # print(tree[f'{side}_child'])
            result[f'children_{side}'][current_index] = node_index(tree[f'{side}_child'], num_internal)
            result['feature'][current_index] = tree['split_feature']
            # Set parents
            traverse_to_list(tree[f'{side}_child'], result, num_internal)


def get_greatest_internal_index(tree):
    if is_internal(tree):
        sub_trees = []
        if 'right_child' in tree.keys():
            sub_trees.append(tree['right_child'])
        if 'left_child' in tree.keys():
            sub_trees.append(tree['left_child'])
        return max(tree['split_index'], *map(get_greatest_internal_index, sub_trees))
    else:
        return -1


def treeToDict(bdt, tree):
    # Extract the relevant tree parameters
    num_leaves = tree['num_leaves']
    num_nodes = 2 * num_leaves - 1
    num_internal = get_greatest_internal_index(tree['tree_structure']) + 1
    # print(f'{num_internal=}')

    treeDict = {
        'feature': [-1] * num_nodes, 'threshold': [-1] * num_nodes, 'value': [None] * num_nodes,
        'children_left': [-1] * num_nodes, 'children_right': [-1] * num_nodes
    }
    # print('Tree Structure')
    # pprint(tree['tree_structure'])
    traverse_to_list(tree['tree_structure'], treeDict, num_internal)
    # print('traversed')
    # pprint(treeDict)

    treeDict = addParentAndDepth(treeDict)
    # print('finished treeToDict')
    pprint(treeDict)
    return treeDict
