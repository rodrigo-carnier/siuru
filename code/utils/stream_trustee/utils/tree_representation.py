from sklearn.tree import _tree
import numpy as np


def recurse_tree(tree, node, feature_name, tree_representation):
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        tree_representation += name

        tree_representation = recurse_tree(tree, tree.children_left[node], feature_name,
                                           tree_representation + "{") + "}"
        tree_representation = recurse_tree(tree, tree.children_right[node], feature_name,
                                           tree_representation + "{") + "}"
    else:
        tree_representation += "LEAF"

    return tree_representation


def recurse_tree_with_classes(tree, node, feature_name, tree_representation):
    if node != _tree.TREE_LEAF and tree.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        tree_representation += name

        tree_representation = recurse_tree_with_classes(tree, tree.children_left[node], feature_name,
                                                        tree_representation + "{") + "}"
        tree_representation = recurse_tree_with_classes(tree, tree.children_right[node], feature_name,
                                                        tree_representation + "{") + "}"
    elif node != _tree.TREE_LEAF:
        tree_representation += f"{np.argmax(tree.value[node][0])}"

    return tree_representation


def string_alphabet_order(str1: str, str2: str):
    return (len(str1) != len(str2) and min(str1, str2, key=len)) or sorted([str1, str2], key=str.lower)[0]


def recurse_tree_with_classes_alphabet_order(tree, node, feature_name, tree_representation):
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree.threshold[node]
        tree_representation += name

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        if tree.feature[left_child] != _tree.TREE_UNDEFINED and tree.feature[right_child] != _tree.TREE_UNDEFINED:
            left_child_name = feature_name[left_child]
            right_child_name = feature_name[right_child]

            if left_child_name == string_alphabet_order(left_child_name, right_child_name):
                tree_representation = recurse_tree_with_classes_alphabet_order(tree, left_child, feature_name,
                                                                               tree_representation + "{") + "}"
                tree_representation = recurse_tree_with_classes_alphabet_order(tree, right_child, feature_name,
                                                                               tree_representation + "{") + "}"
            else:
                tree_representation = recurse_tree_with_classes_alphabet_order(tree, right_child, feature_name,
                                                                               tree_representation + "{") + "}"
                tree_representation = recurse_tree_with_classes_alphabet_order(tree, left_child, feature_name,
                                                                               tree_representation + "{") + "}"
        elif tree.feature[left_child] != _tree.TREE_UNDEFINED:
            tree_representation = recurse_tree_with_classes_alphabet_order(tree, right_child, feature_name,
                                                                           tree_representation + "{") + "}"
            tree_representation = recurse_tree_with_classes_alphabet_order(tree, left_child, feature_name,
                                                                           tree_representation + "{") + "}"
        elif tree.feature[right_child] != _tree.TREE_UNDEFINED:
            tree_representation = recurse_tree_with_classes_alphabet_order(tree, left_child, feature_name,
                                                                           tree_representation + "{") + "}"
            tree_representation = recurse_tree_with_classes_alphabet_order(tree, right_child, feature_name,
                                                                           tree_representation + "{") + "}"
        else:
            left_child_value = np.argmax(tree.value[left_child][0])
            right_child_value = np.argmax(tree.value[right_child][0])
            if min(left_child_value, right_child_value) == left_child_value:
                tree_representation += f"{{{left_child_value}}}{{{right_child_value}}}"
            else:
                tree_representation += f"{{{right_child_value}}}{{{left_child_value}}}"
    else:
        tree_representation += f"{np.argmax(tree.value[node][0])}"

    return tree_representation


def tree_to_bracket_notation(tree, feature_names, recurse_func):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    return recurse_func(tree_, 0, feature_name, "{") + "}"
