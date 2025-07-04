"""
Source code for combining annotation labels according to a hierarchical label tree.
"""
import networkx as nx
import pandas as pd
import argparse
import ast

from pathlib import Path


def _read_label_tree(label_tree_file):
    """
    Create a NetworkX.Graph representation of a LOST label tree read in from a CSV file
    """
    df_label_tree = pd.read_csv(label_tree_file)

    # Index to Node map
    index2node = {int(i): n for i, n in zip(df_label_tree['idx'], df_label_tree['name'])}

    # Create Label Tree as NetworkX Graph
    edge_list = [(index2node[int(s)], index2node[int(t)]) for s, t in
                 zip(df_label_tree['parent_leaf_id'], df_label_tree['idx']) if not pd.isna(s)]

    g = nx.DiGraph(edge_list)

    return g


def _create_label_map(label_tree, desired_labels):
    """
    Create a dictionary to map annotated labels to the appropriate aggregate label.
    """
    l2l_map = {}
    for node in label_tree.nodes():
        if node in desired_labels:
            l2l_map[node] = node
        else:
            ancestors = nx.ancestors(label_tree, node)
            for a in ancestors:
                if a in desired_labels:
                    l2l_map[node] = a
    return l2l_map


def _map_label_list(label_list, label_mapper):
    """
    Map a list of annotated labels to their aggregate labels.
    """
    result = []
    for label in ast.literal_eval(label_list):
        result.append(label_mapper[label])
    return result


def combine_annotation_labels(anno_file, label_tree_file, desired_labels):
    """
    Uses a hierarchical label tree to combine any annotations falling within the desired_labels'
    subtrees, relabelling any of the descendant annotations with the most relevant aggregate label
    from the list of desired_labels.

    :param anno_file: Path to the CSV file output by the LOST Annotation tool.
    :param label_tree_file: Path to the label tree used to complete the LOST annotation task whose
                            result is contained within the anno_file.
    :param desired_labels: A list of strings corresponding to the labels to include in the updated
                           annotation DataFrame.
    :return: pd.DataFrame : A updated DataFrame where labels have either been mapped to an ancestor
                            label from within desired_labels or have been removed.
    """
    # Read LOST Annotation File
    df_annos = pd.read_csv(anno_file)

    # Create a dictionary to map from annotation label to aggregate label
    label_tree = _read_label_tree(label_tree_file)
    label_mapper = _create_label_map(label_tree, desired_labels)

    # Map annotated labels to aggregate labels
    df_annos['anno.lbl.name'] = df_annos['anno.lbl.name'].apply(
        lambda x: _map_label_list(x, label_mapper)
    )

    # Clean up the new annotation dataframe
    cols_to_drop = [c for c in ['anno.lbl.idx', 'anno.lbl.ex'] if c in df_annos.columns]
    new_df_annos = df_annos.drop([cols_to_drop], axis=1)

    return new_df_annos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--anno_file', help='.')
    parser.add_argument('--out_path', help='.')
    parser.add_argument('--label_tree_file', help='.')
    parser.add_argument('--desired_labels', nargs='+', help='.')
    args = parser.parse_args()

    combined_label_df = combine_annotation_labels(Path(args.anno_file),
                                                  Path(args.label_tree_file),
                                                  args.desired_labels)

    combined_label_df.to_csv(Path(args.out_path), index=False)
