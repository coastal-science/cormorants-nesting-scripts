"""
e.g.
python combine_anno_labels.py \
  --anno_file ../input/gab2/gabriola_2_annos.csv \
  --label_tree_file ../input/gab2/cormorants_labels.csv \
  --out_path ../output/gabriola_2_annos_combined.csv \
  --desired_labels Cormorant Nest
"""
import argparse
import pandas as pd
from pathlib import Path
import networkx as nx
import ast


def read_label_tree(label_tree_file):
    df_label_tree = pd.read_csv(label_tree_file)

    # Index to Node map
    index2node = {int(i): n for i, n in zip(df_label_tree['idx'], df_label_tree['name'])}

    # Create Label Tree as NetworkX Graph
    edge_list = [(index2node[int(s)], index2node[int(t)]) for s, t in
                 zip(df_label_tree['parent_leaf_id'], df_label_tree['idx']) if not pd.isna(s)]

    g = nx.DiGraph(edge_list)

    return g


def label2agglabel(label_tree, desired_labels):
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


def convert_label_list(label_list, label_mapper):
    result = []
    for label in ast.literal_eval(label_list):
        result.append(label_mapper[label])
    return result


def combine_labels(anno_file, label_tree_file, desired_labels):
    df_annos = pd.read_csv(anno_file)
    label_tree = read_label_tree(label_tree_file)
    label_mapper = label2agglabel(label_tree, desired_labels)
    df_annos['anno.lbl.name'] = df_annos['anno.lbl.name'].apply(lambda x: convert_label_list(x, label_mapper))
    new_df_annos = df_annos.drop(['anno.lbl.idx', 'anno.lbl.external_id'], axis=1)

    return new_df_annos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that directory.')
    parser.add_argument('--anno_file', help='.')
    parser.add_argument('--out_path', help='.')
    parser.add_argument('--label_tree_file', help='.')
    parser.add_argument('--desired_labels', nargs='+', help='.')
    args = parser.parse_args()

    combined_label_df = combine_labels(Path(args.anno_file),
                                       Path(args.label_tree_file),
                                       args.desired_labels,
                                       )
    combined_label_df.to_csv(Path(args.out_path))


