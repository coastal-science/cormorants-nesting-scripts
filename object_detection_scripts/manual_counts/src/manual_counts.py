"""
e.g.
python3 manual_counts.py \
--anno_folder ../input/SNB_2020/VALIDATION/manual_annotations/ \
--removal_record ../input/SNB_2020/VALIDATION/RemovalRecord.xlsx \
--output_file ../output/SNB_2020/VALIDATION/manual_counts_verified.csv \
--label_tree_file /Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/combine_anno_labels/input/snb6/cormorants_2022_label_tree.csv \
--desired_labels Cormorant Nest

"""
import argparse
import ast
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import networkx as nx


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


def main(anno_folder, removal_record, out_file, label_tree_file=None,
         desired_labels=['Cormorant', 'Nest']):

    annos_to_remove = pd.read_excel(removal_record)

    all_verified_annotations = pd.DataFrame()
    for file in anno_folder.glob('*.csv'):
        # Read manual annotations file & prepare labels
        annotation_df = pd.read_csv(file)
        annotation_df['Label'] = annotation_df['anno.lbl.name'].apply(
            lambda x: ast.literal_eval(x)).explode()

        # Find the date corresponding to the file
        file_date_str = re.search(r"\D\d{8}\D", file.name).group()[1:-1]
        file_date = datetime.strptime(file_date_str, '%Y%m%d')

        # Filter removal record to only include annotations from the relevant date
        date_annos_to_remove = annos_to_remove[annos_to_remove['Date'] == file_date]

        # Remove Annotations
        verified_annotations = annotation_df[~annotation_df['anno.idx'].isin(date_annos_to_remove['Annotation ID'])]
        print(f"{file_date}: Removed {len(annotation_df) - len(verified_annotations)} annotations from {file.name} manual counts")
        verified_annotations = verified_annotations[['Label']].dropna()
        verified_annotations['Date'] = file_date

        all_verified_annotations = pd.concat([all_verified_annotations, verified_annotations])

    # Aggregate labels
    label_tree = read_label_tree(label_tree_file)
    label_mapper = label2agglabel(label_tree, desired_labels)
    all_verified_annotations['Label'] = all_verified_annotations['Label'].apply(lambda x: label_mapper[x])

    # Count number of annotations, grouped by date & label
    all_verified_annotations['count'] = 1
    summary = all_verified_annotations.groupby(['Date', 'Label']).count()
    summary = summary.reset_index().pivot(index='Date', columns='Label', values='count')

    # Write file out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--anno_folder', help='Folder containing manual annotations.', type=str)
    parser.add_argument('--removal_record', help='File containing annotations to remove.', type=str)
    parser.add_argument('--output_file', help='', type=str)
    parser.add_argument('--label_tree_file', help='', type=str)
    parser.add_argument('--desired_labels', nargs='+', help='.')
    args = parser.parse_args()

    main(Path(args.anno_folder), Path(args.removal_record), Path(args.output_file),
         label_tree_file=args.label_tree_file, desired_labels=args.desired_labels)
