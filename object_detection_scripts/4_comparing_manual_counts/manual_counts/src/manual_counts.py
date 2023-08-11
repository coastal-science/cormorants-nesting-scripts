"""
e.g.
python3 manual_counts.py \
--anno_folder ../input/SNB_2020/VALIDATION/manual_annotations/ \
--removal_record ../input/SNB_2020/VALIDATION/RemovalRecord.xlsx \
--output_file ../output/SNB_2020/VALIDATION/manual_counts_verified.csv \
--label_tree_file /Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/combine_anno_labels/input/snb6/cormorants_2022_label_tree.csv \
--desired_labels Cormorant Nest

python3 manual_counts.py \
--anno_folder ../../../../../cormorants-nesting-scripts-bkp/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/SNB_2020/VALIDATION/manual_annotations \
--removal_record ../../../../../cormorants-nesting-scripts-bkp/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/SNB_2020/VALIDATION/RemovalRecord.xlsx \
--output_file ../output/SNB_2020/VALIDATION/manual_counts_verified.csv \
--label_tree_file /Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/combine_anno_labels/input/snb6/cormorants_2022_label_tree.csv \
--desired_labels Cormrant Nest

"""
import argparse
import ast
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np
import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import box


def get_bbox_coords(raw_anno):
    if pd.isna(raw_anno):
        xmin = -1
        ymin = -1
        xmax = -1
        ymax = -1
    else:
        anno_dict = ast.literal_eval(raw_anno)
        xmin = (anno_dict['x']-anno_dict['w']/2)
        ymin = (anno_dict['y']-anno_dict['h']/2)
        xmax = (anno_dict['x']+anno_dict['w']/2)
        ymax = (anno_dict['y']+anno_dict['h']/2)

    return xmin, ymin, xmax, ymax


def create_detection_geom(tile_name, detection_box):
    y_tile, x_tile = Path(tile_name).name.split('.')[:2]
    y_tile = int(y_tile)
    x_tile = int(x_tile)

    x1, y1, x2, y2 = detection_box  # format: [y1, x1, y2, x2]
    b = box(minx=x_tile+x1, miny=y_tile+y1,
            maxx=x_tile+x2, maxy=y_tile+y2)
    return b


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


def lost_to_geom(img_path, anno):
    box = create_detection_geom(tile_name=img_path,
                                detection_box=get_bbox_coords(anno))
    return box


def merge_duplicate_annotations(df, duplicates):
    non_annotations = df[df['anno.idx'].isna()]

    df = df.dropna(subset=['anno.idx']).set_index('anno.idx')

    canonical_detection_map = {}
    polygons_to_merge = {idx: [(idx, lost_to_geom(img, anno))] for idx, img, anno in
                         zip(df.index,
                             df['img.img_path'],
                             df['anno.data']) if idx in list(duplicates['index_x'])}

    for i_x, i_y in zip(duplicates['index_x'], duplicates['index_y']):
        box_y = lost_to_geom(df['img.img_path'].loc[i_y],
                             df['anno.data'].loc[i_y])

        if i_x in canonical_detection_map:
            canonical_detection = canonical_detection_map[i_x]
        else:
            canonical_detection = i_x

        polygons_to_merge[canonical_detection].append((i_y, box_y))
        canonical_detection_map[i_y] = canonical_detection

    merged_detections = {}
    for canonical_index in polygons_to_merge:
        polygons = [p for i, p in polygons_to_merge[canonical_index]]
        if len(polygons):
            merged_detection = str(list(unary_union(polygons).bounds))
            merged_detections[canonical_index] = merged_detection

    for i, d in df["anno.data"].items():
        if i in merged_detections:
            converted_box = merged_detections[i]
        elif isinstance(d, str):
            tile_path = df['img.img_path'].loc[i]
            converted_box = str(list(lost_to_geom(tile_path, d).bounds))
        elif np.isnan(d):
            converted_box = d
        df.loc[i, 'detection_boxes'] = converted_box

    return pd.concat([non_annotations, df.reset_index()])


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

        # Remove Incorrect Annotations
        incorrect_annos = date_annos_to_remove[date_annos_to_remove['Reason'] == 'not a bird or nest']['Annotation ID']
        temp_annos = annotation_df[~annotation_df['anno.idx'].isin(incorrect_annos)]

        # Remove Out-Of-Bounds Annotations
        out_of_bounds_annos = date_annos_to_remove[date_annos_to_remove['Reason'] == 'outside masking area']['Annotation ID']
        temp_annos = temp_annos[~temp_annos['anno.idx'].isin(out_of_bounds_annos)]

        # Merge Duplicate Annotations
        duplicate_annos = date_annos_to_remove[date_annos_to_remove['Reason'] == 'duplicate over tiles'][['Canonical Annotation', 'Annotation ID']]
        duplicate_annos.columns = ['index_x', 'index_y']
        temp_annos = merge_duplicate_annotations(temp_annos, duplicate_annos)
        duplicate_indices = duplicate_annos['index_y']
        temp_annos = temp_annos[~temp_annos['anno.idx'].isin(list(duplicate_indices))]

        verified_annotations = temp_annos
        verified_annotations.to_csv(out_file.parent.joinpath(f"verified_{file.name}"))

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

    # main(
    #     anno_folder=Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/2020/VALIDATION/manual_annotations"),
    #     removal_record=Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/2020/VALIDATION/RemovalRecord_CanonicalAnnotation.xlsx"),
    #     out_file=Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/output/2020/VALIDATION/manual_counts_verified.csv"),
    #     label_tree_file="/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/2020/VALIDATION/cormorants_2022_label_tree.csv",
    #     desired_labels=["Cormorant", "Nest"]
    # )

    main(
        anno_folder=Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/2021/TEST/manual_annotations"),
        removal_record=Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/2021/TEST/RemovalRecord_CanonicalAnnotation_2021.xlsx"),
        out_file=Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/output/2021/TEST/manual_counts_verified.csv"),
        label_tree_file="/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/manual_counts/input/2021/TEST/cormorants_2022_label_tree.csv",
        desired_labels=["Cormorant", "Nest"]
    )