"""
Functions for calculating statistics regarding annotations, such as bounding box size, annotation
time, and annotation counts.
"""
import ast
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import json


def annotation_time(annotation_files, level, remove_admin=True):
    df = pd.concat([pd.read_csv(f) for f in annotation_files])

    # Remove demonstration time
    if remove_admin:
        df = df[df['img.annotator'] != 'admin']

    if level == 'all':
        df_time = df[['img.idx', 'img.anno_time']].drop_duplicates()
        num_secs = df_time['img.anno_time'].sum()
        print(f"Total annotation time (seconds): {round(num_secs, 2)}")

    elif level == 'annotation':
        df = df.groupby(['img.idx', 'img.anno_time'])['anno.data'].count().reset_index()
        mean_time_anno = df['img.anno_time'].sum() / df['anno.data'].sum()
        print(f"Average annotation time per annotation (seconds): {round(mean_time_anno, 2)}")

    elif level == 'tile':
        df = df[['img.idx', 'img.anno_time']].drop_duplicates()
        mean_time_tile = df['img.anno_time'].mean()
        print(f"Average annotation time per tile (seconds): {round(mean_time_tile, 2)}")

    elif level == 'panorama':
        df['pano'] = df['img.img_path'].apply(lambda x: Path(x).parent.name)
        df = df[['pano', 'img.idx', 'img.anno_task_id', 'img.anno_time']].drop_duplicates()
        mean_time_pano = np.mean(df.groupby(['pano', 'img.anno_task_id'])['img.anno_time'].sum())
        print(f"Average annotation time per panorama (seconds): {round(mean_time_pano, 2)}")


def annotation_counts(annotation_files, level, remove_admin=True):
    df = pd.concat([pd.read_csv(f) for f in annotation_files])

    # Remove demonstration time
    if remove_admin:
        df = df[df['img.annotator'] != 'admin']

    if level == 'all':
        df = df[~df['anno.idx'].isna()]
        df['label'] = df['anno.lbl.name'].apply(lambda x: ast.literal_eval(x)[0].split()[0])
        print(f"Total Number of Annotations:")
        print(df.groupby(['label']).count()['anno.data'].to_string(header=False))

    elif level == 'tile':
        counts = df.groupby(['img.img_path']).agg({'anno.data': 'count'})['anno.data']
        print(f"Annotations per Tile:")
        print(counts.describe().drop(index=['count']).round(2))

    elif level == 'panorama':
        df['pano'] = df['img.img_path'].apply(lambda x: Path(x).parent.name)
        counts = df.groupby(['pano', 'img.anno_task_id']).agg({'anno.data': 'count'})['anno.data']
        print(f"Annotations per Panorama:")
        print(counts.describe().drop(index=['count']).round(2))


def annotation_size(annotation_files, level, remove_admin=True):
    # Average size of an annotation
    def calc_anno_size(data):
        data = ast.literal_eval(data)
        return data['w'] * data['h']

    df = pd.concat([pd.read_csv(f) for f in annotation_files])

    # Remove demonstration time
    if remove_admin:
        df = df[df['img.annotator'] != 'admin']

    # Remove tiles without annotations
    df = df[~df['anno.idx'].isna()]

    df['label'] = df['anno.lbl.name'].apply(lambda x: ast.literal_eval(x)[0].split()[0])
    df['bbox_size'] = df['anno.data'].apply(calc_anno_size)*100
    df['bbox_w'] = df['anno.data'].apply(lambda x: ast.literal_eval(x)['w']*100)
    df['bbox_h'] = df['anno.data'].apply(lambda x: ast.literal_eval(x)['h']*100)

    summary = df.groupby('label').agg({
        'bbox_size': ['mean', 'median'],
        'bbox_w': ['mean', 'median'],
        'bbox_h': ['mean', 'median'],
    }).round(2)

    summary.columns = summary.columns.set_levels(['BBox Size (% of tile)',
                                                  'BBox Width (% of tile width)',
                                                  'BBox Height (% of tile height)'], level=0)

    for g, d in summary.groupby(axis=1, level=0):
        print(f"\n*** *** {g} *** ***")
        print(d.droplevel(axis=1, level=0).reset_index().to_string(index=False))


def read_anno_list(json_file):
    with open(json_file, 'r') as f:
        anno_list = json.load(f)
    return anno_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that directory.')
    parser.add_argument('--anno_list', help='directory containing annotations.')
    parser.add_argument('--stat', help='One of {anno-time}')
    parser.add_argument('--level', help='One of {annotation, tile, panorama}',)
    parser.add_argument('--only_tiles_with_annos', action='store_true')
    args = parser.parse_args()

    stat_directory = {'time': annotation_time,
                      'size': annotation_size,
                      'count': annotation_counts}

    stat_directory[args.stat](read_anno_list(args.anno_list), level=args.level)