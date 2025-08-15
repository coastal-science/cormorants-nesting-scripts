"""
e.g.
python3 post_process.py \
--detections_file ../input/2020_SNB/20200603/snb3_v1/detections.csv \
--mask_file ../input/2020_SNB/20200603/span2_mask.csv \
--out_file ../output/span2_mask/20200603_snb3_v1_detections_post_span2.csv \


Take in a detections file and identify and remove duplicates.
We want the option to only do this for specific labels (e.g. Nests).
How do we determine a duplicate?
 * Duplicates will exist on the edges of tiles
 * They will need to be the same label
 * They will need to share some high percentage of their border
 * They will need to match with a detection in another tile

What do we want to do with a duplicate? Two options
 * First option would be to remove the duplicate
 * Second option would be to merge the duplicates

How do we do duplicate detection in the merged image?
 *
"""

import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image

import shapely
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union

import ast
Image.MAX_IMAGE_PIXELS = 3000000000


def find_full_canvas_dims(df):
    tile_ys, tile_xs = zip(*df['image'].transform(lambda x: Path(x).name.split('.')[:2]))
    tile_xs = [int(x) for x in tile_xs]
    tile_ys = [int(y) for y in tile_ys]
    return max(tile_xs) + 1, max(tile_ys) + 1


def create_tile_geom(tile_name):
    y_tile, x_tile = Path(tile_name).name.split('.')[:2]
    y_tile = int(y_tile)
    x_tile = int(x_tile)
    b = box(minx=x_tile, miny=y_tile, maxx=x_tile+1, maxy=y_tile+1)
    return b


def create_detection_geom(tile_name, detection_box):
    y_tile, x_tile = Path(tile_name).name.split('.')[:2]
    y_tile = int(y_tile)
    x_tile = int(x_tile)

    y1, x1, y2, x2 = detection_box  # format: [y1, x1, y2, x2]
    b = box(minx=x_tile+x1, miny=y_tile+y1,
            maxx=x_tile+x2, maxy=y_tile+y2)
    return b


def tiles_share_edge(tile1, tile2):
    if tile1.touches(tile2):
        intersect = tile1.intersection(tile2)
        if isinstance(intersect, LineString):
            return True
    return False


def load_mask(f, resize_dims, mask_name=None):
    df = pd.read_csv(f)
    if len(df) == 1:
        print(f"Loading mask for {df['img.img_path'].iloc[0]}")
        mask_points = [(d['x'], d['y']) for d in ast.literal_eval(df['anno.data'].iloc[0])]
        resized_points = [(x*resize_dims[0], y*resize_dims[1]) for x, y in mask_points]
        mask_geom = Polygon(resized_points)
    elif mask_name:
        matching_masks = df[df['img.img_path'].str.contains(mask_name)]
        if len(matching_masks) == 1:
            print(f"Loading mask for {matching_masks['img.img_path'].iloc[0]}")
            mask_points = [(d['x'], d['y']) for d in ast.literal_eval(matching_masks['anno.data'].iloc[0])]
            resized_points = [(x * resize_dims[0], y * resize_dims[1]) for x, y in mask_points]
            mask_geom = Polygon(resized_points)

    return mask_geom


def find_duplicate_detections(to_process_df):
    # Create Geometries
    # (1) Tiles
    to_process_df['tile'] = to_process_df['image'].apply(create_tile_geom)
    # (3) Detections
    box_geoms = []
    for image, det_box in zip(to_process_df['image'], to_process_df['detection_boxes']):
        box_geoms.append(create_detection_geom(image, ast.literal_eval(det_box)))
    to_process_df['detection_box_geom'] = box_geoms

    # Cross Merge
    cross0 = to_process_df.merge(right=to_process_df, how='cross')
    cross = cross0[cross0['index_x'] < cross0['index_y']]

    # Filter the Cross Merge
    # (1) Make sure the tiles aren't identical
    cross2 = cross[~(cross['image_x'] == cross['image_y'])]
    # (2) make sure the detections are the same label
    cross3 = cross2[cross2['detection_classes_x'] == cross2['detection_classes_y']]
    # (3) make sure the tiles have an edge that touches (not just a corner)
    bool_filter = []
    for tile1, tile2 in zip(cross3['tile_x'], cross3['tile_y']):
        bool_filter.append(tiles_share_edge(tile1, tile2))
    cross4 = cross3[bool_filter]

    # Check if the two detections intersect
    duplicates = cross4[gpd.GeoSeries(cross4['detection_box_geom_x']).buffer(distance=0.01).intersects(gpd.GeoSeries(cross4['detection_box_geom_y']).buffer(distance=0.01))]

    return duplicates


def remove_duplicate_nests(df, merge_duplicates=False):
    assert not df.empty, "No detections provided. Cannot remove duplicates."
    df = df.sort_values(by='detection_scores', ascending=False).reset_index(drop=True)

    nest_class = 1
    # Only remove duplicates from the specified classes
    to_process_df = df[df['detection_classes'].isin([nest_class])]

    # Re-Indexing so when we do the cross merge we can remove half the matches
    # (A-B B-A only needs to be checked once).
    to_process_df = to_process_df.reset_index()

    # Find the indices of duplicates to remove
    duplicates = find_duplicate_detections(to_process_df)

    if merge_duplicates:
        # Merge duplicates & Convert boxes to full canvas dimensions
        df = merge_duplicate_nests(df, duplicates)
    else:
        # Convert all boxes to full canvas dimensions
        for i, d in df['detection_boxes'].items():
            tile_path = df['image'].loc[i]
            converted_box = str(
                list(create_detection_geom(tile_path, ast.literal_eval(d)).bounds))
            df.loc[i, 'detection_boxes'] = converted_box

    # Remove duplicates by their index
    duplicate_indices = duplicates['index_y']
    no_duplicates = df[~df.index.isin(duplicate_indices)]

    return no_duplicates


def merge_duplicate_nests(df, duplicates):
    canonical_detection_map = {}
    polygons_to_merge = {i: [(i, b)] for i, b in
                         zip(duplicates['index_x'], duplicates['detection_box_geom_x'])}

    for i_x, i_y, box_x, box_y in zip(duplicates['index_x'],
                                      duplicates['index_y'],
                                      duplicates['detection_box_geom_x'],
                                      duplicates['detection_box_geom_y']):
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

    for i, d in df['detection_boxes'].items():
        if i in merged_detections:
            converted_box = merged_detections[i]
        else:
            tile_path = df['image'].loc[i]
            converted_box = str(list(create_detection_geom(tile_path, ast.literal_eval(d)).bounds))
        df.loc[i, 'detection_boxes'] = converted_box

    return df


def apply_mask(mask_file, original_pano, tile_size, df):
    assert not df.empty, "No detections provided. Cannot apply mask."
    im_w, im_h = Image.open(original_pano).size
    resize_dims = (im_w/tile_size, im_h/tile_size)

    mask = load_mask(mask_file, resize_dims, mask_name=Path(original_pano).stem)

    box_geoms = []
    for image, det_box in zip(df['image'], df['detection_boxes']):
        box_geoms.append(create_detection_geom(image, ast.literal_eval(det_box)))
    df['detection_box_geom'] = box_geoms

    gs = gpd.GeoSeries(df['detection_box_geom'])
    gs_rescaled = gs.scale(xfact=0.5, yfact=0.5)
    masked_df = df[gs_rescaled.intersects(mask)]

    return masked_df


def remove_nan_detections(df):
    return df[~df['detection_scores'].isna()]


def birds_in_nests(df):
    def out_of_nest(geom):
        if sum(nest_df.geometry.buffer(0.55).intersects(shapely.affinity.scale(geom, xfact=2, yfact=1))) > 0:
            return False
        return True

    df = df.set_geometry('detection_box_geom')
    # Get Nests
    nest_df = df[df['detection_classes'] == 1]

    # Get Birds
    bird_df = df[df['detection_classes']==0]

    # Find the index of any bird out of a nest
    out_of_nest_filter = bird_df.geometry.apply(out_of_nest)
    out_of_nest_idx = out_of_nest_filter[out_of_nest_filter].index

    return df.drop(out_of_nest_idx)


def main(detection_csv, mask=False, deduplicate_nests=False, merge_duplicate_nests=False,  
         mask_file='', original_pano='', tile_size=1000, out_file=''):
    df = pd.read_csv(detection_csv)
    df = remove_nan_detections(df)

    if mask and Path(mask_file).is_file():
        print("Applying Mask...")
        df = apply_mask(mask_file, original_pano, tile_size, df)

    if deduplicate_nests:
        print("Removing Duplicate Nests...")
        df = remove_duplicate_nests(df, merge_duplicates=merge_duplicate_nests)

    print("Writing Output File ...")
    
    # Avoid detection_id=0 to prevent messy data / confusion with NA, "", missing value etc
    id_offset = 11 # ids are two-digit (and above)
    df = df.set_axis(pd.RangeIndex(start=id_offset, stop=len(df)+id_offset, step=1)) # reportedly faster than df.index += 1
    
    df.to_csv(out_file, index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mask', help='.', action='store_true')
    parser.add_argument('--deduplicate_nests', help='.', action='store_true')
    parser.add_argument('--merge_duplicate_nests', help='.', action='store_true')
    parser.add_argument('--detections_file', help='.', type=str, required=True)
    parser.add_argument('--original_pano', help='.', type=str, required=True)
    parser.add_argument('--tile_size', help='.', type=int, required=True)
    parser.add_argument('--out_file', help='.', type=str, required=True)
    parser.add_argument('--mask_file', help='Use this', type=str)
    args = parser.parse_args()

    main(detection_csv=args.detections_file, 
         mask=args.mask, 
         deduplicate_nests=args.deduplicate_nests,
         merge_duplicate_nests=args.merge_duplicate_nests,
         mask_file=args.mask_file, 
         original_pano=args.original_pano,
         tile_size=args.tile_size, 
         out_file=args.out_file)
