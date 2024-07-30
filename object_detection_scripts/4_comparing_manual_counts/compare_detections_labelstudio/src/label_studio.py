import argparse
import pandas as pd
from pathlib import Path
import json

def main(input_file:str, compare_with, output_file):
    """Reshape the input csv into a shape that can be used for a Label Studio labeling task 
    to compare individual detections from one day against the next day's panorama, the status of nest

    Args:
        input_file (Path): input csv (generated from draw_final_detections.py)
                        csv must contain minimum columns for 
                        'index' (int), 
                        'indv_name' (path to detection png image), 
                        'detection_classes' (int), 
                        'image' (str from tiff_tiles)
        compare_with (str): Name of panorama for day 2
        
        A json file containing a list of objects with the key 'data' is produced. 
        This json file can be provided to label studio to create a task. 
        The source images must have been uploaded previously to the label studio VM.
    """
    
    print(f"Reading from {input_file}")
    input_file = Path(input_file)
    df = pd.read_csv(input_file, index_col=0)
    df = df.reset_index().rename(columns={'index':'id'})
    
    first_image = df['indv_name'].first_valid_index()
    sample_image = df['indv_name'].loc[first_image]
    file_type = Path(sample_image).suffix # '.jpg' or '.png'
    df['detection_classes'] = df['detection_classes'].astype(int).replace({0:'nest', 1:'cormorant'})

    df['pano_name'] = df['image'].map(lambda p: Path(p).parent.name).astype(str)
    # df["compare_with"] = df[compare_with] if isinstance(compare_with, str) else compare_with
    df["compare_with"] = compare_with

    image1 = URL_PREFIX + df['compare_with'] + file_type
    image2 = URL_PREFIX + df['pano_name'] + '/detection_tiles/' + df['id'].astype(str) + file_type
    # images = [image1, image2, df['detection_classes']]
    # image_labels = ["image1", "image2", 'detection_classes']
    df['image1'] = image1
    df['image2'] = image2

    for col in ['beam_id']: # if the mandatory column doesn't exist, create an empty column
        if col not in df.columns:
            df[col] = ''
    df = df.rename(columns={'id':'nest_id'})

    ## for elem in zip(image1, image2):
    #    # print(dict(zip(["image1", "image2"], elem)))
    # data = [dict(zip(image_labels, elem)) for elem in zip(*images)]
    # df['data'] = data

    # Shape to list of json shape
    COLS = df.columns.to_list()
    COLS = [c for c in COLS if c not in SORTED_COLUMNS]
    df = df[SORTED_COLUMNS + COLS]

    data = df.head(2).to_dict(orient='records')
    data = [ {'data': d} for d in data ]
    print(json.dumps(data, indent=4))

    # Save to file
    
    print(f"Save file to: {output_file}")
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compare nests between two days. Parse a detections csv into a format to load into label studio for the comparison task." 
                                     "The csv must contain columns for 'index' (int), 'indv_name' (path to detection png image), 'detection_classes' (int), 'image' (from tiff_tiles)")
    parser.add_argument('--input_file', type=str, help='CSV created by `draw_detections.py`')
    parser.add_argument('--compare_with', type=str, help='Name of panorama to compare with')
    parser.add_argument('--folder_prefix', required=True, default='samples/2023_pairs/', type=str, help='The folder on the server containing the images')
    parser.add_argument('--out_folder', type=str, default="output", help='File path to output folder.')

    args = parser.parse_args()

    input_csv_file = Path(args.input_file)
    compare_with = args.compare_with
    folder_prefix = args.folder_prefix
    out_folder = (Path(args.out_folder))#/ input_csv_file.name).with_suffix('.json')
    
    out_name = input_csv_file.stem + "_" + "label_studio"
    output_file = (out_folder/out_name).with_suffix(".json")

    i = 1 # do not overwrite existing file, append a digit to make it a unique file name
    while output_file.exists():
        output_file = output_file.with_stem(out_name + "_" + str(i))
        i += 1
    output_file.parent.mkdir(parents=True, exist_ok=True)

    assert not Path(folder_prefix).is_absolute(), "The folder in label studio cannot be an absolute path"
    assert ".." not in Path(folder_prefix).parents, "The folder in label studio cannot contain relative (..) paths"
    
    folder_prefix = folder_prefix.rstrip("/") + "/"

    # files from Label Studio local storage are served with the url '/data/local-files/?d=samples/2023_pairs/SNB_20210705/detection_tiles/145.jpg'
    # URL_PREFIX='/data/local-files/?d=samples/2023_pairs/'
    URL_PREFIX = f'/data/local-files/?d={folder_prefix}'
    print(f"The URL_PREFIX={URL_PREFIX}")

    SORTED_COLUMNS = ["nest_id", "beam_id", "pano_name", "compare_with", "indv_name", "image1", "image2", "image",]
    
    main(input_csv_file, compare_with, output_file)
