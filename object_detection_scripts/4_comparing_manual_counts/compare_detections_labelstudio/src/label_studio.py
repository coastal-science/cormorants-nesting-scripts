import argparse
import pandas as pd
from pathlib import Path
import json
import sys
PYTHON_VERSION = tuple(map(int, sys.version.split()[0].split(".")))

def main(image_name:str, input_file:str, right, output_file, swap:bool):
    """Reshape the input csv into a shape that can be used for a Label Studio labeling task 
    to compare individual detections from one day against the next day's panorama, the status of nest

    Args:
        input_file (Path): input csv (generated from draw_final_detections.py)
                        csv must contain minimum columns for 
                        'index' (int), 
                        'indv_name' (path to detection png image), 
                        'detection_classes' (int), 
                        'image' (str from tiff_tiles)
        right (str): Name of panorama for day 2
        swap (bool): Swap references left, right references. Default is False

        A json file containing a list of objects with the key 'data' is produced. 
        This json file can be provided to label studio to create a task. 
        The source images must have been uploaded previously to the label studio VM.
    """
    
    print(f"Reading from {input_file}")
    input_file = Path(input_file)
    df = pd.read_csv(input_file, index_col=0)
    df = df.reset_index().rename(columns={'index':'id'})
    df = df.dropna(subset=["indv_name"])
        
    def extract_relative(path:str)-> str:
        """ Extract individual detection file path relative to the name of its own full pano
        """
        path = Path(path)
        i = [i for i, p in enumerate(path.parents) if image_name in p.name]
        assert i, f"'{image_name}' is not in the path of indv_name={path}."
        i = i[0]
        p = path.parents[i]
        p = p.parent if p.parent != "/" else p
        path = path.relative_to(p.parent)
        
        return str(path)

    df['indv_name'] = df['indv_name'].apply(extract_relative)

    df['detection_classes'] = df['detection_classes'].astype(int).replace({0:'nest', 1:'cormorant'})

    df['pano_right'] = df['image'].map(lambda p: Path(p).parent.name).astype(str)
    
    df["pano_left"] = right

    image1 = URL_PREFIX + df['pano_left']
    image2 = URL_PREFIX + df['indv_name']# df['pano_right'] + '/detection_tiles/' + df['id'].astype(str) + file_type
    # images = [image1, image2, df['detection_classes']]
    # image_labels = ["image1", "image2", 'detection_classes']
    df['image_left'] = image1
    df['image_right'] = image2

    for col in ['beam_id']: # if the mandatory column doesn't exist, create an empty column
        if col not in df.columns:
            df[col] = ''
    df = df.rename(columns={'id':'detection_id'})

    ## for elem in zip(image1, image2):
    #    # print(dict(zip(["image1", "image2"], elem)))
    # data = [dict(zip(image_labels, elem)) for elem in zip(*images)]
    # df['data'] = data

    if swap:
        # Swap all column names *left with *right
        swap_dict = {'left':'right', 'right':'left'}
        def swap_str(name:str):
            name = name.lower()
            direction = 'left' if 'left' in name else 'right'
            return name.replace(direction, swap_dict.get(direction))
        
        lr_cols = [ c for c in df.columns if 'left' in c.lower() or 'right' in c.lower() ]
        lr_cols_renamed = list(map(swap_str, lr_cols))
        df = df.rename(columns=dict(zip(lr_cols, lr_cols_renamed)))
        # print(df.loc[0])

    # Shape to list of json
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
    parser.add_argument('--right', type=str, help='Name of panorama to compare with')
    parser.add_argument('--folder_prefix', required=True, default='samples/2023_pairs/', type=str, help='The folder on the server containing the images')
    parser.add_argument('--out_folder', type=str, default="output", help='File path to output folder.')
    if PYTHON_VERSION < (3,9):
        parser.add_argument('--swap', 
                            default=False, required=False, 
                            action='store_true', 
                            help='Swap left, right references. Default is False, which means left_image=pano, right_image=indv.')
        parser.add_argument('--no-swap', dest='swap', action='store_false')
    else:
        parser.add_argument('--swap', 
                            type=bool, 
                            default=False, required=False, 
                            action=argparse.BooleanOptionalAction, 
                            help='Swap left, right references. Default is False, which means left_image=pano, right_image=indv.')

    args = parser.parse_args()

    input_csv_file = Path(args.input_file)
    right = args.right
    folder_prefix = args.folder_prefix
    out_folder = Path(args.out_folder) #/ input_csv_file.name).with_suffix('.json')
    swap = args.swap

    out_name = input_csv_file.stem + "_" + "label_studio"
    output_file = (out_folder/out_name).with_suffix(".json")
    image_name = str(input_csv_file.stem)

    i = 1 # do not overwrite existing file, append a digit to make it a unique file name
    while output_file.exists():
        output_file = output_file.with_name(out_name + "_" + str(i)).with_suffix(".json")
        i += 1
    output_file.parent.mkdir(parents=True, exist_ok=True)

    assert not Path(folder_prefix).is_absolute(), "The folder in label studio cannot be an absolute path"
    assert ".." not in Path(folder_prefix).parents, "The folder in label studio cannot contain relative (..) paths"
    
    folder_prefix = folder_prefix.strip("/") + "/"

    # files from Label Studio local storage are served with the url '/data/local-files/?d=samples/2023_pairs/SNB_20210705/detection_tiles/145.jpg'
    # URL_PREFIX='/data/local-files/?d=samples/2023_pairs/'
    URL_PREFIX = f'/data/local-files/?d={folder_prefix}'
    print(f"The URL_PREFIX={URL_PREFIX}")

    SORTED_COLUMNS = ["detection_id", "beam_id", "pano_left", "pano_right", "indv_name", "image_left", "image_right", "image",]
    
    main(image_name, input_csv_file, right, output_file, swap)