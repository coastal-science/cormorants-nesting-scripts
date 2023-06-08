"""
e.g. RMSE
python3 compare_counts.py \
  --true_counts ../input/SNB_2020/VALIDATION/Counts_Revised.csv \
  --detections_dir ../input/SNB_2020/VALIDATION/detections \
  --file_map ../input/SNB_2020/VALIDATION/file_map.json \
  --out_path ../output/SNB_2020/VALIDATION/revised/2020_val_revised_rmse.png \
  --plot_type rmse

e.g. Count Comparisons
python3 compare_counts.py \
  --true_counts ../input/SNB_2020/VALIDATION/Counts_Revised.csv  \
  --detections_dir ../input/SNB_2020/VALIDATION/detections \
  --file_map ../input/SNB_2020/VALIDATION/file_map.json \
  --out_path ../output/SNB_2020/VALIDATION/revised/2020_val_revised_b0.2_n0.4.png \
  --plot_type counts \
  --threshold "{0.0: 0.2, 1.0:0.4}"
"""
import argparse

import pandas as pd
import seaborn as sns
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from ast import literal_eval

label_map = {0.0: 'Cormorant',
             1.0: 'Nest',
             'Cormorant': 0.1,
             'Nest': 0.0}

palette_map = {
    'Manual': '#fc8d59',                                # orange
    'Model': "#2c7bb6",                                 # darkest blue
    'Model + PP1': "#8fc0e3",                           # med-light blue
    'Model + PP2': "#abd9e9",                           # light blue
    'Model + Masking + Nest Deduplication': "#abd9e9",  # light blue
}


def rmse_plots(detections_dir, file_map_path, out_path, save_csv=False, true_counts_csv=None,
               threshold=None):
    if not threshold:
        threshold = list(np.arange(0, 1, 0.05, ))

    with open(Path(file_map_path)) as f:
        file_map = json.load(f)

    rmse_df = pd.DataFrame()
    for thresh in threshold:
        true_counts = pd.read_csv(true_counts_csv)
        counts = true_counts.melt(id_vars='Date', var_name='variable', value_name='Manual')

        original_columns = counts.columns
        # Detections
        for detection_set in file_map:
            dates = []
            nest_counts = []
            bird_counts = []
            for date, detection_file in file_map[detection_set].items():
                det_df = pd.read_csv(Path(detections_dir).joinpath(Path(detection_file)))
                det_df = det_df[det_df['detection_scores'] >= thresh]
                dates.append(date)
                try:
                    bird_counts.append(det_df['detection_classes'].value_counts()[0.0])
                except KeyError:
                    bird_counts.append(0)
                try:
                    nest_counts.append(det_df['detection_classes'].value_counts()[1.0])
                except KeyError:
                    nest_counts.append(0)

            det_df = pd.DataFrame({'Date': dates,
                                   'Nest': nest_counts,
                                   'Cormorant': bird_counts})
            det_df = det_df.melt(id_vars='Date', value_name=detection_set)
            counts = counts.merge(det_df, on=['Date', 'variable'], how='inner')

            # RMSE
            variables = []
            rmses = []
            detection_methods = []
            for g, gdf in counts.groupby('variable'):
                for c in gdf.columns.difference(original_columns):
                    rmse = mean_squared_error(y_true=gdf['Manual'], y_pred=gdf[c], squared=False)
                    variables.append(g)
                    rmses.append(rmse)
                    detection_methods.append(c)

            rmse_df = pd.concat([rmse_df,
                                 pd.DataFrame({'DetectionMethod': detection_methods,
                                               'Variable': variables,
                                               'ConfidenceThreshold': thresh,
                                               'RMSE': rmses
                                               })],
                                ignore_index=True).drop_duplicates()

    sns.relplot(data=rmse_df, x='ConfidenceThreshold', y='RMSE', hue='Variable',
                col='DetectionMethod', col_wrap=1, kind='line')
    plt.xlabel("Confidence Score Threshold")
    plt.ylabel("RMSE")
    plt.ylim(0, )
    sns.despine()
    plt.savefig(out_path)

    if save_csv:
        rmse_df.to_csv(Path(out_path).parent.joinpath("rmse.csv"), index=False)


def raw_count_plots(detections_dir, file_map_path, out_path, save_csv=False, true_counts_csv=None,
                    threshold=0.5):
    # Threshold
    threshold = literal_eval(threshold)
    if isinstance(threshold, float):
        threshold = {0.0: threshold,
                     1.0: threshold}

    # File Map
    with open(Path(file_map_path)) as f:
        file_map = json.load(f)

    # True Counts
    if true_counts_csv is not None:
        true_counts = pd.read_csv(true_counts_csv)
        true_counts = true_counts[['Date', 'Nest', 'Cormorant']]
        true_counts.columns = ['Date', f"Nest @ {threshold[1]}", f"Cormorant @ {threshold[0]}"]
        counts = true_counts.melt(id_vars='Date', var_name='variable', value_name='Manual')

    else:
        dates_df = pd.DataFrame({'Date': list(list(file_map.values())[0].keys())})
        var_df = pd.DataFrame(
            {'variable': [f"Nest @ {threshold[1]}", f"Cormorant @ {threshold[0]}"]})
        counts = pd.merge(dates_df, var_df, how='cross')

    for detection_set in file_map:
        dates = []
        nest_counts = []
        bird_counts = []
        for date, detection_file in file_map[detection_set].items():
            det_df = pd.read_csv(Path(detections_dir).joinpath(Path(detection_file)))
            det_df = det_df[~det_df['detection_scores'].isna()]
            threshold_bool = []
            for label, score in zip(det_df['detection_classes'], det_df['detection_scores']):
                if score >= threshold[label]:
                    threshold_bool.append(True)
                else:
                    threshold_bool.append(False)
            det_df = det_df[threshold_bool]
            dates.append(date)
            try:
                bird_counts.append(det_df['detection_classes'].value_counts()[0.0])
            except KeyError:
                print("Found no Cormorant")
                bird_counts.append(0)
            try:
                nest_counts.append(det_df['detection_classes'].value_counts()[1.0])
            except KeyError:
                print("Found no Nest")
                nest_counts.append(0)

        det_df = pd.DataFrame({'Date': dates,
                               f'Nest @ {threshold[1]}': nest_counts,
                               f'Cormorant @ {threshold[0]}': bird_counts})
        melted = det_df.melt(id_vars='Date', value_name=detection_set)
        counts = counts.merge(melted, on=['Date', 'variable'], how='inner')

    data = counts.melt(id_vars=['Date', 'variable'], var_name='method')

    # Write Data to CSV
    if save_csv:
        csv_path = Path(out_path).parent.joinpath("counts.csv")
        data.to_csv(csv_path, index=False)

    # Plot
    g = sns.catplot(data=data, x='Date', y='value', col='variable',
                    kind='bar', col_wrap=1, sharex=False,
                    hue='method',
                    palette=palette_map,
                    legend_out=False,
                    )

    (g.set_axis_labels("", "Raw Counts")
     .set_titles("{col_name}")
     .despine())
    plt.xticks(rotation=0)
    sns.despine()
    plt.gcf().set_size_inches(25, 10)
    plt.savefig(out_path, dpi=600)


if __name__ == '__main__':
    function_map = {
        'rmse': rmse_plots,
        'counts': raw_count_plots
    }

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--true_counts', help='.')
    parser.add_argument('--detections_dir', help='.')
    parser.add_argument('--file_map', help='.')
    parser.add_argument('--out_path', help='.')
    parser.add_argument('--plot_type')
    parser.add_argument('--threshold', help='.')
    parser.add_argument('--save_raw_csv', help='.', action='store_true')

    args = parser.parse_args()

    # Make sure the output directory exists
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    # Create the graphs
    function_map[args.plot_type](args.detections_dir,
                                 args.file_map,
                                 args.out_path,
                                 args.save_raw_csv,
                                 args.true_counts,
                                 args.threshold)
