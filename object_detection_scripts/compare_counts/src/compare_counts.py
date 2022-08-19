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

label_map = {0.0: 'Birds',
             1.0: 'Nests',
             'Birds': 0.1,
             'Nests': 0.0}


def rmse_plots(detections_dir, file_map_path, out_path, true_counts_csv=None, threshold=None):
    if not threshold:
        threshold = list(np.arange(0, 1, 0.05, ))

    with open(Path(file_map_path)) as f:
        file_map = json.load(f)

    rmse_df = pd.DataFrame()
    for thresh in threshold:
        # True Counts
        true_counts = pd.read_csv(true_counts_csv, header=1).set_index(
            'date').transpose().rename_axis(
            'date').reset_index()
        # true_counts.columns = ['Date', 'Nests', 'AdultBirds', 'Chicks', 'Birds']
        true_counts.rename(columns={'date': 'Date'}, inplace=True)
        true_counts = true_counts[['Date', 'Nests', 'Birds']]
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
                bird_counts.append(det_df['detection_classes'].value_counts()[0.0])
                nest_counts.append(det_df['detection_classes'].value_counts()[1.0])

            det_df = pd.DataFrame({'Date': dates,
                                   'Nests': nest_counts,
                                   'Birds': bird_counts})
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
                                ignore_index=True)

    sns.relplot(data=rmse_df, x='ConfidenceThreshold', y='RMSE', hue='Variable',
                col='DetectionMethod',  col_wrap=1, kind='line')
    plt.xlabel("Confidence Score Threshold")
    plt.ylabel("RMSE")
    plt.ylim(0, )
    sns.despine()
    plt.savefig(out_path)


def raw_count_plots(detections_dir, file_map_path, out_path, true_counts_csv=None, threshold=0.5):
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
        true_counts = pd.read_csv(true_counts_csv, header=1).set_index(
            'date').transpose().rename_axis(
            'date').reset_index()
        true_counts = true_counts[['date', 'Nests', 'Birds']]
        true_counts.columns = ['Date', f"Nests @ {threshold[1]}", f"Birds @ {threshold[0]}"]
        counts = true_counts.melt(id_vars='Date', var_name='variable', value_name='Manual')
        # original_columns = counts.columns

    else:
        dates_df = pd.DataFrame({'Date': list(list(file_map.values())[0].keys())})
        var_df = pd.DataFrame({'variable': [f"Nests @ {threshold[1]}", f"Birds @ {threshold[0]}"]})
        counts = pd.merge(dates_df, var_df, how='cross')
        # original_columns = counts.columns

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
                print("Found no birds")
                bird_counts.append(0)
            try:
                nest_counts.append(det_df['detection_classes'].value_counts()[1.0])
            except KeyError:
                print("Found no nests")
                nest_counts.append(0)
        print(len(dates), len(nest_counts), len(bird_counts))
        det_df = pd.DataFrame({'Date': dates,
                                f'Nests @ {threshold[1]}': nest_counts,
                                f'Birds @ {threshold[0]}': bird_counts})
        melted = det_df.melt(id_vars='Date', value_name=detection_set)
        counts = counts.merge(melted, on=['Date', 'variable'], how='inner')

    data = counts.melt(id_vars=['Date', 'variable'], var_name='method')

    # Plot
    if len(file_map) == 1:
        g = sns.catplot(data=data, x='Date', y='value', col='variable',
                        kind='bar', col_wrap=1, sharex=False,
                        hue='method',
                        palette=[
                            # '#fc8d59',
                            '#abd9e9'
                        ],
                        legend_out=False,
                        )
    else:
        g = sns.catplot(data=data, x='Date', y='value',
                        hue='method',
                        col='variable',
                        kind='bar',
                        palette=[
                            '#fc8d59',
                            "#2c7bb6",
                            "#abd9e9",
                        ],
                        col_wrap=1, sharex=False,
                        legend_out=False,
                        )
    (g.set_axis_labels("", "Raw Counts")
      .set_titles("{col_name}")
      .despine())
    plt.xticks(rotation=0)
    # plt.legend(loc='upper right')
    # sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.8), title='Count Method', orient='h')
    # g.fig.suptitle('ONE TITLE FOR ALL')
    sns.despine()
    plt.gcf().set_size_inches(15, 10)
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

    args = parser.parse_args()

    function_map[args.plot_type](args.detections_dir,
                                 args.file_map,
                                 args.out_path,
                                 args.true_counts,
                                 args.threshold)





