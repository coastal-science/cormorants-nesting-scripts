import pickle as pkl
import numpy as np
from pathlib import Path

# pkl_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/input/test_eval.pkl'
# pkl_file = Path("~/Downloads/2020_val_eval_0.2.pkl").expanduser()
# pkl_file = Path("~/Downloads/2020_val_eval.pkl").expanduser()
pkl_file = Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/calc_mAP/output/2020_val_eval_TEST.pkl").expanduser()
pkl_file = Path("/Users/jilliana/Downloads/coco_eval_iou0.5.csv")

# # IoU 0.1
# pkl_file = Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/calc_mAP/output/2020_val_eval_iou10.pkl").expanduser()
# pkl_file = Path("/Users/jilliana/Downloads/coco_eval_iou0.1.csv")
with open(pkl_file, 'rb') as f:
    d = pkl.load(f).eval

precision_array = d['precision']
iouThr = 0.5
# t = np.where(iouThr == np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]))[0]
# t = np.where(iouThr == np.array([0.1]))[0]
t = np.where(iouThr == np.array([0.5]))[0]
precision_50 = precision_array[t]
precision_50 = precision_50[:, :, :, 0, 2]

rec = []
pre_c = []
pre_n = []
recThr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
for i, r in enumerate(recThr):
    rec.append(r)
    pre_c.append(precision_50[0][i][0])
    pre_n.append(precision_50[0][i][1])


import pandas as pd
df = pd.DataFrame({'recall': rec,
                   'precision_cormorants': [float(p) for p in pre_c],
                   'precision_nests': [float(p) for p in pre_n]})

# F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
df['f1_score_cormorants'] = 2 * (df['precision_cormorants'] * df['recall']) / (df['precision_cormorants'] + df['recall'])
df['f1_score_nests'] = 2 * (df['precision_nests'] * df['recall']) / (df['precision_nests'] + df['recall'])

df.to_csv(Path("~/Downloads/2020-val-precision-recall-iou50.csv", index=False))
# df.to_csv(Path("~/Downloads/2020-val-precision-recall.csv", index=False))
# df.to_csv(Path("~/Downloads/2020-val-precision-recall-0.2.csv", index=False))
# df.to_csv("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/output/snb5-test-precision-recall.csv", index=False)


top_f1_nests_idx = np.argmax(df['f1_score_nests'])
top_f1_point_nests = df['recall'].loc[top_f1_nests_idx], df['precision_nests'].loc[top_f1_nests_idx]
top_f1_corms_idx = np.argmax(df['f1_score_cormorants'])
top_f1_point_corms = df['recall'].loc[top_f1_corms_idx], df['precision_cormorants'].loc[top_f1_corms_idx]

# Plot
import matplotlib.pyplot as plt
plt.plot(df['recall'], df['precision_cormorants'], color='LightCoral', label='Cormorants')
plt.scatter(*top_f1_point_corms, s=150, facecolors='none', edgecolors='LightCoral', linewidths=2)

plt.plot(df['recall'], df['precision_nests'], color='MediumSeaGreen', label='Nests')
plt.scatter(*top_f1_point_nests, s=150, facecolors='none', edgecolors='MediumSeaGreen', linewidths=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision Recall Curve w/ IoU>0.5")
plt.legend()
plt.show()
# plt.savefig("/Users/jilliana/Downloads/snb-2020-val-pr-curve-iou10.png", dpi=300)
plt.savefig("/Users/jilliana/Downloads/snb-2020-val-pr-curve-iou50.png", dpi=300)
# plt.savefig("/Users/jilliana/Downloads/snb-2020-val-pr-curve-0.2.png", dpi=300)
# plt.savefig("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/output/snb5-test-pr-curve.png", dpi=300)

###########
# RECALL
###########
recall_array = d['recall']
recall_array.shape
recall_10_corm = np.mean(recall_array[:,0,0,2])
recall_10_nest = np.mean(recall_array[:,1,0,2])


###########
# P&R
###########
pkl_files = [
    # '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/input/manuscript_results/cn_hg_v9_test.pkl',
    # '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/input/manuscript_results/cn_hg_v9_val.pkl',
    # '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/input/manuscript_results/cn_rn_w4_val.pkl',
    # '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/input/manuscript_results/rn_w6e_15k_val.pkl',
    '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/input/manuscript_results/rn_w6e_13.5k_val.pkl',
    # '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/pr_curves/input/manuscript_results/rn_w6e_val.pkl'
]
for pkl_file in pkl_files:
    print(f"***** {Path(pkl_file).name} *****")
    with open(pkl_file, 'rb') as f:
        d = pkl.load(f)

    precision_array = d['precision']
    iouThr = 0.5
    t = np.where(iouThr == np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]))[0]
    precision_50 = precision_array[t,:,:,0,2]
    precision_50_all = precision_50[0, :, :]
    precision_50_corm = precision_50[0, :, 0]
    precision_50_nest = precision_50[0, :, 1]


    recall_array = d['recall']
    np.mean(recall_array[:,:,0,1])
    recall_10_corm = np.mean(recall_array[:, 0, 0, 1])
    recall_10_nest = np.mean(recall_array[:, 1, 0, 1])

    print(f"P\tCormorants: {round(np.mean(precision_50_corm)*100, 1)}")
    print(f"R\tCormorants: {round(recall_10_corm*100, 1)}")
    print(f"P\tNests: {round(np.mean(precision_50_nest)*100, 1)}")
    print(f"R\tNests: {round(recall_10_nest*100, 1)}")
    print(f"P\tAll Classes: {round(np.mean(precision_50_all)*100, 1)}")
    print(f"R\tAll Classes: {round(np.mean(recall_array[:,:,0,1])*100, 1)}")


