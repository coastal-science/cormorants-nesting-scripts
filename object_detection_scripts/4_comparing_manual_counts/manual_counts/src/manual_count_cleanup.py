import copy
import pandas as pd
import re
from pathlib import Path
from annotation_utils import read_label_map, BoundingBox, Annotation
import ast
from shapely.ops import unary_union
import json
from PIL import Image
from collections import Counter


def get_anno_id(img_name):
    return int(re.split("[._/]", img_name)[-2])


def get_det_id(img_name):
    return get_anno_id(img_name)


class GTAnnotations:
    def __init__(self):
        self.label_map = lambda x: None
        self.all_annotations = {}
        self.final_annotations = {}

        # self.all_detections = pd.DataFrame()
        self.false_positive_detections = {}

        self.mistaken_annotations = set()
        self.duplicate_annotations = set()
        self.missing_annotations = set()
        self.mislabelled_annotations = set()

    def load_label_map(self, label_map_file):
        basic_map = read_label_map(label_map_file)

        def label_mapper(label_to_map=None):
            if label_to_map is None:
                return basic_map
            for lbl in basic_map:
                if lbl in label_to_map:
                    return basic_map[lbl]

        self.label_map = label_mapper

    def load_lost_annotations(self, annotation_files, tile_directory):
        all_annotations = [pd.read_csv(f) for f in annotation_files]
        annotation_df = pd.concat(all_annotations).dropna(subset=['anno.data'])
        annos = annotation_df.apply(
            lambda x: Annotation(lost_row=x, label_map=self.label_map, tile_directory=tile_directory), axis=1)
        self.all_annotations.update({a.id: a for a in annos})

    def load_coco_false_positive_detections(self, false_positive_detections, tile_directory):
        false_positive_df = pd.read_csv(false_positive_detections)

        annos = false_positive_df.apply(
            lambda x: Annotation(coco_det_row=x), axis=1)

        self.false_positive_detections.update({a.id: a for a in annos})

    def load_excel_removal_record(self, excel_file):
        removal_record = pd.read_excel(excel_file)

        # Load mistaken annotations
        mistaken_annotations = removal_record[
            removal_record['Reason'].isin(['not a bird or nest', 'outside masking area'])]['Annotation ID']
        self.mistaken_annotations.update(mistaken_annotations)

        # Load mislabelled annotations
        mislabelled_annotations = removal_record[
            removal_record['Reason']=="incorrect category"]['Annotation ID']
        self.mislabelled_annotations.update(mislabelled_annotations)#

        # Load duplicate annotations
        duplicate_annotations = removal_record[
            removal_record['Reason'] == 'duplicate over tiles']
        duplicate_pairs = zip(duplicate_annotations['Canonical Annotation'],
                              duplicate_annotations['Annotation ID']
                              )
        duplicates = []
        for a, b in duplicate_pairs:
            if a < b:
                duplicates.append((a, b))
            else:
                duplicates.append((b, a))

        self.duplicate_annotations.update(set(duplicates))

    def load_label_studio(self, false_positive_file=None, false_negative_file=None):
        if false_positive_file:
            fp_df = pd.read_csv(false_positive_file)

            # Load missing annotations
            problem_detections = fp_df[~fp_df['error_type'].isna()]
            missing_annotations = problem_detections[
                problem_detections['error_type'].str.contains("Missing")]
            self.missing_annotations.update(missing_annotations['image'].apply(get_det_id))

            # Load mislabelled annotations
            dets_with_mislabelled_annos = fp_df[(fp_df['error_type'].str.contains("Incorrect")) & (~fp_df['error_type'].isna())]
            self.mislabelled_annotations.update(dets_with_mislabelled_annos['canonical_annotation3'])

        if false_negative_file:
            fn_df = pd.read_csv(false_negative_file)

            # Load mistaken annotations (annotations that were made, but don't actually exist)
            problem_annotations = fn_df[~fn_df['annotation_issue'].isna()]
            mistaken_annotations = problem_annotations[
                problem_annotations['annotation_issue'].str.contains('No Object')]
            self.mistaken_annotations.update(mistaken_annotations['image'].apply(get_anno_id))

            # Load mislabelled annotations
            mislabelled_annotations = problem_annotations[
                problem_annotations['annotation_issue'].str.contains('Wrong Category')]
            self.mislabelled_annotations.update(mislabelled_annotations['image'].apply(get_anno_id))

            # Load duplicate annotations
            duplicate_annotations = problem_annotations[problem_annotations['annotation_issue'].str.contains('Duplicate')]
            duplicate_pairs = zip(duplicate_annotations['image'].apply(get_anno_id),
                                  duplicate_annotations['canonical_annotation1'])
            duplicates = []
            for a, b in duplicate_pairs:
                if a < b:
                    duplicates.append((a, b))
                else:
                    duplicates.append((b, a))

            self.duplicate_annotations.update(duplicates)

    def clean_annotations(self):
        print(f"********* Cleaning Annotations *********")
        self.final_annotations = copy.deepcopy(self.all_annotations)

        self.remove_mistaken_annotations()
        self.correct_mislabelled_annotations()
        self.merge_duplicate_annotations()
        self.add_missing_annotations()

        print(f"***** Finished Cleaning Annotations ****")

    def add_missing_annotations(self):
        print("Adding Missing Annotations")
        for missing_anno in self.missing_annotations:
            new_anno = copy.deepcopy(self.false_positive_detections[-1 * missing_anno])
            assert(new_anno.id not in self.final_annotations)
            self.final_annotations[new_anno.id] = new_anno

        print(f"\t{len(self.missing_annotations)} missing annotations added.")

    def remove_mistaken_annotations(self):
        print("Removing Mistaken Annotations")
        for annotation_id in self.mistaken_annotations:
            self.final_annotations.pop(annotation_id)
        print(f"\t{len(self.mistaken_annotations)} mistaken annotations removed")

    def merge_duplicate_annotations(self):
        print("Merging Duplicate Annotations")
        if not self.final_annotations:
            self.final_annotations = copy.deepcopy(self.all_annotations)
        self.duplicate_annotations = sorted(self.duplicate_annotations, key=lambda x: x[0])

        merge_list = {}
        can_map = {}
        for a, b in self.duplicate_annotations:
            if (a in merge_list) and (b in merge_list):
                # Both - merge these together, add b -> a mapping
                merge_list[a] += merge_list[b]
                merge_list.pop(b)
                can_map[b] = a

            elif (a in merge_list) and (b not in merge_list):
                # Only A - add B to A's list, add b -> mapping
                merge_list[a].append(self.all_annotations[b])
                can_map[b] = a

            elif (a not in merge_list) and (b in merge_list):
                # Only B - add A to B's list, add a -> mapping
                merge_list[b].append(self.all_annotations[a])
                can_map[a] = b

            elif (a not in merge_list) and (b not in merge_list):
                # Neither - check if either are mapped.
                if a in can_map:
                    # add B to can_map[a]'s list
                    merge_list[can_map[a]].append(self.all_annotations[b])
                elif b in can_map:
                    # add A to can_map[b]'s list
                    merge_list[can_map[b]].append(self.all_annotations[a])
                else:
                    # create A in merge list, add B to A's list, add b->a mapping
                    merge_list[a] = [self.all_annotations[b]]
                    can_map[b] = a

        for canonical_annotation_id in merge_list:
            canonical_annotation = self.final_annotations[canonical_annotation_id]
            for annotation in merge_list[canonical_annotation_id]:
                canonical_annotation.merge(annotation)
                self.final_annotations.pop(annotation.id)
            self.final_annotations[canonical_annotation_id] = canonical_annotation

        all_duplicate_annos = {item for duplicate_pair in self.duplicate_annotations for item in duplicate_pair}
        print(f"\t{len(all_duplicate_annos)} duplicated annotations merged into {len(merge_list)} canonical annotations")

    def correct_mislabelled_annotations(self):
        print("Correcting Mislabelled Annotations")
        for annotation_id in self.mislabelled_annotations:
            current_label = self.final_annotations[annotation_id].category_id
            self.final_annotations[annotation_id].category_id = (current_label + 1) % 2
        print(f"\t{len(self.mislabelled_annotations)} mislabelled annotations corrected")

    def to_coco_json(self, out_file=None):


        coco = {
            "info": {},
            "license": [],
            "images": [{"id": img_id} for img_id in {a.image_id for a in self.final_annotations.values()}],
            "annotations": [v.coco_anno_dict() for k, v in self.final_annotations.items()],
            "categories": [{"id": v, "name": k} for k, v in self.label_map().items()]
        }

        if out_file:
            with open(Path(out_file).expanduser(), "w") as f:
                json.dump(coco, f)

        return coco

    def generate_confusion_matrix(self, per_class=False):
        # True positives are those original detections which are kept in the final list of detections
        true_positives = []
        for a, a_anno in self.final_annotations.items():
            if a not in self.mislabelled_annotations:
                if -1 * a not in self.missing_annotations:
                   true_positives.append(a_anno.category_id)

        if per_class:
            true_positives = Counter(true_positives)
            false_positives = (Counter([self.all_annotations[a].category_id for a in self.mistaken_annotations]) +
                               Counter([self.all_annotations[a].category_id for a in self.mislabelled_annotations]))
            false_negatives = Counter([self.false_positive_detections[-1 * a].category_id for a in self.missing_annotations])
            for category in true_positives:
                precision = true_positives[category] / (true_positives[category] + false_positives[category])
                recall = true_positives[category] / (true_positives[category] + false_negatives[category])
                f1_score = (2 * precision * recall) / (precision + recall)
                print(f"==== {category} ====")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1_score:.2f}")

        else:
            true_positives = len(true_positives)
            false_positives = len(self.mistaken_annotations) + len(self.mislabelled_annotations)
            false_negatives = len(self.missing_annotations)

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = (2 * precision * recall) / (precision + recall)
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1_score:.2f}")


if __name__ == '__main__':
    # Input Files
    common_directory = Path('/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/')
    lost_annotation_dir = common_directory.joinpath('4_comparing_manual_counts/manual_counts/input/2020/VALIDATION/manual_annotations/')
    # detections_dir = common_directory.joinpath('3_prediction_pipeline_postprocessing/post_process_detections/output/2020/VALIDATION/snb5_cn_hg_v9')
    label_studio_false_positive_file = common_directory.joinpath('4_comparing_manual_counts/foo/input/project-14-at-2023-08-23-16-46-c96814e9.csv')
    excel_file = common_directory.joinpath("4_comparing_manual_counts/manual_counts/input/2020/VALIDATION/RemovalRecord_CanonicalAnnotation.xlsx")
    label_studio_false_negative_file = common_directory.joinpath('4_comparing_manual_counts/manual_counts/input/2020/VALIDATION/project-16-at-2023-09-11-21-45-74e996e0.csv')
    tile_directory = Path('/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/output/2020_SNB/MANUAL_COUNTS/')
    label_map_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/2_training_pipeline/lost_to_tf/input/snb5/label_map.pbtxt'
    false_positive_detections = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/draw_errors/output/2020/VALIDATION/test/false_positives.csv'
    coco_json_output = common_directory.joinpath('4_comparing_manual_counts/manual_counts/output/2020/VALIDATION/verified_annotations_coco.json')

    # Clean the Annotations
    gta = GTAnnotations()
    gta.load_label_map(label_map_path)
    gta.load_lost_annotations(Path(lost_annotation_dir).glob("*_annos.csv"), tile_directory=tile_directory)
    # gta.load_raw_detections(detection_files=detections_dir.glob('*.csv'))
    gta.load_coco_false_positive_detections(false_positive_detections=false_positive_detections, tile_directory=tile_directory)
    gta.load_label_studio(false_negative_file=label_studio_false_negative_file,
                          false_positive_file=label_studio_false_positive_file)
    gta.load_excel_removal_record(excel_file)

    gta.clean_annotations()

    gta.generate_confusion_matrix(per_class=True)

    # Write the COCO file
    gta.to_coco_json(coco_json_output)



