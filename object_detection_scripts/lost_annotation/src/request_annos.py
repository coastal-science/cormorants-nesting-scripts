from lost.pyapi import script
import os
import pandas as pd
import ast

ENVS = ['lost']
ARGUMENTS = {'polygon': {'value': 'false',
                         'help': 'Add a dummy polygon proposal as example.'},
             'line': {'value': 'false',
                      'help': 'Add a dummy line proposal as example.'},
             'point': {'value': 'false',
                       'help': 'Add a dummy point proposal as example.'},
             'bbox': {'value': 'true',
                      'help': 'Add a dummy bbox proposal as example.'}}


class ReviewAnnos(script.Script):
    """
    Request annotations for each image of an imageset.

    An imageset is basically a folder with images.
    """
    def __init__(self, proposed_annos):
        self.proposed_annotations = proposed_annos
        super(ReviewAnnos, self).__init__()

    def get_img_annos(self, img_name):
        img_annos = self.proposed_annotations[(self.proposed_annotations['img.img_path'].str.split('/').str[-1] == img_name) &
                                              (self.proposed_annotations['anno.dtype'] == 'bbox')]
        boxes = []
        labels = []

        for b, l in zip(img_annos['anno.data'], img_annos['anno.lbl.idx']):
            raw_box = ast.literal_eval(b)
            box = [raw_box['x'], raw_box['y'], raw_box['w'], raw_box['h']]
            boxes.append(box)
            labels.append(l)

        return boxes, labels

    def main(self):
        for ds in self.inp.datasources:
            media_path = ds.path
            for img_file in os.listdir(media_path):
                img_path = os.path.join(media_path, img_file)
                boxes, labels = self.get_img_annos(img_file)
                if len(boxes) == 0:
                    continue
                self.outp.request_bbox_annos(img_path=img_path,
                                             boxes=boxes,
                                             labels=labels)
                self.logger.info('Requested annos for: {}'.format(img_path))
                

if __name__ == "__main__":
    proposed_annos = pd.read_csv('/home/lost/data/prelim_annos/gab2/detections.csv')
    my_script = ReviewAnnos(proposed_annos)

