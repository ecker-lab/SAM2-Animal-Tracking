"""
https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun
"""
import os
import numpy as np
import json
import cv2


# DATA_PATH = 'data/public/dancetrack'
# OUT_PATH = os.path.join(DATA_PATH, 'annotations')
# SPLITS = ['train', 'val', 'test']
# img_folder = 'img1'
# img_name_digits = 8
img_id_offset = 1

DATA_PATH = 'data/public/ChimpACT_processed'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['test/images']
img_folder = ''
img_name_digits = 6
img_id_offset = 0

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:

        data_path = os.path.join(DATA_PATH, split)

        seqs = os.listdir(data_path)

        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq or '.ipy' in seq:
                continue
            ann = {}
            image_info = {}
            out = {'images': [], 'annotations': [], 'videos': [],
                   'categories': [{'id': 1, 'name': 'dancer'}]}

            image_cnt = 0
            ann_cnt = 0
            out_path = os.path.join(
                DATA_PATH, split, seq, 'gt', 'gt_det.json')

            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)

            # for chimpact '' for dancetrack 'img1'
            img_path = os.path.join(seq_path, img_folder)
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            images = os.listdir(img_path)
            # half and half
            num_images = len([image for image in images if 'jpg' in image])

            for i in range(num_images):
                img = cv2.imread(os.path.join(
                    data_path, os.path.join(seq, img_folder, str(i + img_id_offset).zfill(img_name_digits) + '.jpg')))

                height, width = img.shape[:2]
                image_info = {'file_name': str(i + img_id_offset).zfill(img_name_digits) + '.jpg',  # image name.
                              # image number in the entire training set.
                              'id': image_cnt + i + 1,
                              # image number in the video sequence, starting from 1.
                              'frame_id': i + 1,
                              # image number in the entire training set.
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height,
                              'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))

            if split != 'test':
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    category_id = 1
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                    out['annotations'].append(ann)
                print('{}: {} ann images'.format(seq, int(anns[:, 0].max())))

            image_cnt += num_images
            print('loaded {} for {} images and {} samples'.format(
                split, len(out['images']), len(out['annotations'])))
            # json.dump(out, open(out_path, 'w+'))
            with open(out_path, 'w+') as f:
                json_header = {
                    'images': out['images'],
                    'videos': out['videos'],
                    'categories': out['categories'],
                    'annotations': []  # start empty, write manually below
                }
                f.write('{\n')
                f.write('"images": ' +
                        json.dumps(out['images'], indent=2) + ',\n')
                f.write('"videos": ' +
                        json.dumps(out['videos'], indent=2) + ',\n')
                f.write('"categories": ' +
                        json.dumps(out['categories'], indent=2) + ',\n')
                f.write('"annotations": [\n')

                for idx, ann in enumerate(out['annotations']):
                    f.write(json.dumps(ann, indent=2))
                    if idx < len(out['annotations']) - 1:
                        f.write(',\n\n')  # add comma and blank line
                    else:
                        f.write('\n')  # no comma after last annotation

                f.write(']\n}\n')
