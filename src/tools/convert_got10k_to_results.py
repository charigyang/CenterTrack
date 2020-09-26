import json
import numpy as np
import os
from collections import defaultdict
split = 'train'

DET_PATH = '../../data/got-10k/train'
ANN_PATH = '../../data/got-10k/annotations/{}.json'.format(split)
OUT_DIR = '../../data/got-10k/results/'
OUT_PATH = OUT_DIR + '{}_det.json'.format(split)

if __name__ == '__main__':
  if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
  seqs = os.listdir(DET_PATH)
  data = json.load(open(ANN_PATH, 'r'))
  images = data['images']
  anns = data['annotations']
  image_to_anns = defaultdict(list)

  results = {}
  for image_info in anns:
    image_id = image_info['image_id']
    results[image_id] = []
    dets = [image_info['bbox']]
    for det in dets:
      bbox = [float(det[0]), float(det[1]), \
              float(det[0] + det[2]), float(det[1] + det[3])]
      ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
      results[image_id].append(
        {'bbox': bbox, 'score': float(1.0), 'class': 1, 'ct': ct})
  out_path = OUT_PATH
  json.dump(results, open(out_path, 'w'))
