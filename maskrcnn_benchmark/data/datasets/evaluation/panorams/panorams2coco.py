# COCO style evaluation for custom datasets derived from AbstractDataset
# Warning! area is computed using binary maps, therefore results may differ
# because of the precomputed COCO areas
# by botcs@github
import os
from functools import partial

import numpy as np
import torch
import pycocotools.mask as mask_util

from maskrcnn_benchmark.data.datasets.panorams import PanorAMSDataset
from maskrcnn_benchmark.structures.bounding_box import BoxList

import logging
from datetime import datetime
from tqdm import tqdm
import json


def do_conversion_coco_format(dataset, output_folder, dataset_name=None):
    logger = logging.getLogger("maskrcnn_benchmark.coco_conversion")
    logger.info("Converting annotations to COCO format...")
    coco_annotation_dict = convert_panorams_to_coco(dataset)

    if dataset_name is None:
        dataset_name = dataset.__class__.__name__ 
        ann_filename = "ann_" + dataset_name + "_cocostyle" + ".json"
    coco_annotation_path = os.path.join(output_folder, ann_filename)
    logger.info("Saving annotations to %s" % coco_annotation_path)
    with open(coco_annotation_path, "w") as f:
        json.dump(coco_annotation_dict, f, indent=2)

def convert_panorams_to_coco(dataset, num_workers=None, chunksize=100):
    """
    Convert any dataset derived from AbstractDataset to COCO style
    for evaluating with the pycocotools lib
    Conversion imitates required fields of COCO instance segmentation
    ground truth files like: ".../annotations/instances_train2014.json"
    After th conversion is done a dict is returned that follows the same
    format as COCO json files.
    By default .coco_eval_wrapper.py saves it to the hard-drive in json format
    and loads it with the maskrcnn_benchmark's default COCODataset
    Args:
        dataset: any dataset derived from AbstractDataset
        num_workers (optional): number of worker threads to parallelize the
            conversion (default is to use all cores for conversion)
        chunk_size (optional): how many entries one thread processes before
            requesting new task. The larger the less overhead there is.
    """

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    #assert isinstance(dataset, PanorAMSDataset)
    # Official COCO annotations have these fields
    # 'info', 'licenses', 'images', 'type', 'annotations', 'categories'
    coco_dict = {}
    coco_dict["info"] = {
        "year":         2020,
        "version":      1.0,
        "description":  "PanorAMS dataset",
        "contributor":  "UvA",
        "date_created": "%s" % datetime.now(),
    }
    coco_dict["type"] = "instances"

    images = []
    annotations = []

    if num_workers is None:
        num_workers = torch.multiprocessing.cpu_count()
    else:
        num_workers = min(num_workers, torch.multiprocessing.cpu_count())

    dataset_name = dataset.__class__.__name__
    num_images = len(dataset)
    logger.info(
        (
            "Parsing each entry in "
            "%s, total=%d. "
            "Using N=%d workers and chunksize=%d"
        )
        % (dataset_name, num_images, num_workers, chunksize)
    )

    indices = dataset.get_indices()

    with torch.multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=num_images) as progress_bar:
            #args = [(dataset, idx, dataset.get_pano_id(idx)) for idx in indices]
            for img_annots_pair in pool.imap_unordered(partial(process_single_image, dataset=dataset), indices, chunksize=chunksize):
            #for img_annots_pair in iterator:
                image, per_img_annotations = img_annots_pair

                images.append(image)
                annotations.extend(per_img_annotations)
                progress_bar.update(1)

    for ann_id, ann in enumerate(annotations, 1):
        ann["id"] = ann_id

    logger.info("Parsing categories:")
    # CATEGORY DATA
    categories = [
        {"id":  1, "name":  "hoogspanningsmast"},
        {"id":  2, "name":  "tram"},
        {"id":  3, "name":  "train"},
        {"id":  4, "name":  "public_transport_stop"},
        {"id":  5, "name":  "reclamezuil"},
        {"id":  6, "name":  "metro"},
        {"id":  7, "name":  "sport"},
        {"id":  8, "name":  "railway_road_crossings"},
        {"id":  9, "name":  "bus"},
        {"id":  10, "name": "tracks"},
        {"id":  11, "name": "traffic_lights"},
        {"id":  12, "name": "ferry"},
        {"id":  13, "name": "bridge"},
        {"id":  14, "name": "bicycle_route"},
        {"id":  15, "name": "toilets"},
        {"id":  16, "name": "parks"},
        {"id":  17, "name": "building"},
        {"id":  18, "name": "lichtmast"},
        {"id":  19, "name": "waterway"},
        {"id":  20, "name": "trees"},
        {"id":  21, "name": "windturbine"},
        {"id":  22, "name": "trash"},
        {"id":  23, "name": "playground"},
        {"id":  24, "name": "traffic_signs"}
    ]
    # Logging categories
    for cat in categories:
        logger.info(str(cat))

    coco_dict["images"] = images
    coco_dict["annotations"] = annotations
    coco_dict["categories"] = categories
    return coco_dict

def process_single_image(idx, dataset):
    pano_id = dataset.get_pano_id(idx)
    #dataset, idx, pano_id = args
    # IMAGE DATA
    image = {}
    # Official COCO "images" entries have these fields
    # 'license', 'url', 'file_name', 'height', 'width', 'date_captured', 'id'

    img, target, ret_idx = dataset.__getitem__(idx)
    # print(ret_idx)
    # print(idx)

    img_info = dataset.get_img_info(idx)
    assert isinstance(img_info, dict)
    image.update(img_info)
    image["width"], image["height"] = target.size
    image["pano_id"] = pano_id

    if "id" not in image.keys():
        # Start indexing from 1 if "id" field is not present
        image["id"] = idx
    else:
        idx = image["id"]

    # ANNOTATION DATA
    per_img_annotations = []
    # Official COCO "annotations" entries have these fields
    # 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'categoary_id', 'id'

    assert ret_idx == idx, (ret_idx, idx)
    assert isinstance(target, BoxList)

    bboxes = target.convert("xywh").bbox.tolist()
    areas = target.area().tolist()

    cat_ids = target.get_field("labels").long().tolist()
    assert len(bboxes) == len(areas) == len(cat_ids)
    num_instances = len(target)
    for ann_idx in range(num_instances):
        annotation = {}
        annotation["segmentation"] = []
        annotation["area"] = areas[ann_idx]
        annotation["iscrowd"] = 0
        annotation["image_id"] = idx
        annotation["pano_id"] = pano_id
        annotation["bbox"] = bboxes[ann_idx]
        annotation["category_id"] = cat_ids[ann_idx]
        per_img_annotations.append(annotation)

    return image, per_img_annotations