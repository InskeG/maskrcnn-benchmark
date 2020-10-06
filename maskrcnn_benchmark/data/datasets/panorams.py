from maskrcnn_benchmark.structures.bounding_box import BoxList
import os, sys

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import h5py

from skimage import io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def padding1d_circular(input, pad):
    return torch.cat([input[:, :, -pad[0]:], input,
                      input[:, :, 0:pad[1]]], dim=2)

class PanorAMSDataset(object):
    CLASSES = (
        '__background__ ',
        'hoogspanningsmast',
        'tram',
        'train',
        'public_transport_stop',
        'reclamezuil', 
        'metro', 
        'sport', 
        'railway_road_crossings', 
        'bus', 
        'tracks', 
        'traffic_lights', 
        'ferry', 
        'bridge', 
        'bicycle_route', 
        'toilets', 
        'parks', 
        'building', 
        'lichtmast', 
        'waterway', 
        'trees', 
        'windturbine', 
        'trash', 
        'playground', 
        'traffic_signs'
    )

    def __init__(self, img_dir, boxes_file, ann_file, pano_ids, indices, 
        gt=False, transforms=None):
        # as you would do normally
        self.boxes_file = boxes_file
        self.ann_file = ann_file
        self.pano_ids = pano_ids
        self.img_dir = img_dir
        self.indices = indices
        self.transforms = transforms

        self.img_height = 550
        self.img_width = 1450

        self.id_to_img_map = pano_ids

        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        if gt:
            self.dataset_root_boxes = "gt_boxes"
            self.dataset_root_boxes_labels = "gt_label_boxes"
        else:
            self.dataset_root_boxes = "gis_boxes"
            self.dataset_root_boxes_labels = "label_boxes"

        print(len(pano_ids))
        print(max(indices))

    def __getitem__(self, idx):
        pano_id = self.pano_ids[idx]

        with Image.open(os.path.join(self.img_dir, (pano_id + ".jpg"))) as img:
            # img = img.crop(box=(0,0,1400,550))
            # img = transforms.ToTensor()(img)
            # img = padding1d_circular(img, pad=(25, 25))
            # img = transforms.ToPILImage()(img)
            img = np.array(img)
            img = Image.fromarray(np.uint8(img), mode='RGB')

        with h5py.File(self.boxes_file, mode="r") as dataset:
            boxes = dataset[f"{self.dataset_root_boxes}_{idx}"][...]
            labels = dataset[f'{self.dataset_root_boxes_labels}_{idx}'][...]

        labels = labels + 1

        indices = np.arange(boxes.shape[0])

        width = boxes[:, 2] - boxes[:, 0]
        filter = np.asarray(width >= 30).nonzero()
        boxes = boxes[filter]
        labels = labels[filter]
        height = boxes[:, 3] - boxes[:, 1]
        filter = np.asarray(height >= 30).nonzero()
        boxes = boxes[filter]
        labels = labels[filter]

        #adjust x coordinates for 25px circular padding of image
        boxes[:, 0] = boxes[:, 0] + 25
        boxes[:, 2] = boxes[:, 2] + 25

        #check whether there are boxes with x_min = 0
        x_min_0_indices = np.asarray(boxes[:, 0] == 0).nonzero()
        x_min_0 = boxes[x_min_0_indices]
        #if so check for linked boxes
        if x_min_0.size > 0:
            x_min_0_labels = labels[x_min_0_indices]
            for i, box in enumerate(x_min_0):
                label = x_min_0_labels[i]
                box_idx = np.where(np.all(boxes==box, axis=1))
                box_idx = box_idx[0]

                filter = np.delete(indices, box_idx)
                match_boxes = boxes[filter]
                match_labels = labels[filter]
                filter = np.asarray(match_labels == label).nonzero()
                match_boxes = match_boxes[filter]
                filter = np.asarray(match_boxes[:, 2] == 1400).nonzero()
                match_boxes = match_boxes[filter]

                if match_boxes.size > 0:
                    x_min_left_box = box[0]
                    y_min_left_box = box[1]
                    x_max_left_box = box[2]
                    y_max_left_box = box[3]

                    for match_box in match_boxes:
                        x_min_right_box = match_box[0]
                        y_min_right_box = match_box[1]
                        x_max_right_box = match_box[2]
                        y_max_right_box = match_box[3]

                        idx_right = np.where(np.all(boxes==match_box, axis=1))
                        idx_right = idx_right[0]

                        #if true then boxes are linked
                        if (y_min_right_box == y_min_left_box) & (y_max_right_box == y_max_left_box):
                            #x_max - x_min
                            width2add = x_max_right_box - x_min_right_box
                            width2add = min(width2add, 25)

                            x_min_left_box = x_min_left_box - width2add
                            boxes[box_idx, 0] = x_min_left_box

                            width2add = x_max_left_box - x_min_left_box
                            width2add = min(width2add, 25)

                            x_max_right_box = x_max_right_box + width2add
                            boxes[idx_right, 2] = x_max_right_box

                            break
                        else:
                            continue

        filter1 = np.asarray(boxes[:, 0] < 50).nonzero()
        pad_boxes_left = boxes[filter1]
        filter2 = np.asarray(pad_boxes_left[:, 0] >= 25).nonzero()
        pad_boxes_left = pad_boxes_left[filter2]

        if pad_boxes_left.size > 0:
            filtered_idx = filter1[0][filter2]
            for i, box in enumerate(pad_boxes_left):
                box_idx = filtered_idx[i]
                label = labels[box_idx]

                rightbox2add = box
                rightbox2add[0] = rightbox2add[0] + 1400
                rightbox2add[2] = min(rightbox2add[2] + 1400, 1450)

                width = rightbox2add[2] - rightbox2add[0]

                if width >= 20: 
                    boxes = np.vstack((boxes, rightbox2add))
                    labels = np.append(labels, label)

        filter1 = np.asarray(boxes[:, 2] > 1400).nonzero()
        pad_boxes_right = boxes[filter1]
        filter2 = np.asarray(pad_boxes_right[:, 2] <= 1425).nonzero()
        pad_boxes_right = pad_boxes_right[filter2]

        if pad_boxes_right.size > 0:
            filtered_idx = filter1[0][filter2]
            for i, box in enumerate(pad_boxes_right):
                box_idx = filtered_idx[i]
                label = labels[box_idx]

                leftbox2add = box
                leftbox2add[0] = max(leftbox2add[0] - 1400, 0)
                leftbox2add[2] = min(leftbox2add[2] - 1400, 25)

                width = leftbox2add[2] - leftbox2add[0]

                if width >= 20: 
                    boxes = np.vstack((boxes, leftbox2add))
                    labels = np.append(labels, label)

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # # and labels
        # labels = torch.tensor([10, 20])

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xyxy")
        # add the labels to the boxlist
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        target = target.clip_to_image(remove_empty=False)        

        # if not len(target.bbox.tolist()) == len(labels):
        #     print("1")
        #     print(len(target.bbox.tolist()), len(labels))
        if self.transforms:
            img, target = self.transforms(img, target)

        # return the image, the boxlist and the idx in your dataset
        return img, target, idx


    def __len__(self):
        return len(self.indices)
    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.img_height, "width": self.img_width}

    def get_indices(self):
        return self.indices

    def get_pano_id(self, idx):
        return self.pano_ids[idx]



class PanorAMSDatasetFromBoxesIndicesFile(object):
    CLASSES = (
        '__background__ ',
        'hoogspanningsmast',
        'tram',
        'train',
        'public_transport_stop',
        'reclamezuil', 
        'metro', 
        'sport', 
        'railway_road_crossings', 
        'bus', 
        'tracks', 
        'traffic_lights', 
        'ferry', 
        'bridge', 
        'bicycle_route', 
        'toilets', 
        'parks', 
        'building', 
        'lichtmast', 
        'waterway', 
        'trees', 
        'windturbine', 
        'trash', 
        'playground', 
        'traffic_signs'
    )

    def __init__(self, img_dir, boxes_file, ann_file, pano_ids, indices, boxes_indices_file,
        gt=False, transforms=None):
        # as you would do normally
        self.boxes_file = boxes_file
        self.boxes_indices_file = boxes_indices_file
        self.ann_file = ann_file
        self.pano_ids = pano_ids
        self.img_dir = img_dir
        self.indices = indices
        self.transforms = transforms

        self.img_height = 550
        self.img_width = 1450

        self.id_to_img_map = pano_ids

        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        if gt:
            self.labels_dataset_name = "boxes_gt"

            self.labels_dataset_name = "boxes_eval_phase2"
        else:
            self.labels_dataset_name = "boxes_noisy"

        print(len(pano_ids))
        print(max(indices))

    def __getitem__(self, idx):
        pano_id = self.pano_ids[idx]
        label_start_idx = self.boxes_indices_file.loc[self.boxes_indices_file['pano_id'] == pano_id].iloc[0]['start_idx']
        label_end_idx = self.boxes_indices_file.loc[self.boxes_indices_file['pano_id'] == pano_id].iloc[0]['end_idx']

        with Image.open(os.path.join(self.img_dir, (pano_id + ".jpg"))) as img:
            img = np.array(img)
            img = Image.fromarray(np.uint8(img), mode='RGB')

        with h5py.File(self.boxes_file, mode="r") as dataset:
            labels = dataset[f'{self.labels_dataset_name}'][label_start_idx:(label_end_idx + 1)]

        boxes = labels[:, 1:]
        boxes = boxes.astype(np.float32)

        labels = labels[:, 0]
        labels = labels.astype(np.int32)
        labels = labels + 1

        indices = np.arange(boxes.shape[0])

        ###################################################
        # METHOD TO FILTER OUT SMALL BOXES
        # width = boxes[:, 2] - boxes[:, 0]
        # filter = np.asarray(width >= 30).nonzero()
        # boxes = boxes[filter]
        # labels = labels[filter]
        # height = boxes[:, 3] - boxes[:, 1]
        # filter = np.asarray(height >= 30).nonzero()
        # boxes = boxes[filter]
        # labels = labels[filter]
        ###################################################

        #adjust x coordinates for 25px circular padding of image
        boxes[:, 0] = boxes[:, 0] + 25
        boxes[:, 2] = boxes[:, 2] + 25

        #check whether there are boxes with x_min = 0
        x_min_0_indices = np.asarray(boxes[:, 0] == 0).nonzero()
        x_min_0 = boxes[x_min_0_indices]
        #if so check for linked boxes
        if x_min_0.size > 0:
            x_min_0_labels = labels[x_min_0_indices]
            for i, box in enumerate(x_min_0):
                label = x_min_0_labels[i]
                box_idx = np.where(np.all(boxes==box, axis=1))
                box_idx = box_idx[0]

                filter = np.delete(indices, box_idx)
                match_boxes = boxes[filter]
                match_labels = labels[filter]
                filter = np.asarray(match_labels == label).nonzero()
                match_boxes = match_boxes[filter]
                filter = np.asarray(match_boxes[:, 2] == 1400).nonzero()
                match_boxes = match_boxes[filter]

                if match_boxes.size > 0:
                    x_min_left_box = box[0]
                    y_min_left_box = box[1]
                    x_max_left_box = box[2]
                    y_max_left_box = box[3]

                    for match_box in match_boxes:
                        x_min_right_box = match_box[0]
                        y_min_right_box = match_box[1]
                        x_max_right_box = match_box[2]
                        y_max_right_box = match_box[3]

                        idx_right = np.where(np.all(boxes==match_box, axis=1))
                        idx_right = idx_right[0]

                        #if true then boxes are linked
                        if (y_min_right_box == y_min_left_box) & (y_max_right_box == y_max_left_box):
                            #x_max - x_min
                            width2add = x_max_right_box - x_min_right_box
                            width2add = min(width2add, 25)

                            x_min_left_box = x_min_left_box - width2add
                            boxes[box_idx, 0] = x_min_left_box

                            width2add = x_max_left_box - x_min_left_box
                            width2add = min(width2add, 25)

                            x_max_right_box = x_max_right_box + width2add
                            boxes[idx_right, 2] = x_max_right_box

                            break
                        else:
                            continue

        filter1 = np.asarray(boxes[:, 0] < 50).nonzero()
        pad_boxes_left = boxes[filter1]
        filter2 = np.asarray(pad_boxes_left[:, 0] >= 25).nonzero()
        pad_boxes_left = pad_boxes_left[filter2]

        if pad_boxes_left.size > 0:
            filtered_idx = filter1[0][filter2]
            for i, box in enumerate(pad_boxes_left):
                box_idx = filtered_idx[i]
                label = labels[box_idx]

                rightbox2add = box
                rightbox2add[0] = rightbox2add[0] + 1400
                rightbox2add[2] = min(rightbox2add[2] + 1400, 1450)

                width = rightbox2add[2] - rightbox2add[0]

                if width >= 20: 
                    boxes = np.vstack((boxes, rightbox2add))
                    labels = np.append(labels, label)

        filter1 = np.asarray(boxes[:, 2] > 1400).nonzero()
        pad_boxes_right = boxes[filter1]
        filter2 = np.asarray(pad_boxes_right[:, 2] <= 1425).nonzero()
        pad_boxes_right = pad_boxes_right[filter2]

        if pad_boxes_right.size > 0:
            filtered_idx = filter1[0][filter2]
            for i, box in enumerate(pad_boxes_right):
                box_idx = filtered_idx[i]
                label = labels[box_idx]

                leftbox2add = box
                leftbox2add[0] = max(leftbox2add[0] - 1400, 0)
                leftbox2add[2] = min(leftbox2add[2] - 1400, 25)

                width = leftbox2add[2] - leftbox2add[0]

                if width >= 20: 
                    boxes = np.vstack((boxes, leftbox2add))
                    labels = np.append(labels, label)

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # # and labels
        # labels = torch.tensor([10, 20])

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xyxy")
        # add the labels to the boxlist
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        target = target.clip_to_image(remove_empty=False)        

        if self.transforms:
            img, target = self.transforms(img, target)

        # return the image, the boxlist and the idx in your dataset
        return img, target, idx


    def __len__(self):
        return len(self.indices)
    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.img_height, "width": self.img_width}

    def get_indices(self):
        return self.indices

    def get_pano_id(self, idx):
        return self.pano_ids[idx]

    def sample_semi_random(self):
        return False


class PanorAMSDatasetSemiSupervised(object):
    CLASSES = (
        '__background__ ',
        'hoogspanningsmast',
        'tram',
        'train',
        'public_transport_stop',
        'reclamezuil', 
        'metro', 
        'sport', 
        'railway_road_crossings', 
        'bus', 
        'tracks', 
        'traffic_lights', 
        'ferry', 
        'bridge', 
        'bicycle_route', 
        'toilets', 
        'parks', 
        'building', 
        'lichtmast', 
        'waterway', 
        'trees', 
        'windturbine', 
        'trash', 
        'playground', 
        'traffic_signs'
    )

    def __init__(self, img_dir, boxes_file, ann_file, pano_ids, gt_indices, boxes_gt_indices_file,
        noisy_indices, boxes_noisy_indices_file, transforms=None):
        
        self.img_dir = img_dir
        self.boxes_file = boxes_file
        self.ann_file = ann_file

        self.gt_indices = gt_indices
        self.boxes_gt_indices_file = boxes_gt_indices_file
        self.gt_labels_dataset_name = "boxes_gt"

        self.noisy_indices = noisy_indices
        self.boxes_noisy_indices_file = boxes_noisy_indices_file
        self.noisy_labels_dataset_name = "boxes_noisy"

        self.img_height = 550
        self.img_width = 1450
        self.transforms = transforms

        self.id_to_img_map = pano_ids

        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, idx):
        pano_id = self.id_to_img_map[idx]

        #check if gt sample
        if idx in self.gt_indices:
            labels_dataset_name = self.gt_labels_dataset_name

            label_start_idx = self.boxes_gt_indices_file.loc[self.boxes_gt_indices_file['pano_id'] == pano_id].iloc[0]['start_idx']
            label_end_idx = self.boxes_gt_indices_file.loc[self.boxes_gt_indices_file['pano_id'] == pano_id].iloc[0]['end_idx']
        #if idx not gt_indices, it can only be a noisy sample
        else:
            labels_dataset_name = self.noisy_labels_dataset_name

            label_start_idx = self.boxes_noisy_indices_file.loc[self.boxes_noisy_indices_file['pano_id'] == pano_id].iloc[0]['start_idx']
            label_end_idx = self.boxes_noisy_indices_file.loc[self.boxes_noisy_indices_file['pano_id'] == pano_id].iloc[0]['end_idx']           

        with Image.open(os.path.join(self.img_dir, (pano_id + ".jpg"))) as img:
            img = np.array(img)
            img = Image.fromarray(np.uint8(img), mode='RGB')

        with h5py.File(self.boxes_file, mode="r") as dataset:
            labels = dataset[f'{labels_dataset_name}'][label_start_idx:(label_end_idx + 1)]

        boxes = labels[:, 1:]
        boxes = boxes.astype(np.float32)

        labels = labels[:, 0]
        labels = labels.astype(np.int32)
        labels = labels + 1

        indices = np.arange(boxes.shape[0])

        ###################################################
        # METHOD TO FILTER OUT SMALL BOXES
        # width = boxes[:, 2] - boxes[:, 0]
        # filter = np.asarray(width >= 30).nonzero()
        # boxes = boxes[filter]
        # labels = labels[filter]
        # height = boxes[:, 3] - boxes[:, 1]
        # filter = np.asarray(height >= 30).nonzero()
        # boxes = boxes[filter]
        # labels = labels[filter]
        ###################################################

        #adjust x coordinates for 25px circular padding of image
        boxes[:, 0] = boxes[:, 0] + 25
        boxes[:, 2] = boxes[:, 2] + 25

        #check whether there are boxes with x_min = 0
        x_min_0_indices = np.asarray(boxes[:, 0] == 0).nonzero()
        x_min_0 = boxes[x_min_0_indices]
        #if so check for linked boxes
        if x_min_0.size > 0:
            x_min_0_labels = labels[x_min_0_indices]
            for i, box in enumerate(x_min_0):
                label = x_min_0_labels[i]
                box_idx = np.where(np.all(boxes==box, axis=1))
                box_idx = box_idx[0]

                filter = np.delete(indices, box_idx)
                match_boxes = boxes[filter]
                match_labels = labels[filter]
                filter = np.asarray(match_labels == label).nonzero()
                match_boxes = match_boxes[filter]
                filter = np.asarray(match_boxes[:, 2] == 1400).nonzero()
                match_boxes = match_boxes[filter]

                if match_boxes.size > 0:
                    x_min_left_box = box[0]
                    y_min_left_box = box[1]
                    x_max_left_box = box[2]
                    y_max_left_box = box[3]

                    for match_box in match_boxes:
                        x_min_right_box = match_box[0]
                        y_min_right_box = match_box[1]
                        x_max_right_box = match_box[2]
                        y_max_right_box = match_box[3]

                        idx_right = np.where(np.all(boxes==match_box, axis=1))
                        idx_right = idx_right[0]

                        #if true then boxes are linked
                        if (y_min_right_box == y_min_left_box) & (y_max_right_box == y_max_left_box):
                            #x_max - x_min
                            width2add = x_max_right_box - x_min_right_box
                            width2add = min(width2add, 25)

                            x_min_left_box = x_min_left_box - width2add
                            boxes[box_idx, 0] = x_min_left_box

                            width2add = x_max_left_box - x_min_left_box
                            width2add = min(width2add, 25)

                            x_max_right_box = x_max_right_box + width2add
                            boxes[idx_right, 2] = x_max_right_box

                            break
                        else:
                            continue

        filter1 = np.asarray(boxes[:, 0] < 50).nonzero()
        pad_boxes_left = boxes[filter1]
        filter2 = np.asarray(pad_boxes_left[:, 0] >= 25).nonzero()
        pad_boxes_left = pad_boxes_left[filter2]

        if pad_boxes_left.size > 0:
            filtered_idx = filter1[0][filter2]
            for i, box in enumerate(pad_boxes_left):
                box_idx = filtered_idx[i]
                label = labels[box_idx]

                rightbox2add = box
                rightbox2add[0] = rightbox2add[0] + 1400
                rightbox2add[2] = min(rightbox2add[2] + 1400, 1450)

                width = rightbox2add[2] - rightbox2add[0]

                if width >= 20: 
                    boxes = np.vstack((boxes, rightbox2add))
                    labels = np.append(labels, label)

        filter1 = np.asarray(boxes[:, 2] > 1400).nonzero()
        pad_boxes_right = boxes[filter1]
        filter2 = np.asarray(pad_boxes_right[:, 2] <= 1425).nonzero()
        pad_boxes_right = pad_boxes_right[filter2]

        if pad_boxes_right.size > 0:
            filtered_idx = filter1[0][filter2]
            for i, box in enumerate(pad_boxes_right):
                box_idx = filtered_idx[i]
                label = labels[box_idx]

                leftbox2add = box
                leftbox2add[0] = max(leftbox2add[0] - 1400, 0)
                leftbox2add[2] = min(leftbox2add[2] - 1400, 25)

                width = leftbox2add[2] - leftbox2add[0]

                if width >= 20: 
                    boxes = np.vstack((boxes, leftbox2add))
                    labels = np.append(labels, label)

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # # and labels
        # labels = torch.tensor([10, 20])

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xyxy")
        # add the labels to the boxlist
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        target = target.clip_to_image(remove_empty=False)        

        if self.transforms:
            img, target = self.transforms(img, target)

        # return the image, the boxlist and the idx in your dataset
        return img, target, idx


    def __len__(self):
        return len(self.indices)
    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.img_height, "width": self.img_width}

    def get_gt_indices(self):
        return self.gt_indices

    def get_noisy_indices(self):
        return self.noisy_indices

    def get_pano_id(self, idx):
        return self.id_to_img_map[idx]

    def sample_semi_random(self):
        return True