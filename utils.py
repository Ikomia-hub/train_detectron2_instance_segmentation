import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from typing import List, Union
import logging
# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.events import TensorboardXWriter
from detectron2.engine.hooks import BestCheckpointer
from detectron2.utils import comm
from detectron2.utils.events import EventStorage
import os
import torch
import random
import numpy as np
import copy
import pycocotools.mask as mask_util
from detectron2.structures import polygons_to_bitmask
import pycocotools


def polygon_to_rle(polygon: list, shape=(520, 704)):
    '''
    polygon: a list of [x1, y1, x2, y2,....]
    shape: shape of bitmask
    Return: RLE type of mask
    '''
    mask = polygons_to_bitmask([np.asarray(polygon)], shape[0], shape[1]) # add 0.25 can keep the pixels before and after the conversion unchanged
    rle = mask_util.encode(np.asfortranarray(mask))
    return rle


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def polygon2bbox(polygon):
    x1 = min(polygon[::2])
    x2 = max(polygon[::2])
    y1 = min(polygon[1::2])
    y2 = max(polygon[1::2])
    return [x1, y1, x2, y2]


def rgb2mask(img, color2index):
    W = np.power(256, [[0], [1], [2]])
    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)
    for i, c in enumerate(values):
        try:
            mask[img_id == c] = color2index[tuple(img[img_id == c][0])]
        except:
            pass

    return mask


class MyTrainer(DefaultTrainer):
    tensorboard_dir = ""
    output_dir = ""
    eval_period = 0

    def __init__(self, cfg, tb_dir, output_dir, stop_train, log_metrics, update_progress):
        super().__init__(cfg)
        MyTrainer.tensorboard_dir = tb_dir
        MyTrainer.output_dir = output_dir
        MyTrainer.eval_period = cfg.TEST.EVAL_PERIOD
        self.stop_train = stop_train
        self.log_metrics = log_metrics
        self.update_progress = update_progress

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyMapper(True, augmentations=[
            T.RandomBrightness(0.9, 1.1),
            T.RandomFlip(prob=0.5),
            T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MAX_SIZE_TRAIN, max_size=cfg.INPUT.MAX_SIZE_TRAIN)
        ], image_format="RGB"))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, distributed=False, output_dir=cls.output_dir)

    @classmethod
    def build_writers(cls):
        return [TensorboardXWriter(cls.tensorboard_dir)]

    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer(eval_period=self.eval_period,
                                    checkpointer=self.checkpointer,
                                    val_metric="bbox/AP",
                                    mode="max",
                                    file_prefix="model_best"))
        return ret

    def train(self):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.update_progress()
                    if self.iter % 20 == 0:
                        self.log_metrics(
                            {name.replace("@", "_"): value[0] for name, value in self.storage.latest().items()},
                            step=self.iter)
                    if self.stop_train():
                        logger.info("Training stopped by user at iteration {}".format(self.iter))
                        with open(os.path.join(self.output_dir, "model_final.pth"), "w") as f:
                            f.write("")
                        self.checkpointer.save("model_final")
                        break
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results


class MyMapper(DatasetMapper):
    def __init__(
            self,
            is_train: bool,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            keypoint_hflip_indices=None):
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.use_instance_mask = True
        self.use_keypoint = False
        self.instance_mask_format = "bitmask"
        self.recompute_boxes = False
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # can use other ways to read image
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


def register_dataset(dataset_name, images, metadata):
    DatasetCatalog.register(dataset_name, lambda: images)
    MetadataCatalog.get(dataset_name).thing_classes = [v for k, v in metadata["category_names"].items()]


def register_datasets(data, split, had_bckgnd_class):
    data = copy.deepcopy(data)
    class_offset = 0 if had_bckgnd_class else 1

    for i, sample in enumerate(data["images"]):
        if "file_name" not in sample.keys():
            sample["file_name"] = sample["filename"]
            if "category_colors" in data["metadata"]:
                sample["category_colors"] = {color:i for i,color in enumerate(data["metadata"]["category_colors"])}
            sample["image_id"] = i

        if "instance_seg_masks_file" in sample:
            mask_file = sample["instance_seg_masks_file"]
            d = np.load(mask_file)
            masks = np.transpose(d["arr_0"],[2,0,1])
            for mask,anno in zip(masks,sample["annotations"]):
                anno["segmentation"] = pycocotools.mask.encode(np.asarray(mask.astype("uint8"), order="F"))
                anno["bbox_mode"] = BoxMode.XYWH_ABS
                anno["category_id"] += class_offset
        else:
            for anno in sample["annotations"]:
                anno["bbox_mode"] = BoxMode.XYXY_ABS

                if "segmentation_poly" in anno:
                    seg_poly = anno.pop("segmentation_poly")[0]
                    anno["segmentation"] = polygon_to_rle(seg_poly, shape=(sample["height"], sample["width"]))
                    anno["bbox"] = polygon2bbox(seg_poly)
                    anno["category_id"] += class_offset

    random.seed(10)
    random.shuffle(data["images"])
    split_id = int(len(data["images"]) * split)
    train_imgs = data["images"][:split_id]
    test_imgs = data["images"][split_id:]
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    register_dataset("TrainDetectionDataset", train_imgs, data["metadata"])
    register_dataset("TestDetectionDataset", test_imgs, data["metadata"])
    random.seed(0)
