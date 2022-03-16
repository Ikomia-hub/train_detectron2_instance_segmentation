# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.dnn import dnntrain
from ikomia.core.task import TaskParam
# Your imports below
from datetime import datetime
import detectron2
import torch
from detectron2.utils.logger import setup_logger

setup_logger()
import os
import copy

# import some common detectron2 utilities
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from train_detectron2_instance_segmentation.utils import MyMapper, MyTrainer, register_datasets
import gc


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainDetectron2InstanceSegmentationParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.cfg["model_name"] = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
        self.cfg["custom_cfg"] = False
        self.cfg["cfg_path"] = ""
        self.cfg["max_iter"] = 100
        self.cfg["batch_size"] = 2
        self.cfg["input_size"] = 400
        self.cfg["pretrained"] = True
        self.cfg["output_folder"] = os.path.dirname(__file__) + "/runs"
        self.cfg["learning_rate"] = 0.0025
        self.cfg["split_train_test"] = 0.8
        self.cfg["eval_period"] = 50

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["custom_cfg"] = eval(param_map["custom_cfg"])
        self.cfg["cfg_path"] = param_map["cfg_path"]
        self.cfg["max_iter"] = int(param_map["max_iter"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["pretrained"] = eval(param_map["pretrained"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["learning_rate"] = float(param_map["learning_rate"])
        self.cfg["split_train_test"] = float(param_map["split_train_test"])
        self.cfg["eval_period"] = int(param_map["eval_period"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["model_name"] = self.cfg["model_name"]
        param_map["custom_cfg"] = str(self.cfg["custom_cfg"])
        param_map["cfg_path"] = self.cfg["cfg_path"]
        param_map["max_iter"] = str(self.cfg["max_iter"])
        param_map["batch_size"] = str(self.cfg["batch_size"])
        param_map["input_size"] = str(self.cfg["input_size"])
        param_map["pretrained"] = str(self.cfg["pretrained"])
        param_map["output_folder"] = self.cfg["output_folder"]
        param_map["learning_rate"] = str(self.cfg["learning_rate"])
        param_map["split_train_test"] = str(self.cfg["split_train_test"])
        param_map["eval_period"] = str(self.cfg["eval_period"])
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainDetectron2InstanceSegmentation(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        # Example :  self.addInput(dataprocess.CImageIO())
        #           self.addOutput(dataprocess.CImageIO())

        # Create parameters class
        if param is None:
            self.setParam(TrainDetectron2InstanceSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 100

    def update_progress(self):
        self.epochs_done += 1
        steps = range(self.advancement, int(100 * self.epochs_done / self.epochs_todo))
        for step in steps:
            self.emitStepProgress()
            self.advancement += 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        self.stop_train = False
        gc.collect()
        torch.cuda.empty_cache()

        ik_dataset = self.getInput(0)
        had_bckgnd_class = ik_dataset.has_bckgnd_class
        metadata = copy.deepcopy(ik_dataset.data["metadata"]["category_names"])
        images = copy.deepcopy(ik_dataset.data["images"])
        if not had_bckgnd_class:
            tmp_dict = {0: "background"}
            for k, name in metadata.items():
                tmp_dict[k + 1] = name
            metadata["category_names"] = tmp_dict

        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        # Get parameters :
        param = self.getParam()

        tb_logdir = self.getTensorboardLogDir() + "/" + param.cfg["model_name"] + "/" + str_datetime

        if not param.cfg["custom_cfg"]:
            out_dir = param.cfg["output_folder"] + "/" + str_datetime

            cfg = get_cfg()
            config_path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs",
                                       param.cfg["model_name"] + '.yaml')
            cfg.merge_from_file(config_path)
            if not param.cfg["pretrained"]:
                cfg.MODEL.WEIGHTS = None
            cfg.INPUT.MAX_SIZE_TEST = param.cfg["input_size"]
            cfg.INPUT.MAX_SIZE_TRAIN = param.cfg["input_size"]
            cfg.SOLVER.IMS_PER_BATCH = param.cfg["batch_size"]
            cfg.SOLVER.BASE_LR = param.cfg["learning_rate"]
            cfg.SOLVER.MAX_ITER = param.cfg["max_iter"]
            cfg.TEST.EVAL_PERIOD = min(param.cfg["eval_period"], cfg.SOLVER.MAX_ITER - 1)
            cfg.SOLVER.STEPS = (int(0.8 * param.cfg["max_iter"]), int(0.9 * param.cfg["max_iter"]))
            cfg.SOLVER.WARMUP_ITERS = 1000
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.OUTPUT_DIR = out_dir

        else:
            if os.path.isfile(param.cfg["cfg_path"]):
                with open(param.cfg["cfg_path"], 'r') as file:
                    cfg_data = file.read()
                    cfg = CfgNode.load_cfg(cfg_data)
                    out_dir = cfg.OUTPUT_DIR
            else:
                print("Unable to load config file {}".format(param.cfg["cfg_path"]))
                self.endTaskRun()

        os.makedirs(out_dir, exist_ok=True)

        # Fixed dataset names
        cfg.DATASETS.TRAIN = ("TrainDetectionDataset",)
        cfg.DATASETS.TEST = ("TestDetectionDataset",)
        register_datasets({"images": images, "metadata": metadata}, param.cfg["split_train_test"], had_bckgnd_class)
        cfg.CLASS_NAMES = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(cfg.CLASS_NAMES)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.CLASS_NAMES)
        self.advancement = 0
        self.epochs_todo = cfg.SOLVER.MAX_ITER
        self.epochs_done = 0
        with open(os.path.join(out_dir, "config.yaml"), 'w') as f:
            f.write(cfg.dump())
        trainer = MyTrainer(cfg, tb_logdir, out_dir, self.get_stop_train, self.log_metrics, self.update_progress)
        if param.cfg["pretrained"]:
            trainer.resume_or_load(resume=False)  # load MODEL.WEIGHTS
        trainer.train()

        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        # Call endTaskRun to finalize process
        self.endTaskRun()

    def get_stop_train(self):
        return self.stop_train

    def stop(self):
        super(TrainDetectron2InstanceSegmentation, self).stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainDetectron2InstanceSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_detectron2_instance_segmentation"
        self.info.shortDescription = "Train Detectron2 instance segmentation models"
        self.info.description = "Train Detectron2 instance segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentationLink = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "train, detectron2, instance, segmentation"

    def create(self, param=None):
        # Create process object
        return TrainDetectron2InstanceSegmentation(self.info.name, param)
