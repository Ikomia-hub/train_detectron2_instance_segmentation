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
from ikomia.utils import pyqtutils, qtconversion
from train_detectron2_instance_segmentation.train_detectron2_instance_segmentation_process import TrainDetectron2InstanceSegmentationParam
import detectron2
import os

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainDetectron2InstanceSegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainDetectron2InstanceSegmentationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # Model name
        config_paths = os.path.dirname(detectron2.__file__) + "/model_zoo"

        available_cfg = []
        for root, dirs, files in os.walk(config_paths, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                possible_cfg = os.path.join(*file_path.split('/')[-2:])
                if "InstanceSegmentation" in possible_cfg and possible_cfg.endswith('.yaml') and "Base" not in possible_cfg:
                    available_cfg.append(possible_cfg.replace('.yaml', ''))
        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model Name")
        for model_name in available_cfg:
            self.combo_model.addItem(model_name)
        self.combo_model.setCurrentText(self.parameters.cfg["model_name"])
        # Input size
        self.spin_input_size = pyqtutils.append_spin(self.gridLayout, "Max input size",
                                                     self.parameters.cfg["input_size"],
                                                     min=16)
        # Max iteration
        self.spin_max_iter = pyqtutils.append_spin(self.gridLayout, "Max iter", self.parameters.cfg["max_iter"],
                                                   min=10)
        # Batch size
        self.spin_batch_size = pyqtutils.append_spin(self.gridLayout, "Batch size",
                                                     self.parameters.cfg["batch_size"])
        # Split train/test
        self.slider_split = pyqtutils.append_slider(self.gridLayout, "Split train/test (%)",
                                                    round(100 * self.parameters.cfg["dataset_split_ratio"]), min=0,
                                                    max=100)
        # Pretrain
        self.check_pretrained = pyqtutils.append_check(self.gridLayout, "Pretrained on Imagenet",
                                                       self.parameters.cfg["use_pretrained"])
        # Output folder
        self.browse_output_folder = pyqtutils.append_browse_file(self.gridLayout, "Output folder",
                                                                 self.parameters.cfg["output_folder"],
                                                                 mode=pyqtutils.QFileDialog.Directory)
        # Eval period
        self.spin_eval_period = pyqtutils.append_spin(self.gridLayout, "Evaluation period",
                                                      self.parameters.cfg["eval_period"])
        # Base learning rate
        self.double_spin_lr = pyqtutils.append_double_spin(self.gridLayout, "Learning rate",
                                                           self.parameters.cfg["learning_rate"], step=1e-4)
        # Custom config
        self.check_custom_cfg = pyqtutils.append_check(self.gridLayout, "Enable expert mode",
                                                       self.parameters.cfg["use_custom_cfg"])

        self.browse_custom_cfg = pyqtutils.append_browse_file(self.gridLayout, "Custom config path",
                                                              self.parameters.cfg["config"])
        # Disable unused widgets when custom config checkbox is checked
        self.browse_custom_cfg.setEnabled(self.check_custom_cfg.isChecked())
        self.slider_split.setEnabled(not self.check_custom_cfg.isChecked())
        self.double_spin_lr.setEnabled(not self.check_custom_cfg.isChecked())
        self.browse_output_folder.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_pretrained.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_batch_size.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_max_iter.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_input_size.setEnabled(not self.check_custom_cfg.isChecked())
        self.combo_model.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_eval_period.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_custom_cfg.stateChanged.connect(self.on_check)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_check(self, int):
        self.slider_split.setEnabled(not self.check_custom_cfg.isChecked())
        self.double_spin_lr.setEnabled(not self.check_custom_cfg.isChecked())
        self.browse_output_folder.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_pretrained.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_batch_size.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_max_iter.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_input_size.setEnabled(not self.check_custom_cfg.isChecked())
        self.combo_model.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_eval_period.setEnabled(not self.check_custom_cfg.isChecked())
        self.browse_custom_cfg.setEnabled(self.check_custom_cfg.isChecked())

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.cfg["model_name"] = self.combo_model.currentText()
        self.parameters.cfg["use_custom_cfg"] = self.check_custom_cfg.isChecked()
        self.parameters.cfg["config"] = self.browse_custom_cfg.path
        self.parameters.cfg["max_iter"] = self.spin_max_iter.value()
        self.parameters.cfg["batch_size"] = self.spin_batch_size.value()
        self.parameters.cfg["input_size"] = self.spin_input_size.value()
        self.parameters.cfg["use_pretrained"] = self.check_pretrained.isChecked()
        self.parameters.cfg["output_folder"] = self.browse_output_folder.path
        self.parameters.cfg["learning_rate"] = self.double_spin_lr.value()
        self.parameters.cfg["dataset_split_ratio"] = self.slider_split.value() / 100
        self.parameters.cfg["eval_period"] = self.spin_eval_period.value()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainDetectron2InstanceSegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_detectron2_instance_segmentation"

    def create(self, param):
        # Create widget object
        return TrainDetectron2InstanceSegmentationWidget(param, None)
