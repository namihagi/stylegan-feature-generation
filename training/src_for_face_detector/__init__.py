import os

from .anchor_generator import AnchorGenerator
from .detector_input_feature import Detector
from .network_input_feature import FeatureExtractor


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
