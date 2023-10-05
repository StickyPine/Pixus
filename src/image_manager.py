import numpy as np
import cv2
import os
from typing import *


class ImageManager:

    def __init__(self, assets_path: str):
        self.__cursors = self.__load_img_from_dir(assets_path
                                                  + "cursor_templates/")

    @staticmethod
    def __load_img_from_dir(dir_path: str) -> List[np.ndarray]:
        images = []
        for dirpath, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(dirpath, filename)
                    images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        return images

    @property
    def cursors(self):
        return self.__cursors

