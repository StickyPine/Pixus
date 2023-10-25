from typing import *


class RessourceWindow:
    def __init__(self, ratio_width: int, ratio_height: int):
        self.__RATIO_W = ratio_width
        self.__RATIO_H = ratio_height
        self.__SCALE_W = 0.85
        self.__SCALE_H = 0.75
        self.__SCALE_OFFSET_Y = 0.05

        self.__top_left_x = None
        self.__top_left_y = None
        self.__bottom_right_x = None
        self.__bottom_right_y = None

    def update_window(self, img_h: int, img_w: int) -> None:
        # compute the virtual window size
        win_w = (img_h*self.__RATIO_W // self.__RATIO_H) * self.__SCALE_W
        win_h = img_h * self.__SCALE_H

        # centered
        self.__top_left_x = int((img_w - win_w) // 2)
        self.__bottom_right_x = int(img_w - self.__top_left_x)

        self.__top_left_y = int(img_h * self.__SCALE_OFFSET_Y)
        self.__bottom_right_y = int(self.__top_left_y + win_h)

    def is_point_in_window(self, x: int, y: int) -> bool:
        if x <= self.__top_left_x or x >= self.__bottom_right_x:
            return False
        if y <= self.__top_left_y or y >= self.__bottom_right_y:
            return False
        return True

    @property
    def ratio_w(self) -> int:
        return self.__RATIO_W

    @property
    def ratio_h(self) -> int:
        return self.__RATIO_H

    @property
    def top_left(self) -> Tuple[int, int]:
        return self.__top_left_x, self.__top_left_y

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return self.__bottom_right_x, self.__bottom_right_y
