from ultralytics import YOLO
import cv2
import numpy as np
from window_handler import WindowHandlerAbstract
from PySide6.QtCore import QThread, QWaitCondition, QMutex, Signal
import time
from typing import *
from pynput import keyboard, mouse


@final
class BotWorker(QThread):
    display_image_signal = Signal(np.ndarray, str)

    def __init__(self, window_manager: WindowHandlerAbstract,
                 ressource_model: YOLO):
        super().__init__()
        # Should be multiple of 32
        self.__WIN_HEIGTH = 768
        self.__WIN_WIDTH = 960

        self.__RESSOURCE_WIN_WIDTH = 1030
        self.__RESSOURCE_WIN_HEIGHT = 740
        self.__RESSOURCE_WIN_X_OFFSET = 0
        self.__RESSOURCE_WIN_Y_OFFSET = 70

        self.__ANIMATION_SLEEP = 1.5

        self.__WM = window_manager
        self.is_running = True
        self.resize_window()

        self.__wait_condition = QWaitCondition()
        self.__mutex = QMutex()
        self.__paused = True

        self.__debug_window = False

        self.__ressource_model = ressource_model
        self.__threshold = 0.8

        self.__mouse = mouse.Controller()
        self.__keyboard = keyboard.Controller()

        self.__desired_class = set()

    def __shift_click(self) -> None:
        self.__keyboard.press(keyboard.Key.shift)
        self.__mouse.click(mouse.Button.left, 1)
        self.__keyboard.release(keyboard.Key.shift)

    def __is_coord_in_ressource_win(self, x: int, y: int,
                                    img_height: int, img_width) -> bool:
        x_offset = (img_width - self.__RESSOURCE_WIN_WIDTH) // 2
        y_offset = (img_height - self.__RESSOURCE_WIN_HEIGHT) // 2

        x1 = x_offset + self.__RESSOURCE_WIN_X_OFFSET
        y1 = y_offset - self.__RESSOURCE_WIN_Y_OFFSET
        x2 = img_width - x_offset + self.__RESSOURCE_WIN_X_OFFSET
        y2 = img_height - y_offset - self.__RESSOURCE_WIN_Y_OFFSET

        if x <= x1 or x >= x2:
            return False
        if y <= y1 or y >= y2:
            return False
        return True

    def __draw_ressource_window(self, img: np.ndarray) -> None:
        img_height, img_width = img.shape[0], img.shape[1]
        x_offset = (img_width - self.__RESSOURCE_WIN_WIDTH) // 2
        y_offset = (img_height - self.__RESSOURCE_WIN_HEIGHT) // 2

        x1 = x_offset + self.__RESSOURCE_WIN_X_OFFSET
        y1 = y_offset - self.__RESSOURCE_WIN_Y_OFFSET
        x2 = img_width - x_offset + self.__RESSOURCE_WIN_X_OFFSET
        y2 = img_height - y_offset - self.__RESSOURCE_WIN_Y_OFFSET

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    def __draw_ressource_boxes(self, img, result_list, names):
        for result in result_list:
            x1, y1, x2, y2, score, class_id = result
            if score < self.__threshold:
                continue
            if class_id not in self.__desired_class:
                continue

            class_name = names[int(class_id)]
            box_center = int((x1 + x2)//2), int((y1 + y2)//2)
            x1, y1, x2, y2, score, class_id = result
            cv2.rectangle(img, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(img, class_name.upper(),
                        (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(img, box_center, 5, (0, 255, 0), -1)

    def __send_ressource_debug(self, img, result_list, names):
        self.__draw_ressource_window(img)
        self.__draw_ressource_boxes(img, result_list, names)
        self.display_image_signal.emit(img, "Pixus Debug")

    def stop(self):
        self.is_running = False
        self.resume()

    def resize_window(self):
        self.__WM.resize_window(self.__WIN_WIDTH, self.__WIN_HEIGTH)

    def add_desired_class(self, class_id: int) -> None:
        self.__desired_class.add(class_id)

    def remove_desired_class(self, class_id: int) -> None:
        self.__desired_class.remove(class_id)

    def debug_window_on(self):
        self.__debug_window = True

    def debug_window_off(self):
        self.__debug_window = False

    def pause(self) -> None:
        self.__paused = True

    def resume(self) -> None:
        self.__paused = False
        self.__wait_condition.wakeAll()

    def run(self) -> None:
        while self.is_running:
            img = self.__WM.get_window_array()

            results = self.__ressource_model(img)[0]
            result_list = results.boxes.data.tolist()

            if self.__paused:
                self.__wait_condition.wait(self.__mutex)

            if self.__debug_window:
                self.__send_ressource_debug(img, result_list, results.names)

            if result_list:
                result_list = sorted(result_list, key=lambda item: (
                    (item[0]**2 + item[1]**2)**(1/2)))

            for x1, y1, x2, y2, score, class_id in result_list:
                if class_id not in self.__desired_class:
                    continue
                if score < self.__threshold:
                    continue

                c_x, c_y = (x1 + x2)//2, (y1 + y2)//2
                img_h, img_w = img.shape[0], img.shape[1]
                if not self.__is_coord_in_ressource_win(c_x, c_y, img_h, img_w):
                    continue

                abs_coord = self.__WM.translate_position(c_x, c_y)
                self.__mouse.position = abs_coord
                self.__shift_click()
                time.sleep(self.__ANIMATION_SLEEP)
                break
