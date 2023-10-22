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
        self.__HEIGTH = 736 
        self.__WIDTH = 928

        self.__ANIMATION_SLEEP = 1.5

        self.__WM = window_manager
        self.is_running = True
        self.resize_window()

        self.__wait_condition = QWaitCondition()
        self.__mutex = QMutex()
        self.__paused = True

        self.__debug_window = False

        self.__ressource_model = ressource_model
        self.__threshold = 0.9

        self.__mouse = mouse.Controller()
        self.__keyboard = keyboard.Controller()

        self.__desired_class = set()

    def __draw_boxe(self, img: np.ndarray, x1: int, x2: int, y1: int, y2: int,
                    class_name: str) -> None:
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(img, class_name.upper(),
                    (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3, cv2.LINE_AA)

    def __resize_image(self, img):
        h, w, _ = img.shape
        longest_edge = max(h, w)
        top_padding = (longest_edge - h) // 2
        bottom_padding = longest_edge - h - top_padding
        left_padding = (longest_edge - w) // 2
        right_padding = longest_edge - w - left_padding
        padded_image = cv2.copyMakeBorder(img, top_padding, bottom_padding,
                                          left_padding, right_padding,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized_image = cv2.resize(padded_image, (self.__WIDTH, self.__HEIGTH))
        return resized_image

    def __shift_click(self) -> None:
        self.__keyboard.press(keyboard.Key.shift)
        self.__mouse.click(mouse.Button.left, 1)
        self.__keyboard.release(keyboard.Key.shift)

    def __debug(self, img, result_list, names):
        for result in result_list:
            x1, y1, x2, y2, score, class_id = result
            if score > self.__threshold:
                class_name = names[int(class_id)]
                box_center = int((x1 + x2)//2), int((y1 + y2)//2)
                x1, y1, x2, y2, score, class_id = result
                self.__draw_boxe(img, x1, x2, y1, y2, class_name)
                cv2.circle(img, box_center, 10, (0, 255, 0), -1)
        self.display_image_signal.emit(img, "Pixus Debug")

    def stop(self):
        self.is_running = False
        self.resume()

    def resize_window(self):
        self.__WM.resize_window(self.__WIDTH, self.__HEIGTH)

    def add_desired_class(self, class_id: int) -> None:
        self.__desired_class.add(class_id)

    def remove_desired_class(self, class_id: int) -> None:
        self.__desired_class.remove(class_id)

    def debug_window_on(self):
        self.__debug_window = True

    def debug_window_off(self):
        self.__debug_window = False

    def pause(self) -> None:
        self.__mutex.lock()
        self.__paused = True
        self.__mutex.unlock()

    def resume(self) -> None:
        self.__mutex.lock()
        self.__paused = False
        self.__mutex.unlock()
        self.__wait_condition.wakeAll()

    def run(self) -> None:
        while self.is_running:
            img = self.__WM.get_window_array()
            results = self.__ressource_model(img)[0]
            result_list = results.boxes.data.tolist()

            self.__mutex.lock()
            if self.__paused:
                self.__wait_condition.wait(self.__mutex)
            self.__mutex.unlock()

            if self.__debug_window:
                self.__debug(img, result_list, results.names)

            if result_list:
                result_list = sorted(result_list, key=lambda item: (
                    (item[0]**2 + item[1]**2)**(1/2)))
                x1, y1, x2, y2, score, class_id = result_list[0]
                if score > self.__threshold:
                    box_center = int((x1 + x2)//2), int((y1 + y2)//2)

                    abs_coord = self.__WM.translate_position(box_center[0], box_center[1])
                    self.__mouse.position = abs_coord
                    self.__shift_click()
                    time.sleep(self.__ANIMATION_SLEEP)
