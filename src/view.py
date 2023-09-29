from typing import *
from bot_worker import BotWorker
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from pynput import keyboard
import cv2


@final
class PixusView(QWidget):
    def __init__(self, bot_model: BotWorker):
        super().__init__()
        self.__key_listener = KeyListener()
        self.__key_listener.start()
        self.__key_listener.stop_key_signal.connect(self.__toggle_start_stop)
        self.__bot_worker = bot_model
        self.__bot_worker.start()
        self.__bot_worker.display_image_signal.connect(self.__display_image)

        self.__init_ui()

    def __display_image(self, img: np.ndarray, title: str) -> None:
        cv2.imshow(title, img)
        cv2.waitKey(1)

    def __init_ui(self):
        self.layout = QVBoxLayout()

        self.start_button = QPushButton('Start', self)
        self.debug_button = QPushButton('Debug_off', self)

        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.debug_button)

        self.start_button.clicked.connect(self.__toggle_start_stop)
        self.debug_button.clicked.connect(self.__toggle_debug)

        self.setLayout(self.layout)

    def __toggle_start_stop(self):
        if self.start_button.text() == 'Start':
            self.start_button.setText('Stop')
            self.__bot_worker.resume()
        else:
            self.start_button.setText('Start')
            self.__bot_worker.pause()

    def __toggle_debug(self):
        if self.debug_button.text() == 'Debug_off':
            self.debug_button.setText('Debug_on')
            self.__bot_worker.debug_window = True
        else:
            self.debug_button.setText('Debug_off')
            self.__bot_worker.debug_window = False


class KeyListener(QThread):

    stop_key_signal = pyqtSignal()

    def run(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if key == keyboard.Key.f1:
                print("F1 pressed")
                self.stop_key_signal.emit()
        except AttributeError:
            pass

