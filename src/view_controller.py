from PySide6.QtWidgets import *
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem
from ui.TableViewModel import CheckboxDelegate, CustomTableModel
from ui.ui_main_window import Ui_MainWindow
from ressources_manager import RessourcesManager
from bot_worker import BotWorker
from typing import *
from pynput import keyboard
import numpy as np
import cv2

@final
class MainViewController(QMainWindow):
    def __init__(self, bot_model : BotWorker):
        super(MainViewController, self).__init__()

        # setup bot part
        self.__bot_worker = bot_model
        self.__bot_worker.start()
        self.__bot_worker.display_image_signal.connect(self.__display_image)

        # setup ui part
        self.debug = False
        self.run = False
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.BtStart.clicked.connect(self.__toggle_start_stop)
        self.ui.BtDebug.clicked.connect(self.__toggle_debug)
        self.ui.BtResize.clicked.connect(self.__bot_worker.resize_window)
        self.ui.CBoxCateg.currentIndexChanged.connect(self.__load_category)
        
        # setup selection part
        self.ressources_manager = RessourcesManager()
        self.ui.CBoxCateg.addItems([cat.name for cat in self.ressources_manager.data])
        self.ui.CBoxCateg.setStyleSheet("QComboBox QAbstractItemView { alignment: AlignCenter; }")
        self.__load_category()
        
        # other
        self.__key_listener = KeyListener()
        self.__key_listener.stop_key_signal.connect(self.__toggle_start_stop)
        self.__key_listener.finished.connect(self.__key_listener.deleteLater)  # Cleanup the thread when it's finished
        self.__key_listener.start()
        self.__key_listener.stop_key_signal.connect(self.__toggle_start_stop)
    
    def __load_category(self):
        category = self.ui.CBoxCateg.currentText()
        print(category)
        data = self.ressources_manager.get_ressources(category)

        model = CustomTableModel(data)
        self.ui.tableView.setModel(model)
        checkbox_delegate = CheckboxDelegate()
        self.ui.tableView.setItemDelegateForColumn(1, checkbox_delegate)  # Set the delegate for the checkbox column
        # hide row numbers
        self.ui.tableView.verticalHeader().hide()
        
        # set col sizes
        self.ui.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)   # fill the space
        self.ui.tableView.setColumnWidth(1, 30) # fixed size
    
    def __display_image(self, img: np.ndarray, title: str) -> None:
        cv2.imshow(title, img)
        cv2.waitKey(1)
    
    def __toggle_start_stop(self):
        if (self.run):
            self.ui.BtStart.setText('Start')
            self.__bot_worker.pause()
        else:
            self.ui.BtStart.setText('Stop')
            self.__bot_worker.resume()
        self.run = not self.run
            
    def __toggle_debug(self):
        if (self.debug):
            self.ui.BtDebug.setText("Debug off")
            self.__bot_worker.debug_window_off()
        else:
            self.ui.BtDebug.setText("Debug on")
            self.__bot_worker.debug_window_on()
        self.debug = not self.debug


@final
class KeyListener(QThread):

    stop_key_signal = Signal()

    def run(self):
        with keyboard.Listener(on_press=self.__on_press) as listener:
            listener.join()

    def __on_press(self, key):
        try:
            if key == keyboard.Key.f1:
                print("F1 pressed")
                self.stop_key_signal.emit()
        except AttributeError:
            pass