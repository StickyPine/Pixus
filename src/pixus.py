from controller import MainViewController
from window_handler import create_window_handler
from bot_worker import BotWorker
from view import PixusView
from PyQt6.QtWidgets import QApplication
from ultralytics import YOLO
import sys


def main():
    app = QApplication(sys.argv)
    model_path = "../yolov8_models/best_chataignier_only.pt"

    ressource_model = YOLO(model_path)
    wh = create_window_handler(win_name="Dofus", x11_name="dofus.exe")
    bot_worker = BotWorker(wh, ressource_model)

    pixus = MainViewController(bot_worker)
    pixus.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
