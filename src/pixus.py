from bot_worker import BotWorker
from PySide6.QtWidgets import QApplication
from ultralytics import YOLO
import sys

from view_controller import MainViewController
from window_handler import create_window_handler
from ressource_window import RessourceWindow


def main():
    app = QApplication(sys.argv)
    model_path = "../yolov8_models/best_chataignier_only.pt"

    rs_model = YOLO(model_path)
    wh = create_window_handler(win_name="Dofus", x11_name="dofus.exe")
    rs_window = RessourceWindow(4, 3)
    bot_worker = BotWorker(wh, rs_window, rs_model)

    pixus = MainViewController(bot_worker)
    pixus.setWindowTitle("Pixus")
    pixus.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
