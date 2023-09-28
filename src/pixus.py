from windows import WindowManagerX11
from bot_worker import BotWorker
from view import PixusView
from PyQt6.QtWidgets import QApplication
import sys


def main():
    app = QApplication(sys.argv)
    model_path = "../yolov8_models/best_chataignier_only.pt"
    wm11 = WindowManagerX11("dofus.exe")
    model = BotWorker(wm11, model_path)
    pixus = PixusView(model)
    pixus.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
