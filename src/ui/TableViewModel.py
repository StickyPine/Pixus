from typing import Union
from PySide6.QtCore import QAbstractTableModel, Qt, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QStyledItemDelegate, QCheckBox

from ressources_manager import Ressource

class CustomTableModel(QStandardItemModel):
    checkboxStateChanged = Signal(int, bool)    # signal emitted when a checkbox is clicked
    
    def __init__(self, ressources: [Ressource], parent=None):
        super(CustomTableModel, self).__init__(parent)
        self.col_names = ["Nom", "Actif"]
        self.setColumnCount(2)  # Two columns: one for the string, one for the checkbox
        self.setHorizontalHeaderLabels(self.col_names)

        self.ressources = ressources
        for ressource in ressources:
            string_item = QStandardItem(ressource.pretty_name)
            checkbox_item = QStandardItem()
            checkbox_item.setCheckable(True)
            checkbox_item.setCheckState(Qt.Checked if ressource.enabled else Qt.Unchecked)
            
            checkbox_item.emitDataChanged()

            self.appendRow([string_item, checkbox_item])

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.CheckStateRole and index.column() == 1:   # update the enabled value
            enabled = value != 0
            self.ressources[index.row()].enabled = enabled
            self.checkboxStateChanged.emit(self.ressources[index.row()].id, enabled)    # send signal to controller
            return super().setData(index, value, Qt.CheckStateRole)
        return super().setData(index, value, role)
    
class CheckboxDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
    
    def createEditor(self, parent, option, index):
        editor = QCheckBox(parent)
        editor.setCheckState(Qt.Unchecked)
        editor.clicked.connect(self.commitData)
        return editor

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.CheckStateRole)
        editor.setCheckState(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.checkState(), Qt.CheckStateRole)