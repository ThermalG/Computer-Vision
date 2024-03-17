from PyQt5.QtWidgets import QDialog, QCheckBox, QPushButton, QVBoxLayout, QProgressBar
from PyQt5.QtCore import pyqtSignal


class GUI(QDialog):
    start_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.calc_speed = QCheckBox('Real-time speed estimation')
        self.write_video = QCheckBox('Save processed video')
        self.start_button = QPushButton('START')
        self.progress_bar = QProgressBar()
        layout = QVBoxLayout()
        layout.addWidget(self.calc_speed)
        layout.addWidget(self.write_video)
        layout.addWidget(self.start_button)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
        self.start_button.clicked.connect(self.emit_start_signal)

    def emit_start_signal(self):
        self.start_signal.emit()

    def get_pref(self):
        calc_speed = self.calc_speed.isChecked()
        write_video = self.write_video.isChecked()
        return calc_speed, write_video
