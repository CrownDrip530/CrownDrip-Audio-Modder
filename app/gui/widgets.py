"""
gui/widgets.py
Custom-styled widgets for the gold & black theme.
"""

from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, Property, Signal
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QWidget


class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None, checked=False):
        super().__init__(parent)
        self._checked = checked
        self._circle_pos = 22.0 if checked else 3.0
        self.setFixedSize(46, 24)
        self.setCursor(Qt.PointingHandCursor)

        self._anim = QPropertyAnimation(self, b"circle_pos", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def get_circle_pos(self):
        return self._circle_pos

    def set_circle_pos(self, pos):
        self._circle_pos = pos
        self.update()

    circle_pos = Property(float, get_circle_pos, set_circle_pos)

    def is_checked(self):
        return self._checked

    def set_checked(self, checked: bool, emit=True):
        if self._checked == checked:
            return
        self._checked = checked
        self._anim.stop()
        self._anim.setStartValue(self._circle_pos)
        self._anim.setEndValue(22.0 if checked else 3.0)
        self._anim.start()
        if emit:
            self.toggled.emit(checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_checked(not self._checked)

    def sizeHint(self):
        return QSize(46, 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track_color = QColor("#ffd166") if self._checked else QColor("#2a2a2a")
        border_color = QColor("#ffd166") if self._checked else QColor("#5a4a1a")

        painter.setPen(border_color)
        painter.setBrush(track_color)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 12, 12)

        circle_color = QColor("#0d0d0d") if self._checked else QColor("#8a7a4a")
        painter.setPen(Qt.NoPen)
        painter.setBrush(circle_color)
        painter.drawEllipse(int(self._circle_pos), 3, 18, 18)
