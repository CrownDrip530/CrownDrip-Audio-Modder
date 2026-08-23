"""
gui/main_window.py
Main application window: gold & black themed control panel for the mic
routing engine + soundboard.
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QListWidget, QListWidgetItem,
    QFrame, QFileDialog, QInputDialog, QMessageBox, QLineEdit, QStatusBar
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config as cfg
from audio_engine import AudioEngine
from gui.widgets import ToggleSwitch


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CrownDrip Audio Modder")
        self.resize(780, 720)

        self.config = cfg.load_config()
        self.engine = AudioEngine()

        self._load_theme()
        self._build_ui()
        self._apply_loaded_config()
        self._populate_devices()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Select your mic + CABLE Input, then hit Start.")

    def _load_theme(self):
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            base_path = Path(sys._MEIPASS) / "gui"
        else:
            base_path = Path(__file__).resolve().parent
        qss_path = base_path / "theme.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 14, 18, 14)
        root_layout.setSpacing(10)

        title = QLabel("\u2654 CROWNDRIP AUDIO MODDER")
        title.setObjectName("AppTitle")
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        # ---- Device panel ----
        device_panel = self._make_panel()
        dp_layout = QVBoxLayout(device_panel)
        dp_layout.addWidget(self._section_label("Devices"))

        mic_row = QHBoxLayout()
        mic_row.addWidget(QLabel("Mic Input:"))
        self.mic_combo = QComboBox()
        mic_row.addWidget(self.mic_combo, 1)
        dp_layout.addLayout(mic_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output (Virtual Cable):"))
        self.output_combo = QComboBox()
        out_row.addWidget(self.output_combo, 1)
        dp_layout.addLayout(out_row)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("\u25B6  Start Routing")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self.toggle_engine)
        btn_row.addWidget(self.start_btn)
        dp_layout.addLayout(btn_row)

        root_layout.addWidget(device_panel)

        # ---- Mic control panel ----
        mic_panel = self._make_panel()
        mp_layout = QVBoxLayout(mic_panel)
        mp_layout.addWidget(self._section_label("Mic Controls"))

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Mic Gain (dB):"))
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setMinimum(-50)
        self.gain_slider.setMaximum(50)
        self.gain_slider.setValue(0)
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        gain_row.addWidget(self.gain_slider, 1)
        self.gain_value_label = QLabel("0 dB")
        self.gain_value_label.setFixedWidth(50)
        gain_row.addWidget(self.gain_value_label)
        mp_layout.addLayout(gain_row)

        fry_row = QHBoxLayout()
        fry_label = QLabel("\U0001F35F Deep Fried Mode (Mic):")
        fry_row.addWidget(fry_label)
        fry_row.addStretch()
        self.deep_fry_toggle = ToggleSwitch()
        self.deep_fry_toggle.toggled.connect(self.on_deep_fry_toggled)
        fry_row.addWidget(self.deep_fry_toggle)
        mp_layout.addLayout(fry_row)

        root_layout.addWidget(mic_panel)

        # ---- Soundboard panel ----
        sb_panel = self._make_panel()
        sb_layout = QVBoxLayout(sb_panel)
        sb_layout.addWidget(self._section_label("Soundboard"))

        sound_gain_row = QHBoxLayout()
        sound_gain_row.addWidget(QLabel("MP3 Volume (dB):"))
        self.sound_gain_slider = QSlider(Qt.Horizontal)
        self.sound_gain_slider.setMinimum(-50)
        self.sound_gain_slider.setMaximum(50)
        self.sound_gain_slider.setValue(0)
        self.sound_gain_slider.valueChanged.connect(self.on_sound_gain_changed)
        sound_gain_row.addWidget(self.sound_gain_slider, 1)
        self.sound_gain_value_label = QLabel("0 dB")
        self.sound_gain_value_label.setFixedWidth(50)
        sound_gain_row.addWidget(self.sound_gain_value_label)
        sb_layout.addLayout(sound_gain_row)

        sound_fry_row = QHBoxLayout()
        sound_fry_label = QLabel("\U0001F35F Deep Fried Mode (MP3):")
        sound_fry_row.addWidget(sound_fry_label)
        sound_fry_row.addStretch()
        self.sound_deep_fry_toggle = ToggleSwitch()
        self.sound_deep_fry_toggle.toggled.connect(self.on_sound_deep_fry_toggled)
        sound_fry_row.addWidget(self.sound_deep_fry_toggle)
        sb_layout.addLayout(sound_fry_row)

        self.sound_list = QListWidget()
        self.sound_list.itemDoubleClicked.connect(self.on_play_selected)
        sb_layout.addWidget(self.sound_list, 1)

        sb_btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add MP3")
        add_btn.clicked.connect(self.on_add_mp3)
        play_btn = QPushButton("\u25B6 Play")
        play_btn.clicked.connect(self.on_play_selected)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self.on_rename_selected)
        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("DangerButton")
        remove_btn.clicked.connect(self.on_remove_selected)
        stop_btn = QPushButton("\u25A0 Stop All")
        stop_btn.clicked.connect(self.on_stop_all)

        for b in (add_btn, play_btn, rename_btn, remove_btn, stop_btn):
            sb_btn_row.addWidget(b)
        sb_layout.addLayout(sb_btn_row)

        root_layout.addWidget(sb_panel, 1)

    def _make_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("Panel")
        return panel

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionTitle")
        return lbl

    # ---------------- config wiring ----------------

    def _apply_loaded_config(self):
        mic_gain = int(self.config.get("mic_gain_db", 0.0))
        self.gain_slider.setValue(mic_gain)
        self.gain_value_label.setText(f"{mic_gain} dB")

        sound_gain = int(self.config.get("soundboard_volume_db", 0.0))
        self.sound_gain_slider.setValue(sound_gain)
        self.sound_gain_value_label.setText(f"{sound_gain} dB")

        deep_fry_cfg = self.config.get("effects", {}).get("deep_fry", {})
        self.deep_fry_toggle.set_checked(deep_fry_cfg.get("enabled", False), emit=False)

        sound_fry_cfg = self.config.get("effects", {}).get("soundboard_deep_fry", {})
        self.sound_deep_fry_toggle.set_checked(sound_fry_cfg.get("enabled", False), emit=False)

        self.engine.load_config_dict(self.config)

        self._refresh_sound_list()

    def _populate_devices(self):
        mics = AudioEngine.list_input_devices()
        outputs = AudioEngine.list_output_devices()

        self.mic_combo.clear()
        for d in mics:
            self.mic_combo.addItem(d["name"], d["index"])

        self.output_combo.clear()
        cable_index = None
        for d in outputs:
            self.output_combo.addItem(d["name"], d["index"])
            if "CABLE Input" in d["name"]:
                cable_index = self.output_combo.count() - 1

        saved_mic = self.config.get("mic_device")
        if saved_mic:
            idx = self.mic_combo.findText(saved_mic)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)

        saved_out = self.config.get("output_device")
        if saved_out:
            idx = self.output_combo.findText(saved_out)
            if idx >= 0:
                self.output_combo.setCurrentIndex(idx)
        elif cable_index is not None:
            self.output_combo.setCurrentIndex(cable_index)

    def _refresh_sound_list(self):
        self.sound_list.clear()
        for entry in self.config.get("soundboard", []):
            item = QListWidgetItem(entry["name"])
            item.setData(Qt.UserRole, entry["id"])
            self.sound_list.addItem(item)

    def _save(self):
        self.config["mic_device"] = self.mic_combo.currentText()
        self.config["output_device"] = self.output_combo.currentText()
        self.config.update(self.engine.to_config_dict())
        cfg.save_config(self.config)

    # ---------------- actions ----------------

    def toggle_engine(self):
        if self.engine.is_running():
            self.engine.stop()
            self.start_btn.setText("\u25B6  Start Routing")
            self.status.showMessage("Stopped.")
        else:
            mic_idx = self.mic_combo.currentData()
            out_idx = self.output_combo.currentData()
            if mic_idx is None or out_idx is None:
                QMessageBox.warning(self, "Missing device", "Please select a mic and output device.")
                return
            self.engine.mic_device = mic_idx
            self.engine.output_device = out_idx
            try:
                self.engine.start()
                self.start_btn.setText("\u25A0  Stop Routing")
                self.status.showMessage(f"Routing live: mic -> {self.output_combo.currentText()}")
            except Exception as e:
                QMessageBox.critical(self, "Failed to start", str(e))
            self._save()

    def on_gain_changed(self, value):
        self.gain_value_label.setText(f"{value} dB")
        self.engine.set_mic_gain_db(float(value))
        self._save()

    def on_deep_fry_toggled(self, checked):
        self.engine.set_deep_fry_enabled(checked)
        self.status.showMessage("Mic Deep Fried Mode: ON" if checked else "Mic Deep Fried Mode: OFF")
        self._save()

    def on_sound_gain_changed(self, value):
        self.sound_gain_value_label.setText(f"{value} dB")
        self.engine.set_soundboard_gain_db(float(value))
        self._save()

    def on_sound_deep_fry_toggled(self, checked):
        self.engine.set_soundboard_deep_fry_enabled(checked)
        self.status.showMessage("MP3 Deep Fried Mode: ON" if checked else "MP3 Deep Fried Mode: OFF")
        self._save()

    def on_add_mp3(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select MP3", "", "Audio Files (*.mp3 *.wav *.ogg)")
        if not filepath:
            return
        default_name = Path(filepath).stem
        name, ok = QInputDialog.getText(self, "Name this sound", "Display name:", QLineEdit.Normal, default_name)
        if not ok:
            return
        cfg.add_sound_to_library(self.config, filepath, name)
        self._refresh_sound_list()
        self.status.showMessage(f"Added '{name}' to soundboard.")

    def _selected_entry(self):
        item = self.sound_list.currentItem()
        if not item:
            return None
        sound_id = item.data(Qt.UserRole)
        return next((e for e in self.config["soundboard"] if e["id"] == sound_id), None)

    def on_play_selected(self):
        entry = self._selected_entry()
        if not entry:
            QMessageBox.information(self, "No sound selected", "Select a sound from the list first.")
            return
        filepath = cfg.get_sound_path(entry)
        try:
            self.engine.play_sound(filepath, entry.get("volume", 1.0))
            self.status.showMessage(f"Playing: {entry['name']}")
        except Exception as e:
            QMessageBox.critical(self, "Playback error", str(e))

    def on_rename_selected(self):
        entry = self._selected_entry()
        if not entry:
            return
        name, ok = QInputDialog.getText(self, "Rename sound", "New name:", QLineEdit.Normal, entry["name"])
        if ok and name.strip():
            cfg.rename_sound(self.config, entry["id"], name)
            self._refresh_sound_list()

    def on_remove_selected(self):
        entry = self._selected_entry()
        if not entry:
            return
        confirm = QMessageBox.question(self, "Remove sound", f"Remove '{entry['name']}' from soundboard?")
        if confirm == QMessageBox.Yes:
            cfg.remove_sound(self.config, entry["id"])
            self._refresh_sound_list()

    def on_stop_all(self):
        self.engine.stop_all_sounds()
        self.status.showMessage("Stopped all soundboard playback.")

    def closeEvent(self, event):
        self._save()
        self.engine.stop()
        event.accept()


def run():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
