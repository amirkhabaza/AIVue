import sys
import os
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
)
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QFont, QColor, QPalette

class LaunchScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aivue Calibration")
        self.setFixedSize(600, 300)
        self.setStyleSheet("background-color: #16213e;") 

        self.setup_ui()

    

    def setup_ui(self):
        # Company name label
        title = QLabel("Aivue")
        title.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        button_width = 200
        button_height = 40

        # Start button
        start_button = QPushButton("Start Calibration")
        start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        start_button.setFixedSize(button_width, button_height)
        start_button.setStyleSheet("""
            QPushButton {
                background-color:  #f8a5c2;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #ff758f;
            }
            QPushButton:pressed {
                background-color: #f55674;
            }
        """)
        start_button.clicked.connect(self.run_script)
        

        # Settings button
        settings_button = QPushButton("⚙️")
        settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_button.setFixedSize(50, 30)
        settings_button.setStyleSheet("""
            QPushButton {
                background-color: #16213e;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {

                border: 1px solid grey;
            }
        """)
        

        # Quit button
        quit_button = QPushButton("Quit")
        quit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        quit_button.setFixedSize(button_width, button_height)
        quit_button.setStyleSheet("""
            QPushButton {
                background-color: #16213e;
                color: white;
                border: 2px solid white;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f55151;
                border: none;
            }
             QPushButton:pressed {
                background-color: #534B62;
            }
        """)
        quit_button.clicked.connect(self.close)  # Closes the app

        # Layout setup
        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(quit_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(settings_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        self.setLayout(layout)

    def run_script(self):
        self.setWindowState(Qt.WindowState.WindowMinimized)
        subprocess.run(["python3", "pyhandler.py"], check = True)    #CHANGE PATH
        subprocess.run(["python3", "pipefacemac.py"], check = True)  #CHANGE PATH
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LaunchScreen()
    window.show()
    sys.exit(app.exec())
