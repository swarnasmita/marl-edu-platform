from PySide6.QtWidgets import QApplication, QLabel
app = QApplication([])
label = QLabel("PySide6 works!")
label.resize(300, 100)
label.show()
app.exec()