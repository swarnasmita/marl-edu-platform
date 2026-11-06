import mujoco
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
W, H = 640, 480
renderer = mujoco.Renderer(model, height=H, width=W)

app = QApplication([])
view = QLabel()
view.resize(W, H)
view.show()

def tick():
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    frame = np.ascontiguousarray(renderer.render())
    img = QImage(frame.data, W, H, 3 * W, QImage.Format_RGB888).copy()
    view.setPixmap(QPixmap.fromImage(img))

timer = QTimer()
timer.timeout.connect(tick)
timer.start(16)  # roughly 60 fps
app.exec()