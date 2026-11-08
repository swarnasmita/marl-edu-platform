"""Disaster Response viewer (post-earthquake urban scene).

Arrow keys move your robot (purple). Drive up to a victim to pick them up, then
bring them to the medical station's triage area (red-bordered tarp with cots).
The blue robots use the greedy path-planning policy for now; trained MARL agents
will replace it later.

Victims are highlighted with circles in their triage colour, labelled with
severity and time left; robots are labelled YOU / R1 / R2.

Mouse (free camera): left-drag = rotate, right-drag = pan, wheel = zoom.
Keys: arrows = move, C = next camera, F = free camera follows your robot,
      H = hide/show highlights, L = show your robot's distance sensors, R = restart.
Run from the project root:  python envs/view_disaster.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from agents.heuristic_policy import GreedyPolicy, direction_to_action
from envs.disaster_env import (CAMERAS, CARRIED, DT, LEARNER, LOST, MAX_STEPS, N_ROBOTS, N_VICTIMS,
                               R_RESCUE, RAY_ANGLES, RAY_RANGE, SAFE_CENTER, SEVERITY_COLOURS,
                               SEVERITY_NAMES, UNREPORTED, URGENT_STEPS, WAITING, DisasterResponseEnv)

W, H = 1152, 720
STEP_MS = int(1000 * DT)          # real-time playback
CAMERA_LABELS = {"free": "Free (mouse)", "overview": "Overview", "isometric": "Isometric",
                 "follow": "Follow your robot", "triage": "Medical station",
                 "ruins": "Collapsed houses (NW)", "collapse": "Collapse site (NE/E)"}
ROBOT_LABELS = {0: ("YOU", QColor(170, 90, 240)), 1: ("R1", QColor(40, 130, 245)), 2: ("R2", QColor(40, 130, 245))}


def qcolour(rgb, alpha=255):
    return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha)


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.env = DisasterResponseEnv(render_mode="rgb_array", width=W, height=H, camera="free")
        self.policy = GreedyPolicy(self.env)
        self.pressed = set()
        self.highlights = True
        self.show_rays = False
        self.reported_at = {}
        self.follow_me = False
        self.drag = None
        self.message = "Reach a victim to pick them up, then bring them to the triage tarp."

        self.image = QLabel()
        self.image.setFixedSize(W, H)
        self.status = QLabel()
        self.status.setStyleSheet("font-size: 14px; padding: 6px;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.image)
        layout.addWidget(self.status)
        self.setWindowTitle("Disaster Response - urban earthquake rescue")
        self.setFocusPolicy(Qt.StrongFocus)

        self.restart()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(STEP_MS)

    def restart(self):
        self.reported_at = {}
        self.env.reset()
        self.policy.reset()
        self.draw()

    # --- keyboard -------------------------------------------------------------------
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_R:
            self.message = "Restarted with new victim locations."
            self.restart()
        elif k == Qt.Key_C:
            i = CAMERAS.index(self.env.camera)
            self.env.camera = CAMERAS[(i + 1) % len(CAMERAS)]
            self.draw()
        elif k == Qt.Key_F:
            self.follow_me = not self.follow_me
            self.env.camera = "free"
            self.message = "Free camera follows your robot." if self.follow_me else "Free camera unlocked."
        elif k == Qt.Key_H:
            self.highlights = not self.highlights
            self.draw()
        elif k == Qt.Key_L:
            self.show_rays = not self.show_rays
            self.draw()
        elif not e.isAutoRepeat():
            self.pressed.add(k)

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat():
            self.pressed.discard(e.key())

    def learner_action(self):
        vx = (Qt.Key_Right in self.pressed) - (Qt.Key_Left in self.pressed)
        vy = (Qt.Key_Up in self.pressed) - (Qt.Key_Down in self.pressed)
        return direction_to_action(np.array([vx, vy], dtype=float))

    # --- mouse: free camera ---------------------------------------------------------
    def mousePressEvent(self, e):
        self.drag = (e.button(), e.position())
        if self.env.camera != "free":
            self.env.camera = "free"

    def mouseReleaseEvent(self, e):
        self.drag = None

    def mouseMoveEvent(self, e):
        if self.drag is None:
            return
        button, last = self.drag
        dx, dy = e.position().x() - last.x(), e.position().y() - last.y()
        self.drag = (button, e.position())
        cam = self.env.free_cam
        if button == Qt.LeftButton:
            cam.azimuth -= 0.3 * dx
            cam.elevation = float(np.clip(cam.elevation - 0.3 * dy, -89, -8))
        else:
            self.follow_me = False
            az = np.radians(cam.azimuth)
            fwd, right = np.array([np.cos(az), np.sin(az)]), np.array([np.sin(az), -np.cos(az)])
            cam.lookat[:2] += (-dx * right + dy * fwd) * cam.distance * 0.0016
        self.draw()

    def wheelEvent(self, e):
        cam = self.env.free_cam
        cam.distance = float(np.clip(cam.distance * 0.9 ** (e.angleDelta().y() / 120), 4, 70))
        self.env.camera = "free"
        self.draw()

    # --- main loop ------------------------------------------------------------------
    def tick(self):
        if not self.env.agents:          # episode over: wait for R
            return
        actions = {LEARNER: self.learner_action()}
        for i, a in self.policy.act(range(1, N_ROBOTS)).items():
            actions[f"robot_{i}"] = a
        _, _, _, _, infos = self.env.step(actions)

        for kind, robot, victim in infos[LEARNER]["events"]:
            self.message = self.describe(kind, robot, victim)
            if kind == "report":
                self.reported_at[victim] = self.env.t
        if not self.env.agents:
            self.message = (f"Operation over - rescued {self.env.rescued_count}/{N_VICTIMS}, "
                            f"lost {self.env.lost_count}. Press R to start again.")
        if self.follow_me:
            self.env.free_cam.lookat[:2] = self.env.robot_positions()[0]
        self.draw()

    def describe(self, kind, robot, victim):
        sev = self.env.severity[victim]
        where = self.env.victim_spot[victim].replace("_", " ")
        who = "You" if robot == 0 else f"R{robot}"
        if kind == "report":
            return f"NEW: search team reports a {SEVERITY_NAMES[sev]} victim ({where})."
        if kind == "pickup":
            return f"{who} picked up a {SEVERITY_NAMES[sev]} victim ({where})."
        if kind == "rescue":
            return f"{who} delivered a {SEVERITY_NAMES[sev]} victim to triage (+{R_RESCUE * sev:.0f})."
        return f"A {SEVERITY_NAMES[sev]} victim ({where}) was lost - nobody reached them in time."

    # --- drawing --------------------------------------------------------------------
    def draw(self):
        frame = np.ascontiguousarray(self.env.render())
        img = QImage(frame.data, W, H, 3 * W, QImage.Format_RGB888).copy()
        if self.highlights:
            self.draw_overlay(img)
        self.image.setPixmap(QPixmap.fromImage(img))
        env = self.env
        time_left = (MAX_STEPS - env.t) * DT
        carrying = "carrying a victim" if env.carrying[0] >= 0 else "empty"
        known = int(np.sum(env.status != UNREPORTED))
        self.status.setText(
            f"Time left: {time_left:5.1f}s   |   Team score: {env.score:6.1f}   |   Reported: {known}/{N_VICTIMS}   "
            f"Rescued: {env.rescued_count}   Lost: {env.lost_count}   |   You: {carrying}   |   "
            f"Camera: {CAMERA_LABELS[env.camera]}\n"
            f"{self.message}\n"
            f"Arrows move  ·  C camera  ·  F follow  ·  mouse rotate/pan/zoom  ·  H highlights  ·  "
            f"L sensors  ·  R restart"
        )

    def label(self, p, x, y, text, colour):
        """Rounded label with a coloured edge, centred above (x, y)."""
        font = QFont("Segoe UI", 9, QFont.Bold)
        p.setFont(font)
        tw = p.fontMetrics().horizontalAdvance(text) + 14
        rect = QRectF(x - tw / 2, y - 22, tw, 20)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(20, 20, 24, 200))
        p.drawRoundedRect(rect, 5, 5)
        p.setBrush(colour)
        p.drawRoundedRect(QRectF(rect.left(), rect.top(), 4, rect.height()), 2, 2)
        p.setPen(QColor(250, 250, 250))
        p.drawText(rect.adjusted(4, 0, 0, 0), Qt.AlignCenter, text)

    def draw_overlay(self, img):
        env = self.env
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        blink = (env.t // 6) % 2 == 0

        # your robot's distance sensors (what an RL agent "sees" of obstacles)
        if self.show_rays:
            p0 = env.robot_positions()[0]
            dists = env.lidar(0)
            ends = [[p0[0] + np.cos(a) * d, p0[1] + np.sin(a) * d, 0.3] for a, d in zip(RAY_ANGLES, dists)]
            pts = env.project([[*p0, 0.3]] + ends)
            (ox, oy) = pts[0]
            for (x, y), d in zip(pts[1:], dists):
                if not (np.isfinite(ox) and np.isfinite(x)):
                    continue
                near = d < RAY_RANGE
                col = QColor(255, 80, 60, 220) if d < 1.5 else (QColor(255, 200, 60, 200) if near else QColor(120, 220, 255, 120))
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(ox, oy), QPointF(x, y))
                if near:
                    p.setBrush(col)
                    p.drawEllipse(QPointF(x, y), 3, 3)

        # triage area
        (tx, ty), = env.project([[*SAFE_CENTER, 0.0]])
        if np.isfinite(tx):
            self.label(p, tx, ty, "TRIAGE", QColor(230, 40, 40))

        # victims
        for j in range(N_VICTIMS):
            s = env.status[j]
            if s not in (WAITING, LOST):
                continue
            x, y = env.victim_pos[j]
            (cx, cy), (ex, ey) = env.project([[x, y, 0.15], [x + 1.0, y, 0.15]])
            if not (np.isfinite(cx) and np.isfinite(ex)):
                continue
            r = float(np.clip(np.hypot(ex - cx, ey - cy), 14, 80))
            on_screen = 0 <= cx < W and 0 <= cy < H
            if not on_screen:          # small marker pinned to the screen edge, pointing the way
                r = 11.0
                cx, cy = float(np.clip(cx, 60, W - 60)), float(np.clip(cy, 34, H - 16))
            if s == LOST:
                p.setPen(QPen(QColor(150, 150, 150, 200), 2, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(cx, cy), r, r)
                self.label(p, cx, cy - r, "LOST", QColor(150, 150, 150))
                continue
            sev = env.severity[j]
            col = qcolour(SEVERITY_COLOURS[sev])
            urgent = env.time_left[j] < URGENT_STEPS
            p.setBrush(qcolour(SEVERITY_COLOURS[sev], 45))
            p.setPen(QPen(col, 4 if (urgent and blink) else 3))
            p.drawEllipse(QPointF(cx, cy), r, r)
            if urgent and blink:
                p.setPen(QPen(qcolour(SEVERITY_COLOURS[sev], 140), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(cx, cy), r + 7, r + 7)
            text = SEVERITY_NAMES[sev].upper()
            if sev > 1:
                text += f"  {env.time_left[j] * DT:.0f}s"
            if env.t - self.reported_at.get(j, -10 ** 6) < 80:      # flag newly reported victims for 4 s
                text = "NEW  " + text
                if blink:
                    p.setPen(QPen(QColor(255, 255, 255, 230), 3))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(QPointF(cx, cy), r + 12, r + 12)
            if not on_screen:
                text = "◀ " + text if cx < W / 2 else text + " ▶"
            self.label(p, cx, cy - r, text, col)

        # robots
        pos = env.robot_positions()
        pts = env.project([[*pos[i], 1.3] for i in range(N_ROBOTS)])
        for i in range(N_ROBOTS):
            x, y = pts[i]
            if not (np.isfinite(x) and 0 <= x < W and 0 <= y < H):
                continue
            name, col = ROBOT_LABELS[i]
            if env.carrying[i] >= 0:
                name += f" + {SEVERITY_NAMES[env.severity[env.carrying[i]]]}"
            self.label(p, x, y, name, col)
        p.end()


if __name__ == "__main__":
    app = QApplication([])
    viewer = Viewer()
    viewer.show()
    app.exec()