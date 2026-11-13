"""Disaster Response viewer (post-earthquake urban scene).

Arrow keys move your robot (purple). Drive up to a victim to pick them up, then
bring them to the medical station's triage area (red-bordered tarp with cots).
The blue robots decide which victim to go for with the trained MAPPO policy
(agents/checkpoints/mappo_disaster.pt) if it exists, otherwise with the greedy
rule; press M to switch. Dashed lines show where each blue robot is heading.

Victims are highlighted with circles in their triage colour, labelled with
severity and time left; robots are labelled YOU / R1 / R2.

The panel on the right is the learner model: from the choices you make while
playing, it estimates how well you understand five RL ideas (Bayesian
Knowledge Tracing) and explains each decision it judged. It persists across
restarts (R); press K to start a fresh learner profile.

Mouse (free camera): left-drag = rotate, right-drag = pan, wheel = zoom.
Keys: arrows = move, C = next camera, F = free camera follows your robot,
      H = hide/show highlights, L = show your robot's distance sensors,
      M = switch blue robots between MAPPO and the greedy rule, R = restart,
      K = reset your learner profile.
Run from the project root:  python envs/view_disaster.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from agents.heuristic_policy import direction_to_action
from agents.hrl import WAIT, HierarchicalController, MappoChooser, greedy_choice
from learner.learner_model import CONCEPTS, LearnerModel
from envs.disaster_env import (CAMERAS, CARRIED, DT, LEARNER, LOST, MAX_STEPS, N_ROBOTS, N_VICTIMS,
                               R_RESCUE, RAY_ANGLES, RAY_RANGE, SAFE_CENTER, SEVERITY_COLOURS,
                               SEVERITY_NAMES, UNREPORTED, URGENT_STEPS, WAITING, DisasterResponseEnv)

W, H = 1024, 640
PANEL_W = 300
STEP_MS = int(1000 * DT)          # real-time playback
CAMERA_LABELS = {"free": "Free (mouse)", "overview": "Overview", "isometric": "Isometric",
                 "follow": "Follow your robot", "triage": "Medical station",
                 "ruins": "Collapsed houses (NW)", "collapse": "Collapse site (NE/E)"}
CHECKPOINT = os.path.join(ROOT, "agents", "checkpoints", "mappo_disaster.pt")
ROBOT_LABELS = {0: ("YOU", QColor(170, 90, 240)), 1: ("R1", QColor(40, 130, 245)), 2: ("R2", QColor(40, 130, 245))}


class LearnerPanel(QWidget):
    """Draws the learner model's current estimate: one bar per concept, decision quality, last feedback."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(PANEL_W, H)
        self.summary, self.feedback = None, None

    def update_state(self, summary, feedback):
        self.summary, self.feedback = summary, feedback
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(28, 30, 36))
        p.setPen(QColor(240, 240, 240))
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(QRectF(16, 12, PANEL_W - 32, 24), Qt.AlignLeft, "Your understanding")
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(160, 165, 175))
        p.drawText(QRectF(16, 36, PANEL_W - 32, 30), Qt.TextWordWrap,
                   "Estimated from your choices in the scene (Bayesian Knowledge Tracing).")
        if self.summary is None:
            return
        y = 74
        for c, label in CONCEPTS.items():
            m = self.summary["mastery"][c]
            good, n = self.summary["evidence"][c]
            p.setPen(QColor(225, 225, 230))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(16, y, PANEL_W - 32, 18), Qt.AlignLeft, label)
            y += 20
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(55, 58, 66))
            p.drawRoundedRect(QRectF(16, y, PANEL_W - 90, 12), 6, 6)
            col = QColor(220, 70, 60) if m < 0.4 else (QColor(235, 180, 50) if m < 0.75 else QColor(70, 190, 100))
            if n == 0:
                col = QColor(95, 98, 108)          # not tested yet
            p.setBrush(col)
            p.drawRoundedRect(QRectF(16, y, max(12.0, (PANEL_W - 90) * m), 12), 6, 6)
            p.setPen(QColor(200, 200, 205))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRectF(PANEL_W - 70, y - 3, 60, 18), Qt.AlignRight,
                       f"{m:.0%}  ({good}/{n})" if n else "not tested")
            y += 26
        q = self.summary["quality"]
        p.setPen(QColor(225, 225, 230))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(16, y + 4, PANEL_W - 32, 18), Qt.AlignLeft,
                   f"Agreement with the trained agent: {q:.0%}" if q is not None else "Agreement with the trained agent: -")
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(160, 165, 175))
        p.drawText(QRectF(16, y + 24, PANEL_W - 32, 18), Qt.AlignLeft,
                   f"Decisions assessed: {self.summary['decisions']}   (K = new profile)")
        y += 52
        p.setPen(QColor(240, 240, 240))
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.drawText(QRectF(16, y, PANEL_W - 32, 20), Qt.AlignLeft, "Your last decision")
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(215, 215, 220))
        text = self.feedback or "Pick up a victim: your choice will be explained here."
        p.drawText(QRectF(16, y + 24, PANEL_W - 32, H - y - 34), Qt.TextWordWrap, text)
        p.end()


def qcolour(rgb, alpha=255):
    return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha)


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.env = DisasterResponseEnv(render_mode="rgb_array", width=W, height=H, camera="free")
        self.ctrl = HierarchicalController(self.env)
        self.mappo = MappoChooser(self.env, self.ctrl, CHECKPOINT) if os.path.exists(CHECKPOINT) else None
        self.brain = "mappo" if self.mappo else "greedy"
        self.learner = LearnerModel(self.env, self.ctrl, expert=self.mappo)
        self.panel = LearnerPanel()
        self.feedback = None
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
        row = QHBoxLayout()
        row.addWidget(self.image)
        row.addWidget(self.panel)
        layout.addLayout(row)
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
        self.ctrl.reset()
        self.learner.new_episode()
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
        elif k == Qt.Key_K:
            self.learner.reset()
            self.feedback = None
            self.message = "Started a fresh learner profile."
            self.draw()
        elif k == Qt.Key_M:
            if self.mappo is None:
                self.message = "No trained MAPPO model found (agents/checkpoints/mappo_disaster.pt)."
            else:
                self.brain = "greedy" if self.brain == "mappo" else "mappo"
                self.ctrl.replan = [True] * N_ROBOTS
                self.message = f"Blue robots now use: {self.brain_name()}."
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
    def brain_name(self):
        return "MAPPO (trained)" if self.brain == "mappo" else "greedy rule"

    def choose(self, i):
        if self.brain == "mappo" and self.mappo is not None:
            return self.mappo(i)
        return greedy_choice(self.env, self.ctrl, i)

    def tick(self):
        if not self.env.agents:          # episode over: wait for R
            return
        actions = self.ctrl.actions(self.choose, teammate=None, learner_action=self.learner_action())
        _, _, _, _, infos = self.env.step(actions)
        self.ctrl.notify(infos[LEARNER]["events"])
        n_before = len(self.learner.decisions)
        self.learner.observe(infos[LEARNER]["events"])
        if len(self.learner.decisions) > n_before:
            self.feedback = self.learner.last_feedback()

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
        self.panel.update_state(self.learner.summary(), self.feedback)
        env = self.env
        time_left = (MAX_STEPS - env.t) * DT
        carrying = "carrying a victim" if env.carrying[0] >= 0 else "empty"
        known = int(np.sum(env.status != UNREPORTED))
        self.status.setText(
            f"Time left: {time_left:5.1f}s   |   Team score: {env.score:6.1f}   |   Reported: {known}/{N_VICTIMS}   "
            f"Rescued: {env.rescued_count}   Lost: {env.lost_count}   |   You: {carrying}   |   "
            f"Camera: {CAMERA_LABELS[env.camera]}   |   Blue robots: {self.brain_name()} (M)\n"
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

        # where each blue robot is heading (its current decision)
        pos = env.robot_positions()
        for i in (1, 2):
            tgt = self.ctrl.target[i]
            if env.carrying[i] >= 0:
                goal = SAFE_CENTER
            elif tgt is not None and env.status[tgt] == WAITING:
                goal = env.victim_pos[tgt]
            else:
                continue
            (ax, ay), (bx, by) = env.project([[*pos[i], 0.4], [*goal, 0.15]])
            if np.isfinite(ax) and np.isfinite(bx):
                p.setPen(QPen(QColor(40, 130, 245, 200), 2, Qt.DashLine))
                p.drawLine(QPointF(ax, ay), QPointF(bx, by))

        # robots
        pts = env.project([[*pos[i], 1.3] for i in range(N_ROBOTS)])
        for i in range(N_ROBOTS):
            x, y = pts[i]
            if not (np.isfinite(x) and 0 <= x < W and 0 <= y < H):
                continue
            name, col = ROBOT_LABELS[i]
            if env.carrying[i] >= 0:
                name += f" + {SEVERITY_NAMES[env.severity[env.carrying[i]]]}"
            elif i > 0 and self.ctrl.target[i] is None and env.status[env.status == WAITING].size:
                name += " (waiting)"
            self.label(p, x, y, name, col)
        p.end()


if __name__ == "__main__":
    app = QApplication([])
    viewer = Viewer()
    viewer.show()
    app.exec()
