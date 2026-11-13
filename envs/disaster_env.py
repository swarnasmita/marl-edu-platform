"""Disaster Response multi-agent environment (PettingZoo ParallelEnv).

Post-earthquake urban neighbourhood (see scene_builder.py). Three stretcher
robots share one team objective: reach victims, carry them to the medical
station (triage area) and prioritise by severity before critical victims run
out of time. robot_0 is the learner's slot (a human in the app, a simulated
human or an RL agent during training); robot_1 and robot_2 are the RL agents.

Task: 10 victims. 6 are known at the start; 4 more are reported at random
times during the operation (found by search teams), so plans have to change
mid-episode. Critical victims must be reached (picked up) within 25 s of being reported,
moderate within 45 s; once on a stretcher their timer stops. Robots slow down
when carrying someone.

RL interface: 9 discrete actions per robot (stay / 8 directions). Observation
per robot (90 numbers): own state, triage direction, 6 features per victim,
teammates, time left, and 16 distance-sensor rays (a 2-D lidar) so a learned
policy can see obstacles. The global state for a centralised critic is the
three observations concatenated. The reward is shared by the team.
"""
import os

import mujoco
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

XML_PATH = os.path.join(os.path.dirname(__file__), "disaster_response.xml")

N_ROBOTS = 3
N_VICTIMS = 10
N_INITIAL = 6              # victims known at the start; the rest are reported during the episode
REPORT_WINDOW = (200, 1400)   # steps (10 s - 70 s) in which the late victims are reported
SEVERITY_MIX = [3, 3, 3, 2, 2, 2, 2, 1, 1, 1]
LEARNER = "robot_0"

# --- timing -----------------------------------------------------------------
SUBSTEPS = 5               # physics steps per env step -> 0.05 s per decision
DT = SUBSTEPS * 0.01
MAX_STEPS = 3000           # 150 s episode

# --- geometry ---------------------------------------------------------------
SPEED = 2.0                # robot speed when empty, m/s
CARRY_SPEED = 1.2          # slower when carrying a casualty on the stretcher
PICKUP_RADIUS = 1.0        # robot centre to victim (hips) distance for pickup
ARENA = 16.0               # half-width of the rescue site, used to normalise positions

# The triage area (safe zone) is read from the scene so the XML stays the single source of truth.
_scene = mujoco.MjModel.from_xml_path(XML_PATH)
_sz = _scene.geom("safe_zone").id
SAFE_CENTER = _scene.geom_pos[_sz][:2].copy()
SAFE_MIN = SAFE_CENTER - _scene.geom_size[_sz][:2]
SAFE_MAX = SAFE_CENTER + _scene.geom_size[_sz][:2]
del _scene, _sz

# --- victims ----------------------------------------------------------------
# severity: 3 = critical, 2 = moderate, 1 = minor; colours follow START triage tags
SEVERITY_NAMES = {3: "critical", 2: "moderate", 1: "minor"}
SEVERITY_COLOURS = {3: (0.9, 0.08, 0.08), 2: (1.0, 0.78, 0.1), 1: (0.15, 0.75, 0.25)}
LOST_COLOUR = (0.3, 0.3, 0.3)
RING_ALPHA = 0.4
URGENT_STEPS = 200         # beacon blinks when a victim has less than 10 s left
VICTIM_Z_GROUND = 0.15     # hips height when lying or sitting on the ground
VICTIM_Z_CARRIED = 0.62    # lying on a robot's stretcher
TIME_LIMIT = {3: 500, 2: 900, 1: 10 * MAX_STEPS}   # steps after being reported before a victim is lost
WAITING, CARRIED, RESCUED, LOST, UNREPORTED = 0, 1, 2, 3, 4

# --- rewards (shared by the whole team) ------------------------------------
R_RESCUE = 10.0            # x severity
R_PICKUP = 1.0
R_LOST = -5.0              # x severity
R_STEP = -0.01

# --- actions: 0 = stay, 1..8 = move in one of 8 directions -----------------
DIRS = np.array([[0.0, 0.0]] + [[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4])

# --- distance sensors (2-D lidar) ---------------------------------------------
N_RAYS = 16
RAY_RANGE = 8.0            # metres
RAY_ANGLES = np.arange(N_RAYS) * 2 * np.pi / N_RAYS    # 0 = east, counter-clockwise

OBS_DIM = 7 + 6 * N_VICTIMS + 3 * (N_ROBOTS - 1) + 1 + N_RAYS

CAMERAS = ["free", "overview", "isometric", "follow", "triage", "ruins", "collapse"]
TURN_RATE = 0.25           # how quickly a robot's body turns toward its direction of travel


class DisasterResponseEnv(ParallelEnv):
    metadata = {"name": "disaster_response_v2", "render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, width=1280, height=800, camera="overview", observe=True):
        self.observe = observe          # False skips building observations (hierarchical training is faster)
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None
        self._size = (height, width)
        self.camera = camera

        self.possible_agents = [f"robot_{i}" for i in range(N_ROBOTS)]
        self.agents = []
        self.np_random = np.random.default_rng()

        m = self.model
        self._robot_body = [m.body(f"robot_{i}").id for i in range(N_ROBOTS)]
        self._robot_qpos = [(m.joint(f"r{i}_x").qposadr[0], m.joint(f"r{i}_y").qposadr[0]) for i in range(N_ROBOTS)]
        self._robot_dof = [(m.joint(f"r{i}_x").dofadr[0], m.joint(f"r{i}_y").dofadr[0]) for i in range(N_ROBOTS)]
        self.robot_radius = float(m.geom_size[m.geom("robot_0_col").id][0])
        self.robot_half_size = np.array([self.robot_radius, self.robot_radius])
        self._robot_vis = [m.body(f"robot_{i}_vis").mocapid[0] for i in range(N_ROBOTS)]

        # free camera (mouse controlled in the viewer)
        self.free_cam = mujoco.MjvCamera()
        self.free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.free_cam.lookat[:] = (0.0, -1.0, 0.0)
        self.free_cam.distance, self.free_cam.azimuth, self.free_cam.elevation = 30.0, 90.0, -58.0

        # victims: a lying pose and a sitting pose, toggled by making geoms (in)visible
        self._victim_mocap, self._pose, self._tag = [], [], []
        self._ring, self._beacon = [], []
        names = [m.geom(g).name for g in range(m.ngeom)]
        for j in range(N_VICTIMS):
            self._victim_mocap.append(m.body(f"victim_{j}").mocapid[0])
            pose = {}
            for p in ("lie", "sit"):
                ids = [g for g, n in enumerate(names) if n.startswith(f"victim_{j}_{p}_") and not n.endswith("tag")]
                pose[p] = (ids, m.geom_rgba[ids].copy())
            self._pose.append(pose)
            self._tag.append({p: m.geom(f"victim_{j}_{p}_tag").id for p in ("lie", "sit")})
            self._ring.append(m.geom(f"victim_{j}_ring").id)
            self._beacon.append(m.geom(f"victim_{j}_beacon").id)

        # victim spawn points (by situation) and rescue cots, defined as sites in the scene
        self.spawn_names, spawn = [], []
        slots = {}
        for s in range(m.nsite):
            n = m.site(s).name
            if n.startswith("spawn_"):
                self.spawn_names.append(n[len("spawn_"):])
                q = m.site_quat[s]
                spawn.append((*m.site_pos[s][:2], 2 * np.arctan2(q[3], q[0])))
            elif n.startswith("rescue_slot_"):
                slots[int(n.rsplit("_", 1)[1])] = m.site_pos[s].copy()
        self.spawn_points = np.array(spawn)
        self.rescue_slots = np.array([slots[k] for k in sorted(slots)])

        # lidar: rays hit only collision geometry (group 3): buildings, rubble, vehicles, people, robots
        self._ray_vec = np.stack([np.cos(RAY_ANGLES), np.sin(RAY_ANGLES), np.zeros(N_RAYS)], 1).ravel()
        self._ray_group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
        self._ray_gid = np.zeros(N_RAYS, dtype=np.int32)
        self._ray_dist = np.zeros(N_RAYS)

        self._obs_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self._act_space = spaces.Discrete(len(DIRS))

    # --- PettingZoo API -------------------------------------------------------
    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        self.episode_seed = seed
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.agents = list(self.possible_agents)
        self.t = 0

        # robots start beside the medical station with a little random offset
        for qx, qy in self._robot_qpos:
            self.data.qpos[[qx, qy]] = self.np_random.uniform(-0.3, 0.3, 2)

        # victims are placed at random spawn points; the last few are reported later
        idx = self.np_random.choice(len(self.spawn_points), N_VICTIMS, replace=False)
        self.victim_spot = [self.spawn_names[i] for i in idx]
        self.victim_pos = self.spawn_points[idx, :2].copy()
        self.victim_yaw = self.spawn_points[idx, 2].copy()
        self.severity = self.np_random.permutation(SEVERITY_MIX)
        self.report_time = np.zeros(N_VICTIMS, dtype=int)
        self.report_time[N_INITIAL:] = np.sort(self.np_random.integers(*REPORT_WINDOW, N_VICTIMS - N_INITIAL))
        self.status = np.where(self.report_time > 0, UNREPORTED, WAITING)
        self.time_left = np.array([TIME_LIMIT[s] for s in self.severity], dtype=float)
        self.carrier = np.full(N_VICTIMS, -1)
        self.carrying = np.full(N_ROBOTS, -1)
        self.pickup_time = np.full(N_VICTIMS, -1)     # step at which each victim was picked up (for metrics)
        self.picked_by = np.full(N_VICTIMS, -1)
        self.rescued_by = np.full(N_VICTIMS, -1)
        self.heading = np.zeros(N_ROBOTS)        # 0 = facing north (+y)
        self.rescued_count = 0
        self.lost_count = 0
        self.score = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._sync_robots()
        self._sync_victims()
        mujoco.mj_forward(self.model, self.data)
        obs = {a: (self._obs(i) if self.observe else None) for i, a in enumerate(self.agents)}
        return obs, {a: {"victim_spots": list(self.victim_spot)} for a in self.agents}

    def step(self, actions):
        for i, agent in enumerate(self.possible_agents):
            speed = CARRY_SPEED if self.carrying[i] >= 0 else SPEED
            self.data.ctrl[2 * i:2 * i + 2] = speed * DIRS[actions.get(agent, 0)]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.t += 1

        reward = R_STEP
        events = []
        pos = self.robot_positions()

        for i in range(N_ROBOTS):
            if self.carrying[i] < 0:
                # pick up the nearest waiting victim within reach
                waiting = np.where(self.status == WAITING)[0]
                if len(waiting):
                    d = np.linalg.norm(self.victim_pos[waiting] - pos[i], axis=1)
                    k = int(np.argmin(d))
                    if d[k] < PICKUP_RADIUS:
                        j = waiting[k]
                        self.status[j], self.carrying[i], self.carrier[j] = CARRIED, j, i
                        self.pickup_time[j], self.picked_by[j] = self.t, i
                        reward += R_PICKUP
                        events.append(("pickup", i, j))
            elif np.all(pos[i] > SAFE_MIN) and np.all(pos[i] < SAFE_MAX):
                # deliver the carried victim to the triage area: laid on the next free cot
                j = self.carrying[i]
                self.status[j], self.carrying[i], self.carrier[j] = RESCUED, -1, -1
                self.rescued_by[j] = i
                self.victim_pos[j] = self.rescue_slots[self.rescued_count % len(self.rescue_slots)][:2]
                self.victim_yaw[j] = 0.0
                self.rescued_count += 1
                reward += R_RESCUE * self.severity[j]
                events.append(("rescue", i, j))

        # new victims reported by search teams
        for j in np.where((self.status == UNREPORTED) & (self.report_time <= self.t))[0]:
            self.status[j] = WAITING
            events.append(("report", None, j))

        self._sync_robots()
        for j in np.where(self.status == CARRIED)[0]:
            h = self.heading[self.carrier[j]]
            self.victim_pos[j] = pos[self.carrier[j]] + 0.05 * np.array([-np.sin(h), np.cos(h)])
            self.victim_yaw[j] = h          # lie along the stretcher

        # uncollected victims lose time; they are lost when it runs out
        waiting = self.status == WAITING
        self.time_left[waiting] -= 1
        for j in np.where(waiting & (self.time_left <= 0))[0]:
            self.status[j] = LOST
            self.lost_count += 1
            reward += R_LOST * self.severity[j]
            events.append(("lost", None, j))

        self._sync_victims()
        self.score += reward

        done = bool(np.all((self.status == RESCUED) | (self.status == LOST)))
        truncated = self.t >= MAX_STEPS and not done
        agents = self.agents
        obs = {a: (self._obs(self.possible_agents.index(a)) if self.observe else None) for a in agents}
        rewards = {a: reward for a in agents}
        terminations = {a: done for a in agents}
        truncations = {a: truncated for a in agents}
        infos = {a: {"events": events} for a in agents}
        if done or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def state(self):
        """Global state for a centralised critic (e.g. MAPPO)."""
        return np.concatenate([self._obs(i) for i in range(N_ROBOTS)])

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self._size)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True     # atmospheric dust haze
        cam = self.free_cam if self.camera == "free" else self.camera
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def project(self, points):
        """Pixel coordinates (u, v) of 3-D world points in the last rendered frame.
        Points behind the camera get NaN. Used by the viewer to draw overlays."""
        cam = self._renderer.scene.camera[0]
        pos, fwd, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
        right = np.cross(fwd, up)
        h, w = self._size
        top, bottom, near = cam.frustum_top, cam.frustum_bottom, cam.frustum_near
        half_w = (top - bottom) / 2 * w / h
        out = []
        for p in np.atleast_2d(points):
            d = np.asarray(p, dtype=float) - pos
            z = d @ fwd
            if z <= 0.1:
                out.append((np.nan, np.nan))
                continue
            xs, ys = near * (d @ right) / z, near * (d @ up) / z
            out.append(((xs + half_w) / (2 * half_w) * w, (top - ys) / (top - bottom) * h))
        return np.array(out)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- helpers ----------------------------------------------------------------
    def robot_positions(self):
        return np.array([self.data.xpos[b][:2] for b in self._robot_body])

    def robot_velocities(self):
        return np.array([self.data.qvel[[dx, dy]] for dx, dy in self._robot_dof])

    def lidar(self, i):
        """Distances (m) along N_RAYS horizontal rays from robot i's centre, capped at RAY_RANGE."""
        b = self._robot_body[i]
        mujoco.mj_multiRay(self.model, self.data, self.data.xpos[b], self._ray_vec, self._ray_group, True, b,
                           self._ray_gid, self._ray_dist, None, N_RAYS, RAY_RANGE)
        d = self._ray_dist.copy()
        d[(d < 0) | (d > RAY_RANGE)] = RAY_RANGE
        return d

    def obstacle_boxes(self):
        """Static obstacles as (cx, cy, half_x, half_y): every axis-aligned box collider in the world body."""
        m = self.model
        out = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0 and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                out.append((*m.geom_pos[g][:2], *m.geom_size[g][:2]))
        return np.array(out)

    def _obs(self, i):
        pos, vel = self.robot_positions(), self.robot_velocities()
        p = pos[i]
        parts = [p / ARENA, vel[i] / SPEED, [float(self.carrying[i] >= 0)], (SAFE_CENTER - p) / ARENA]
        for j in range(N_VICTIMS):
            s = self.status[j]
            if s == UNREPORTED:             # not known yet: nothing to observe
                parts.append(np.zeros(6))
                continue
            parts.append((self.victim_pos[j] - p) / ARENA)
            parts.append([self.severity[j] / 3, float(s == WAITING), float(s == CARRIED),
                          min(max(self.time_left[j], 0) / MAX_STEPS, 1.0)])
        for k in range(N_ROBOTS):
            if k != i:
                parts.append((pos[k] - p) / ARENA)
                parts.append([float(self.carrying[k] >= 0)])
        parts.append([1 - self.t / MAX_STEPS])
        parts.append(self.lidar(i) / RAY_RANGE)
        return np.concatenate(parts).astype(np.float32)

    def _sync_robots(self):
        """Turn each robot's visual model smoothly toward its direction of travel."""
        pos, vel = self.robot_positions(), self.robot_velocities()
        for i in range(N_ROBOTS):
            if np.linalg.norm(vel[i]) > 0.3:
                target = np.arctan2(vel[i][1], vel[i][0]) - np.pi / 2
                diff = (target - self.heading[i] + np.pi) % (2 * np.pi) - np.pi
                self.heading[i] += TURN_RATE * diff
            h = self.heading[i]
            self.data.mocap_pos[self._robot_vis[i]] = [*pos[i], 0.3]
            self.data.mocap_quat[self._robot_vis[i]] = [np.cos(h / 2), 0.0, 0.0, np.sin(h / 2)]

    def _sync_victims(self):
        m = self.model
        for j in range(N_VICTIMS):
            s = self.status[j]
            mid = self._victim_mocap[j]
            if s == CARRIED:
                z = VICTIM_Z_CARRIED
            elif s == RESCUED:
                z = self.rescue_slots[0][2]
            else:
                z = VICTIM_Z_GROUND
            self.data.mocap_pos[mid] = [*self.victim_pos[j], z]
            yaw = self.victim_yaw[j]
            self.data.mocap_quat[mid] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

            # minor injuries sit up while waiting; everyone else lies down; unreported = not drawn
            shown = None if s == UNREPORTED else ("sit" if (s == WAITING and self.severity[j] == 1) else "lie")
            for p in ("lie", "sit"):
                ids, base = self._pose[j][p]
                rgba = base.copy()
                if p != shown:
                    rgba[:, 3] = 0.0
                m.geom_rgba[ids] = rgba

            colour = LOST_COLOUR if s == LOST else SEVERITY_COLOURS[self.severity[j]]
            for p in ("lie", "sit"):
                m.geom_rgba[self._tag[j][p]] = (*colour, 1.0 if p == shown else 0.0)
            if shown is None:
                m.geom_rgba[self._ring[j]] = (*colour, 0.0)
                m.geom_rgba[self._beacon[j]] = (*colour, 0.0)
                continue
            ring_alpha = RING_ALPHA if s in (WAITING, LOST) else 0.0
            beacon_alpha = 1.0 if s in (WAITING, CARRIED) else 0.0
            if s == WAITING and self.time_left[j] < URGENT_STEPS and (self.t // 6) % 2:
                beacon_alpha = 0.15     # blink: running out of time
            m.geom_rgba[self._ring[j]] = (*colour, ring_alpha)
            m.geom_rgba[self._beacon[j]] = (*colour, beacon_alpha)
