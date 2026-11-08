"""Greedy rule-based policy with path planning, for the urban Disaster Response env.

Not RL. Each robot picks the waiting victim with the best severity / path-length
ratio, drives there along the shortest obstacle-free route, and carries the
victim to the triage area. Every step it tries all 8 moves, rejects those that
would run into an obstacle, steers away from other robots, and takes the one
that gets it closest to the goal along the planned route. Used to sanity-check
the environment and as a baseline for the MARL agents.
"""
import heapq

import numpy as np

from envs.disaster_env import ARENA, DIRS, SAFE_CENTER, WAITING

SQRT2 = np.sqrt(2.0)
NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def direction_to_action(vec):
    """Map a 2-D direction vector to the nearest of the 8 move actions (0 = stay)."""
    if np.linalg.norm(vec) < 0.05:
        return 0
    angle = np.arctan2(vec[1], vec[0])
    return int(np.round(angle / (np.pi / 4))) % 8 + 1


class GridPlanner:
    """Occupancy grid over the site. Robots never rotate, so obstacles are inflated
    by the robot's half-size along x and y separately (exact for axis-aligned boxes)."""

    def __init__(self, env, res=0.2, margin=0.06):
        self.res = res
        self.lo = -ARENA - 0.5
        n = int(np.ceil((2 * ARENA + 1.0) / res))
        self.n = n
        centres = self.lo + (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(centres, centres, indexing="ij")
        rx, ry = env.robot_half_size + margin
        free = np.ones((n, n), dtype=bool)
        for cx, cy, hx, hy in env.obstacle_boxes():
            free &= ~((np.abs(X - cx) < hx + rx) & (np.abs(Y - cy) < hy + ry))
        self.free = free
        self.centres = centres
        self._fields = {}

    def cell(self, p):
        i = int(np.clip((p[0] - self.lo) // self.res, 0, self.n - 1))
        j = int(np.clip((p[1] - self.lo) // self.res, 0, self.n - 1))
        return i, j

    def point(self, c):
        return np.array([self.centres[c[0]], self.centres[c[1]]])

    def nearest_free(self, c):
        """Closest free cell (e.g. a victim lying under debris sits in a blocked cell)."""
        if self.free[c]:
            return c
        idx = np.argwhere(self.free)
        d = np.abs(idx - np.array(c)).sum(axis=1)
        return tuple(idx[np.argmin(d)])

    def field(self, goal):
        """Distance-to-goal map (Dijkstra on the 8-connected grid), cached per goal cell."""
        g = self.nearest_free(self.cell(goal))
        if g in self._fields:
            return self._fields[g]
        dist = np.full((self.n, self.n), np.inf)
        dist[g] = 0.0
        heap = [(0.0, g)]
        free = self.free
        while heap:
            d, (i, j) = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di, dj, w in NEIGHBOURS:
                a, b = i + di, j + dj
                if 0 <= a < self.n and 0 <= b < self.n and free[a, b]:
                    if di and dj and not (free[i + di, j] and free[i, j + dj]):
                        continue        # no corner cutting
                    nd = d + w
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(heap, (nd, (a, b)))
        dist *= self.res
        self._fields[g] = dist
        return dist

    def path_length(self, p, goal):
        return self.field(goal)[self.nearest_free(self.cell(p))]

    def step_direction(self, p, goal, lookahead=4):
        """Direction from p that follows the distance field downhill a few cells ahead."""
        dist = self.field(goal)
        c = self.nearest_free(self.cell(p))
        if not np.isfinite(dist[c]):
            return np.zeros(2)
        for _ in range(lookahead):
            best, best_d = c, dist[c]
            for di, dj, _ in NEIGHBOURS:
                a, b = c[0] + di, c[1] + dj
                if 0 <= a < self.n and 0 <= b < self.n and dist[a, b] < best_d:
                    best, best_d = (a, b), dist[a, b]
            if best == c:
                break
            c = best
        target = self.point(c) if dist[c] > 0 else np.asarray(goal, dtype=float)
        return target - np.asarray(p)


class GreedyPolicy:
    PROBE = (0.35, 0.8)     # metres ahead checked for each candidate move
    PERSONAL_SPACE = 1.6    # keep this far from other robots' centres
    CHECK_EVERY = 30        # steps between progress checks
    MIN_PROGRESS = 0.3      # metres; less than this counts as stuck
    ESCAPE_STEPS = 8

    def __init__(self, env, rng=None, planner=None):
        self.env = env
        self.rng = rng or np.random.default_rng()
        self.planner = planner or GridPlanner(env)
        self.reset()

    def reset(self):
        self.anchor, self.counter, self.escape, self.prev = {}, {}, {}, {}
        self.targets = {}

    def choose_target(self, i, pos, taken):
        env, plan = self.env, self.planner
        if env.carrying[i] >= 0:
            return SAFE_CENTER
        best, best_v = None, 0.0
        for j in range(len(env.status)):
            if env.status[j] != WAITING or j in taken:
                continue
            d = plan.path_length(pos[i], env.victim_pos[j])
            if np.isfinite(d):
                v = env.severity[j] / (d + 2.0)
                if self.targets.get(i) == j:
                    v *= 1.15        # commitment: avoid switching targets back and forth
                if v > best_v:
                    best, best_v = j, v
        self.targets[i] = best
        if best is None:
            return None
        taken.add(best)
        return env.victim_pos[best]

    def best_move(self, i, pos, goal):
        """Pick the move that descends the route's distance field fastest without
        hitting obstacles or crowding other robots."""
        plan = self.planner
        dist = plan.field(goal)
        here = dist[plan.cell(pos[i])]
        if not np.isfinite(here):
            # inside an obstacle's safety margin: head straight for the nearest free cell
            c = plan.nearest_free(plan.cell(pos[i]))
            return direction_to_action(plan.point(c) - pos[i])
        if here < 0.3:
            return direction_to_action(np.asarray(goal) - pos[i])
        best_a, best_score = 0, -np.inf
        for a in range(1, len(DIRS)):
            score = 0.0
            ok = True
            for step in self.PROBE:
                q = pos[i] + DIRS[a] * step
                d = dist[plan.cell(q)]
                if not np.isfinite(d):
                    ok = False
                    break
                score = here - d            # progress along the route (metres)
            if not ok:
                continue
            q = pos[i] + DIRS[a] * self.PROBE[-1]
            for k in range(len(pos)):
                if k != i:
                    gap = np.linalg.norm(q - pos[k])
                    if gap < self.PERSONAL_SPACE:
                        score -= 1.5 * (self.PERSONAL_SPACE - gap)
            if self.prev.get(i) == a:
                score += 0.05               # smoothness: keep heading when it is as good
            if score > best_score:
                best_a, best_score = a, score
        return best_a

    def act(self, robot_ids):
        pos = self.env.robot_positions()
        taken = set()
        actions = {}
        for i in robot_ids:
            if self.escape.get(i, (0, 0))[0] > 0:
                left, a = self.escape[i]
                self.escape[i] = (left - 1, a)
                actions[i] = a
                continue
            goal = self.choose_target(i, pos, taken)
            a = 0 if goal is None else self.best_move(i, pos, goal)
            actions[i] = a
            self.prev[i] = a

            # last resort if two robots are deadlocked: a short sideways move
            self.counter[i] = self.counter.get(i, 0) + 1
            if i not in self.anchor:
                self.anchor[i] = pos[i].copy()
            if self.counter[i] >= self.CHECK_EVERY:
                if np.linalg.norm(pos[i] - self.anchor[i]) < self.MIN_PROGRESS and a != 0:
                    side = (a - 1 + self.rng.choice((2, -2))) % 8 + 1
                    self.escape[i] = (self.ESCAPE_STEPS, int(side))
                self.anchor[i] = pos[i].copy()
                self.counter[i] = 0
        return actions