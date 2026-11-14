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
        top = self.n - 1
        i = int((float(p[0]) - self.lo) // self.res)
        j = int((float(p[1]) - self.lo) // self.res)
        return min(max(i, 0), top), min(max(j, 0), top)

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
        key = self.cell(goal)
        if key in self._fields:
            return self._fields[key]
        g = self.nearest_free(key)
        if g in self._fields:
            self._fields[key] = self._fields[g]
            return self._fields[g]
        dist = self.dijkstra(self.free, g)
        self._fields[g] = self._fields[key] = dist
        return dist

    def detour_field(self, goal, blockers, me, radius=1.2):
        """Distance-to-goal map that treats the given points (parked robots) as obstacles, keeping the
        cells right around the robot itself (me) open so it can start moving. Not cached."""
        free = self.free.copy()
        X, Y = np.meshgrid(self.centres, self.centres, indexing="ij")
        for b in blockers:
            to_b = (X - b[0]) ** 2 + (Y - b[1]) ** 2
            to_me = (X - me[0]) ** 2 + (Y - me[1]) ** 2
            free &= ~((to_b < radius ** 2) & (to_b < to_me))     # block only the blocker's side
        g = self.nearest_free(self.cell(goal))
        return self.dijkstra(free, g)

    def dijkstra(self, free, g):
        dist = np.full((self.n, self.n), np.inf)
        if not free[g]:
            return dist
        dist[g] = 0.0
        heap = [(0.0, g)]
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

    def best_move(self, i, pos, goal, dist=None):
        """Pick the move that descends the route's distance field fastest without
        hitting obstacles or crowding other robots. dist: optional detour field to follow instead."""
        plan = self.planner
        if dist is None:
            dist = plan.field(goal)
        here = dist[plan.cell(pos[i])]
        if not np.isfinite(here):
            # inside an obstacle's safety margin: head straight for the nearest free cell
            c = plan.nearest_free(plan.cell(pos[i]))
            return direction_to_action(plan.point(c) - pos[i])
        if here < 0.3:
            return direction_to_action(np.asarray(goal) - pos[i])
        best_a, best_score, best_progress = 0, -np.inf, -np.inf
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
            best_progress = max(best_progress, score)
            q = pos[i] + DIRS[a] * self.PROBE[-1]
            for k in range(len(pos)):
                if k != i and self._gives_way(i, k):
                    gap = np.linalg.norm(q - pos[k])
                    if gap < self.PERSONAL_SPACE:
                        score -= 1.5 * (self.PERSONAL_SPACE - gap)
            if self.prev.get(i) == a:
                score += 0.05               # smoothness: keep heading when it is as good
            if score > best_score:
                best_a, best_score = a, score
        if best_progress <= 0.05:
            # Tight spot (narrow alley, sharp turn): no direction is clear for the whole look-ahead, or none
            # makes progress, so the robot would rock back and forth. Step to the neighbouring route cell
            # that is closest to the goal instead: Dijkstra guarantees it exists and is free.
            return self._descend(i, pos, dist, here)
        return best_a

    def _gives_way(self, i, k):
        """Right of way, so two robots meeting head-on don't both wait for each other: the human's robot
        (robot_0) first, then a robot carrying a victim, then the lower index. Robot i keeps its distance
        only from robots that have priority over it; the other one will move aside."""
        if k == 0:
            return True
        if i == 0:
            return False
        ci, ck = self.env.carrying[i] >= 0, self.env.carrying[k] >= 0
        if ci != ck:
            return ck
        return k < i

    def _descend(self, i, pos, dist, here):
        plan = self.planner
        ci, cj = plan.cell(pos[i])
        best, best_d = None, here
        for di, dj, _ in NEIGHBOURS:
            a, b = ci + di, cj + dj
            if 0 <= a < plan.n and 0 <= b < plan.n and dist[a, b] < best_d:
                if di and dj and not (plan.free[ci + di, cj] and plan.free[ci, cj + dj]):
                    continue                # no corner cutting
                best, best_d = (a, b), dist[a, b]
        if best is None:
            return 0
        # aim a little past the neighbour's centre so the robot does not stall on the cell boundary
        return direction_to_action(plan.point(best) - pos[i] + 0.5 * (plan.point(best) - plan.point((ci, cj))))

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
