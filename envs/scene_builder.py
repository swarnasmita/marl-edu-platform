"""Builds envs/disaster_response.xml: a post-earthquake urban neighbourhood.

The scene is assembled from reusable components (buildings, ruins, vehicles,
infrastructure, effects, people, robots). Run this file after editing the
layout:   python envs/scene_builder.py

Conventions
- Every physical obstacle has an invisible, axis-aligned box collider in the
  world body (class "collider"). Everything else is decoration: no contacts,
  no mass, so it costs nothing in physics, only in rendering.
- Decorative assemblies are static <body> elements with their own position and
  orientation, so their parts are written in local coordinates.
- Victim spawn points and rescue cot slots are <site>s the environment reads.

Units are metres. x = east, y = north.
"""
import math
import os
import random

N_VICTIM_MODELS = 10          # must match N_VICTIMS in disaster_env.py
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disaster_response.xml")
rng = random.Random(7)

WORLD = []   # worldbody lines
_counter = {}


def uid(prefix):
    _counter[prefix] = _counter.get(prefix, 0) + 1
    return f"{prefix}_{_counter[prefix]}"


def fmt(*vals):
    return " ".join(f"{v:.4g}" for v in vals)


def geom(gtype, pos=None, size=None, cls="deco", rgba=None, mat=None, euler=None,
         fromto=None, name=None, extra=""):
    attrs = [f'type="{gtype}"']
    if name:
        attrs.insert(0, f'name="{name}"')
    if cls:
        attrs.append(f'class="{cls}"')
    if fromto is not None:
        attrs.append(f'fromto="{fmt(*fromto)}"')
    if pos is not None:
        attrs.append(f'pos="{fmt(*pos)}"')
    if size is not None:
        attrs.append(f'size="{fmt(*size)}"')
    if euler is not None:
        attrs.append(f'euler="{fmt(*euler)}"')
    if mat:
        attrs.append(f'material="{mat}"')
    if rgba is not None:
        attrs.append(f'rgba="{fmt(*rgba)}"')
    if extra:
        attrs.append(extra)
    return f"<geom {' '.join(attrs)}/>"


def quat(yaw=0.0, pitch=0.0, roll=0.0):
    """Orientation: yaw about world z first, then pitch about the new y, then roll about the new x."""
    def q(axis, deg):
        a = math.radians(deg) / 2
        v = [0.0, 0.0, 0.0]
        v[axis] = math.sin(a)
        return (math.cos(a), *v)

    def mul(p, r):
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = r
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    return mul(mul(q(2, yaw), q(1, pitch)), q(0, roll))


class Body:
    """A static decorative assembly with its own frame (yaw, then pitch, then roll)."""

    def __init__(self, pos, yaw=0.0, pitch=0.0, roll=0.0, name=None):
        self.pos, self.quat = pos, quat(yaw, pitch, roll)
        self.name = name
        self.lines = []

    def add(self, *args, **kw):
        self.lines.append(geom(*args, **kw))

    def emit(self, into=None):
        name = f' name="{self.name}"' if self.name else ""
        out = [f'<body{name} pos="{fmt(*self.pos)}" quat="{fmt(*self.quat)}">']
        out += ["  " + l for l in self.lines]
        out.append("</body>")
        (into if into is not None else WORLD).extend(out)


def collider(x0, x1, y0, y1, h=1.5, name=None):
    """Invisible axis-aligned obstacle the robots collide with and the planner avoids."""
    WORLD.append(geom("box", pos=((x0 + x1) / 2, (y0 + y1) / 2, h / 2),
                      size=((x1 - x0) / 2, (y1 - y0) / 2, h / 2), cls="collider",
                      name=name or uid("col")))


def rect_aabb(cx, cy, length, width, yaw_deg):
    """World AABB of a length x width rectangle (length along local x) rotated by yaw."""
    c, s = abs(math.cos(math.radians(yaw_deg))), abs(math.sin(math.radians(yaw_deg)))
    hx = c * length / 2 + s * width / 2
    hy = s * length / 2 + c * width / 2
    return cx - hx, cx + hx, cy - hy, cy + hy


# --------------------------------------------------------------------------- palette
GLASS = (0.26, 0.33, 0.39, 1)
GLASS_BROKEN = (0.05, 0.05, 0.06, 1)
DARK = (0.08, 0.08, 0.08, 1)
CRACK = (0.12, 0.11, 0.1, 1)
REBAR = (0.36, 0.22, 0.14, 1)
WOOD = (0.45, 0.32, 0.2, 1)
STEEL = (0.45, 0.46, 0.48, 1)
TIRE = (0.06, 0.06, 0.06, 1)


# =========================================================================== ground
def ground_and_roads():
    WORLD.append(geom("plane", pos=(0, 0, 0), size=(60, 60, 0.1), cls=None, mat="soil", name="floor"))
    # streets: main street (E-W) and cross street (N-S); they continue beyond the site
    road = lambda x0, x1, y0, y1: WORLD.append(geom("box", pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.002),
                                                     size=((x1 - x0) / 2, (y1 - y0) / 2, 0.002), mat="asphalt"))
    road(-40, 40, -3.5, 3.5)
    road(-3.5, 3.5, -40, 40)
    road(-40, 40, 22.5, 28)       # outer ring road (scenery)
    # sidewalks with curbs
    walk = lambda x0, x1, y0, y1: WORLD.append(geom("box", pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.004),
                                                     size=((x1 - x0) / 2, (y1 - y0) / 2, 0.004), mat="paving"))
    for y0, y1 in ((3.5, 5), (-5, -3.5)):
        walk(-40, -3.5, y0, y1)
        walk(3.5, 40, y0, y1)
    for x0, x1 in ((3.5, 5), (-5, -3.5)):
        walk(x0, x1, -40, -5)
        walk(x0, x1, 5, 22.5)
    for x0, x1, y0, y1 in ((-5, -3.5, 3.5, 5), (3.5, 5, 3.5, 5), (-5, -3.5, -5, -3.5), (3.5, 5, -5, -3.5)):
        walk(x0, x1, y0, y1)
    curb = (0.62, 0.61, 0.58, 1)
    for y in (3.5, -3.5):
        for xs in ((-40, -3.5), (3.5, 40)):
            WORLD.append(geom("box", pos=(sum(xs) / 2, y, 0.012), size=((xs[1] - xs[0]) / 2, 0.08, 0.012), rgba=curb))
    for x in (3.5, -3.5):
        for ys in ((-40, -3.5), (3.5, 22.5)):
            WORLD.append(geom("box", pos=(x, sum(ys) / 2, 0.012), size=(0.08, (ys[1] - ys[0]) / 2, 0.012), rgba=curb))

    # lane markings: dashed white centre lines, solid edge lines, crosswalks
    white, yellow = (0.85, 0.85, 0.8, 1), (0.9, 0.72, 0.15, 1)
    for x in [v * 3.0 for v in range(-13, 14)]:
        if abs(x) > 5.5:
            WORLD.append(geom("box", pos=(x, 0, 0.005), size=(0.8, 0.06, 0.001), rgba=yellow))
            WORLD.append(geom("box", pos=(0, x, 0.005), size=(0.06, 0.8, 0.001), rgba=yellow))
    for k in range(-3, 4):
        off = k * 0.9
        for c in (5.5, -5.5):
            WORLD.append(geom("box", pos=(c, off, 0.005), size=(0.6, 0.22, 0.001), rgba=white))
            WORLD.append(geom("box", pos=(off, c, 0.005), size=(0.22, 0.6, 0.001), rgba=white))


def crack(x, y, length, angle, width=0.035):
    """A branching crack in asphalt or pavement."""
    WORLD.append(geom("box", pos=(x, y, 0.0065), size=(length / 2, width, 0.001), euler=(0, 0, angle), rgba=CRACK))
    a = math.radians(angle)
    for t in (-0.3, 0.25):
        bx, by = x + math.cos(a) * length * t, y + math.sin(a) * length * t
        ang = angle + rng.choice((-1, 1)) * rng.uniform(25, 50)
        bl = length * rng.uniform(0.2, 0.35)
        ba = math.radians(ang)
        WORLD.append(geom("box", pos=(bx + math.cos(ba) * bl / 2, by + math.sin(ba) * bl / 2, 0.0065),
                          size=(bl / 2, width * 0.7, 0.001), euler=(0, 0, ang), rgba=CRACK))


def pothole(x, y, r):
    WORLD.append(geom("cylinder", pos=(x, y, 0.006), size=(r, 0.002), rgba=(0.1, 0.09, 0.08, 1)))
    WORLD.append(geom("cylinder", pos=(x, y, 0.005), size=(r * 1.25, 0.002), rgba=(0.3, 0.28, 0.25, 1)))


def sinkhole(x0, x1, y0, y1):
    """Road collapse at the intersection: impassable crater with broken slabs."""
    collider(x0, x1, y0, y1, name="sinkhole")
    cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    WORLD.append(geom("cylinder", pos=(cx, cy, 0.007), size=(min(hx, hy) * 0.95, 0.002), rgba=(0.04, 0.035, 0.03, 1)))
    b = Body((cx, cy, 0))
    for k in range(9):
        a = k * 2 * math.pi / 9 + rng.uniform(-0.2, 0.2)
        r = min(hx, hy) * rng.uniform(0.75, 1.0)
        b.add("box", pos=(math.cos(a) * r, math.sin(a) * r, 0.08), size=(rng.uniform(0.35, 0.6), rng.uniform(0.25, 0.4), 0.06),
              euler=(rng.uniform(-25, 25), rng.uniform(-25, 25), math.degrees(a)), mat="asphalt")
    b.add("box", pos=(0, 0, 0.012), size=(hx * 0.4, 0.3, 0.004), euler=(0, 0, 30), rgba=(0.04, 0.04, 0.04, 1))
    b.emit()


def uplifted_slab(x, y, angle):
    """Buckled road slab (visual unevenness)."""
    WORLD.append(geom("box", pos=(x, y, 0.06), size=(0.9, 0.6, 0.05), euler=(rng.uniform(6, 10), 0, angle), mat="asphalt"))


# =========================================================================== buildings
def windows_on_face(b, face, w, d, floors, broken_p, first_floor=0, door=True):
    """Window grid on one facade. face in 'S','N','E','W'."""
    length = w if face in "SN" else d
    n = max(1, int((length - 1.0) / 2.3))
    xs = [-length / 2 + (i + 0.5) * length / n for i in range(n)]
    for f in range(first_floor, floors):
        for i, s in enumerate(xs):
            z = f * 3.0 + 1.65
            col = GLASS_BROKEN if rng.random() < broken_p else GLASS
            if f == 0 and door and i == n // 2 and face == "S":
                pos_s, size_s, z, col = s, 0.55, 1.1, (0.2, 0.16, 0.12, 1)
                hz = 1.1
            else:
                pos_s, size_s, hz = s, 0.5, 0.65
            if face == "S":
                b.add("box", pos=(pos_s, -d / 2 - 0.015, z), size=(size_s, 0.03, hz), rgba=col)
                b.add("box", pos=(pos_s, -d / 2 - 0.02, z - hz - 0.05), size=(size_s + 0.08, 0.05, 0.04), mat="concrete")
            elif face == "N":
                b.add("box", pos=(pos_s, d / 2 + 0.015, z), size=(size_s, 0.03, hz), rgba=col)
            elif face == "E":
                b.add("box", pos=(w / 2 + 0.015, pos_s, z), size=(0.03, size_s, hz), rgba=col)
            else:
                b.add("box", pos=(-w / 2 - 0.015, pos_s, z), size=(0.03, size_s, hz), rgba=col)


def facade_cracks(b, w, d, h, n):
    for _ in range(n):
        face = rng.choice("SEW")
        z = rng.uniform(0.8, h - 0.5)
        ang = rng.uniform(40, 80) * rng.choice((-1, 1))
        L = rng.uniform(0.8, 1.8)
        if face == "S":
            b.add("box", pos=(rng.uniform(-w / 2 + 0.5, w / 2 - 0.5), -d / 2 - 0.035, z), size=(L / 2, 0.01, 0.025),
                  euler=(0, ang, 0), rgba=CRACK)
        else:
            sx = w / 2 + 0.035 if face == "E" else -w / 2 - 0.035
            b.add("box", pos=(sx, rng.uniform(-d / 2 + 0.5, d / 2 - 0.5), z), size=(0.01, L / 2, 0.025),
                  euler=(ang, 0, 0), rgba=CRACK)


def rubble_pile(b, cx, cy, rx, ry, height, n, mats=("concrete", "concrete", "brick")):
    """Heap of broken slabs, blocks, beams and rebar (decoration, local coords)."""
    for _ in range(n):
        u, v = rng.uniform(-1, 1), rng.uniform(-1, 1)
        r = min(1.0, math.hypot(u, v))
        z = max(0.08, height * (1 - r) * rng.uniform(0.5, 1.0))
        sx, sy, sz = rng.uniform(0.25, 0.8), rng.uniform(0.2, 0.6), rng.uniform(0.06, 0.18)
        b.add("box", pos=(cx + u * rx, cy + v * ry, z), size=(sx, sy, sz),
              euler=(rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(0, 180)), mat=rng.choice(mats))
    for _ in range(max(2, n // 6)):
        x, y = cx + rng.uniform(-rx, rx) * 0.7, cy + rng.uniform(-ry, ry) * 0.7
        z = height * rng.uniform(0.3, 0.7)
        b.add("cylinder", fromto=(x, y, z, x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5), z + rng.uniform(0.4, 0.9)),
              size=(0.016,), rgba=REBAR)
    for _ in range(max(1, n // 8)):
        x, y = cx + rng.uniform(-rx, rx) * 0.8, cy + rng.uniform(-ry, ry) * 0.8
        b.add("box", pos=(x, y, height * rng.uniform(0.2, 0.6)), size=(rng.uniform(0.8, 1.5), 0.09, 0.09),
              euler=(rng.uniform(-20, 20), rng.uniform(-25, 25), rng.uniform(0, 180)), rgba=WOOD)


def building(x0, x1, y0, y1, floors, wall, damage, roof="roof", solid=True, name=None):
    """Residential / commercial building.

    damage 0 intact, 1 damaged (cracks, broken windows, roof debris),
    2 partial collapse (upper floor gone, hanging slab, exposed columns),
    3 collapsed (rubble mound with broken pillars)."""
    w, d = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if solid:
        collider(x0, x1, y0, y1, h=2.0, name=name)
    b = Body((cx, cy, 0))
    if damage == 3:
        rubble_pile(b, 0, 0, w / 2 * 0.95, d / 2 * 0.95, 2.4, int(w * d * 1.2) + 10)
        for sx, sy in ((-1, -1), (1, -1), (-1, 1)):
            h = rng.uniform(1.2, 3.2)
            b.add("box", pos=(sx * (w / 2 - 0.3), sy * (d / 2 - 0.3), h / 2), size=(0.17, 0.17, h / 2), mat="concrete",
                  euler=(rng.uniform(-6, 6), rng.uniform(-6, 6), 0))
            b.add("cylinder", fromto=(sx * (w / 2 - 0.3), sy * (d / 2 - 0.3), h, sx * (w / 2 - 0.3) + 0.15,
                                      sy * (d / 2 - 0.3), h + 0.5), size=(0.014,), rgba=REBAR)
        hw = rng.uniform(1.5, 2.6)
        b.add("box", pos=(-w / 4, d / 2 - 0.15, hw / 2), size=(w / 4, 0.12, hw / 2), mat=wall)
        b.emit()
        return
    h_full = 3.0 * (1 if damage == 2 else floors)
    b.add("box", pos=(0, 0, h_full / 2), size=(w / 2, d / 2, h_full / 2), mat=wall)
    b.add("box", pos=(0, 0, 0.3), size=(w / 2 + 0.03, d / 2 + 0.03, 0.3), mat="concrete")
    for f in range(1, int(h_full / 3) + 1):
        b.add("box", pos=(0, 0, f * 3.0), size=(w / 2 + 0.05, d / 2 + 0.05, 0.09), mat="concrete")
    broken = {0: 0.0, 1: 0.35, 2: 0.7}[damage]
    for face in "SNEW":
        windows_on_face(b, face, w, d, int(h_full / 3), broken)
    if damage >= 1:
        facade_cracks(b, w, d, h_full, 3 + 2 * damage)
    if damage < 2:
        # roof with parapet; a broken section on damaged buildings
        b.add("box", pos=(0, 0, h_full + 0.05), size=(w / 2 - 0.15, d / 2 - 0.15, 0.04), mat=roof)
        for sx in (-1, 1):
            b.add("box", pos=(sx * (w / 2 - 0.1), 0, h_full + 0.3), size=(0.1, d / 2, 0.3), mat="concrete")
        b.add("box", pos=(0, d / 2 - 0.1, h_full + 0.3), size=(w / 2, 0.1, 0.3), mat="concrete")
        if damage == 0:
            b.add("box", pos=(0, -d / 2 + 0.1, h_full + 0.3), size=(w / 2, 0.1, 0.3), mat="concrete")
        else:
            b.add("box", pos=(-w / 4, -d / 2 + 0.1, h_full + 0.3), size=(w / 4, 0.1, 0.3), mat="concrete")
            rubble_pile(b, w / 4, -d / 2 - 0.6, 0.9, 0.5, 0.4, 6)
            for _ in range(3):
                b.add("box", pos=(rng.uniform(-w / 3, w / 3), rng.uniform(-d / 3, d / 3), h_full + 0.15),
                      size=(0.3, 0.2, 0.1), euler=(rng.uniform(-20, 20), 0, rng.uniform(0, 90)), mat="concrete")
        b.add("box", pos=(w / 4, d / 4, h_full + 0.35), size=(0.45, 0.3, 0.3), rgba=STEEL)   # rooftop unit
    else:
        # partial collapse: exposed columns, jagged upper walls, hanging slab, debris at the base
        for k in range(floors - 1):
            z0 = 3.0 * (k + 1)
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                h = rng.uniform(0.6, 2.8)
                b.add("box", pos=(sx * (w / 2 - 0.2), sy * (d / 2 - 0.2), z0 + h / 2), size=(0.18, 0.18, h / 2), mat="concrete")
                b.add("cylinder", fromto=(sx * (w / 2 - 0.2), sy * (d / 2 - 0.2), z0 + h,
                                          sx * (w / 2 - 0.2) + 0.1, sy * (d / 2 - 0.2), z0 + h + 0.45), size=(0.013,), rgba=REBAR)
            hw = rng.uniform(1.0, 2.2)
            b.add("box", pos=(-w / 4, d / 2 - 0.12, z0 + hw / 2), size=(w / 4, 0.12, hw / 2), mat=wall)
            b.add("box", pos=(w / 2 - 0.12, d / 4, z0 + hw / 3), size=(0.12, d / 4, hw / 3), mat=wall)
            b.add("box", pos=(0, -d / 2 - 0.5, z0 - 0.9), size=(w / 2 * 0.8, 1.2, 0.09), euler=(-50, 0, rng.uniform(-6, 6)),
                  mat="concrete")
        b.add("box", pos=(0, 0, h_full + 0.06), size=(w / 2 - 0.1, d / 2 - 0.1, 0.05), mat="concrete")
        rubble_pile(b, 0, 0, w / 2 * 0.7, d / 2 * 0.7, h_full + 0.6, 14)
        rubble_pile(b, -w / 4, -d / 2 - 0.55, w / 3, 0.45, 0.5, 8)
    b.emit()


def ruin_shell(x0, x1, y0, y1, door_x0, door_x1, wall="plaster_beige", t=0.3):
    """Partially accessible collapsed building: roofless shell with a doorway.

    Walls are colliders, the interior is free space (robots can enter)."""
    h = 3.0
    pieces = [  # (x0, x1, y0, y1, height)
        (x0, x0 + t, y0, y1, rng.uniform(2.2, 3.0)),          # west
        (x0, x1, y1 - t, y1, rng.uniform(2.4, 3.2)),          # north
        (x1 - t, x1, y0, y1 - 2.0, rng.uniform(1.4, 2.2)),    # east (corner collapsed)
        (x0, door_x0, y0, y0 + t, rng.uniform(1.6, 2.4)),     # south-left
        (door_x1, x1, y0, y0 + t, rng.uniform(1.0, 1.8)),     # south-right
    ]
    for (a0, a1, b0, b1, hh) in pieces:
        collider(a0, a1, b0, b1, h=2.0)
        b = Body(((a0 + a1) / 2, (b0 + b1) / 2, 0))
        sx, sy = (a1 - a0) / 2, (b1 - b0) / 2
        # jagged top: two stacked segments of different heights
        b.add("box", pos=(0, 0, hh / 2), size=(sx, sy, hh / 2), mat=wall)
        if sx > sy:
            b.add("box", pos=(-sx / 2, 0, hh + 0.35), size=(sx / 2 * rng.uniform(0.4, 0.9), sy, 0.35), mat=wall)
        else:
            b.add("box", pos=(0, sy / 3, hh + 0.35), size=(sx, sy / 2 * rng.uniform(0.4, 0.9), 0.35), mat=wall)
        b.add("box", pos=(0, 0, 0.25), size=(sx + 0.02, sy + 0.02, 0.25), mat="concrete")
        b.emit()
    # collapsed corner (NE) is rubble + a fallen floor slab inside
    collider(x1 - 1.2, x1, y1 - 2.0, y1 - t, h=2.0)
    b = Body((x1 - 0.8, y1 - 1.1, 0))
    rubble_pile(b, 0, 0, 0.7, 0.9, 1.6, 12)
    b.add("box", pos=(-0.7, 0.0, 1.0), size=(1.1, 0.9, 0.08), euler=(0, 32, 10), mat="concrete")
    b.emit()
    # door frame (lintel) and interior details
    WORLD.append(geom("box", pos=((door_x0 + door_x1) / 2, y0 + t / 2, 2.15), size=((door_x1 - door_x0) / 2 + 0.1, t / 2, 0.12),
                      mat="concrete"))
    WORLD.append(geom("box", pos=(x0 + 1.0, y1 - 1.0, 0.012), size=(0.9, 0.6, 0.008), rgba=(0.35, 0.3, 0.26, 1)))  # rug
    b = Body((x0 + 0.9, y0 + 2.6, 0))      # toppled shelf
    b.add("box", pos=(0, 0, 0.2), size=(0.2, 0.7, 0.2), euler=(0, 70, 0), rgba=WOOD)
    b.emit()


# =========================================================================== vehicles
def wheels(b, L, W, r=0.33, axles=(-0.32, 0.32)):
    for ax in axles:
        for sy in (-1, 1):
            b.add("cylinder", pos=(ax * L, sy * (W / 2 - 0.08), r), size=(r, 0.11), euler=(90, 0, 0), rgba=TIRE)
            b.add("cylinder", pos=(ax * L, sy * (W / 2 - 0.02), r), size=(r * 0.5, 0.06), euler=(90, 0, 0), rgba=STEEL)


def car_parts(b, color, burnt=False):
    L, W = 4.2, 1.8
    body = (0.12, 0.11, 0.1, 1) if burnt else color
    b.add("box", pos=(0, 0, 0.62), size=(L / 2, W / 2, 0.3), rgba=body, mat=None if burnt else "paint")
    b.add("box", pos=(-0.25, 0, 1.15), size=(1.15, W / 2 - 0.08, 0.26), rgba=body)
    b.add("box", pos=(-0.25, 0, 1.15), size=(1.1, W / 2 - 0.05, 0.2), rgba=GLASS_BROKEN if burnt else GLASS)
    b.add("box", pos=(L / 2 + 0.01, 0, 0.6), size=(0.03, W / 2 - 0.1, 0.08), rgba=(0.2, 0.2, 0.2, 1))
    for sy in (-1, 1):
        b.add("box", pos=(L / 2, sy * 0.65, 0.72), size=(0.02, 0.15, 0.06), rgba=(0.9, 0.9, 0.8, 1))
        b.add("box", pos=(-L / 2, sy * 0.65, 0.72), size=(0.02, 0.15, 0.06), rgba=(0.6, 0.05, 0.05, 1))
    wheels(b, L, W)


def car(cx, cy, yaw, color, state="parked"):
    """state: parked, crashed, burnt, overturned (upside down)."""
    L, W = 4.2, 1.8
    x0, x1, y0, y1 = rect_aabb(cx, cy, L, W, yaw)
    collider(x0, x1, y0, y1)
    if state == "overturned":
        b = Body((cx, cy, 1.42), yaw=yaw, roll=180)
    elif state == "crashed":
        b = Body((cx, cy, 0), yaw=yaw, pitch=-3)
    else:
        b = Body((cx, cy, 0), yaw=yaw)
    car_parts(b, color, burnt=(state == "burnt"))
    if state == "crashed":
        b.add("box", pos=(L / 2 - 0.2, 0, 0.75), size=(0.3, W / 2 + 0.02, 0.12), euler=(0, 25, 0), rgba=color)
    b.emit()


def ambulance(cx, cy, yaw, collide=True):
    L, W, H = 5.6, 2.1, 2.6
    if collide:
        collider(*rect_aabb(cx, cy, L, W, yaw))
    b = Body((cx, cy, 0), yaw=yaw)
    white, red, yel = (0.94, 0.94, 0.92, 1), (0.85, 0.08, 0.08, 1), (0.95, 0.8, 0.1, 1)
    b.add("box", pos=(-0.5, 0, 1.45), size=(2.3, W / 2, 1.05), rgba=white)
    b.add("box", pos=(2.15, 0, 1.05), size=(0.65, W / 2 - 0.05, 0.65), rgba=white)
    b.add("box", pos=(2.35, 0, 1.4), size=(0.42, W / 2 - 0.02, 0.28), rgba=GLASS)
    b.add("box", pos=(0.4, 0, 1.0), size=(L / 2 + 0.01, W / 2 + 0.01, 0.12), rgba=red)
    b.add("box", pos=(0.4, 0, 0.82), size=(L / 2 + 0.01, W / 2 + 0.01, 0.05), rgba=yel)
    for sy in (-1, 1):
        b.add("box", pos=(-0.6, sy * (W / 2 + 0.01), 1.85), size=(0.35, 0.01, 0.1), rgba=red)
        b.add("box", pos=(-0.6, sy * (W / 2 + 0.01), 1.85), size=(0.1, 0.01, 0.35), rgba=red)
    b.add("box", pos=(-0.6, 0, 2.51), size=(0.35, 0.1, 0.01), rgba=red)
    b.add("box", pos=(-0.6, 0, 2.51), size=(0.1, 0.35, 0.01), rgba=red)
    b.add("box", pos=(1.75, 0.45, 2.55), size=(0.12, 0.35, 0.07), rgba=(0.95, 0.1, 0.1, 1))
    b.add("box", pos=(1.75, -0.45, 2.55), size=(0.12, 0.35, 0.07), rgba=(0.15, 0.35, 1.0, 1))
    wheels(b, L, W, r=0.38, axles=(-0.3, 0.35))
    b.emit()


def fire_truck(cx, cy, yaw):
    L, W = 8.2, 2.5
    collider(*rect_aabb(cx, cy, L, W, yaw))
    b = Body((cx, cy, 0), yaw=yaw)
    red, white = (0.75, 0.06, 0.05, 1), (0.92, 0.92, 0.9, 1)
    b.add("box", pos=(-0.8, 0, 1.55), size=(3.3, W / 2, 1.0), rgba=red, mat="paint")
    b.add("box", pos=(3.3, 0, 1.4), size=(0.8, W / 2 - 0.03, 0.95), rgba=red)
    b.add("box", pos=(3.65, 0, 1.75), size=(0.47, W / 2 - 0.02, 0.4), rgba=GLASS)
    b.add("box", pos=(0, 0, 1.05), size=(L / 2 + 0.01, W / 2 + 0.01, 0.07), rgba=white)
    for k in range(5):
        b.add("box", pos=(-3.4 + k * 1.3, 0, 1.6), size=(0.6, W / 2 + 0.015, 0.75), rgba=(0.65, 0.05, 0.04, 1))
    # ladder
    for sy in (-0.35, 0.35):
        b.add("box", pos=(-0.6, sy, 2.75), size=(3.6, 0.05, 0.06), rgba=STEEL)
    for k in range(14):
        b.add("cylinder", fromto=(-4.0 + k * 0.52, -0.35, 2.75, -4.0 + k * 0.52, 0.35, 2.75), size=(0.025,), rgba=STEEL)
    b.add("box", pos=(3.3, 0, 2.4), size=(0.12, 0.6, 0.06), rgba=(0.95, 0.15, 0.1, 1))
    wheels(b, L, W, r=0.48, axles=(-0.33, -0.18, 0.36))
    b.emit()


def rescue_truck(cx, cy, yaw):
    L, W = 6.4, 2.3
    collider(*rect_aabb(cx, cy, L, W, yaw))
    b = Body((cx, cy, 0), yaw=yaw)
    navy, orange = (0.12, 0.18, 0.3, 1), (0.95, 0.45, 0.05, 1)
    b.add("box", pos=(-0.7, 0, 1.5), size=(2.4, W / 2, 1.0), rgba=navy, mat="paint")
    b.add("box", pos=(2.3, 0, 1.25), size=(0.85, W / 2 - 0.03, 0.8), rgba=navy)
    b.add("box", pos=(2.55, 0, 1.6), size=(0.6, W / 2 - 0.02, 0.35), rgba=GLASS)
    b.add("box", pos=(0, 0, 1.15), size=(L / 2 + 0.01, W / 2 + 0.01, 0.1), rgba=orange)
    b.add("box", pos=(2.2, 0, 2.12), size=(0.12, 0.6, 0.06), rgba=(0.15, 0.35, 1.0, 1))
    wheels(b, L, W, r=0.45, axles=(-0.3, 0.33))
    b.emit()


def bus_on_side(cx, cy, yaw):
    """City bus thrown onto its side across the road."""
    L, W, H = 11.0, 2.5, 3.0
    collider(*rect_aabb(cx, cy, L, H, yaw))
    # roll 90 deg about the bus's long axis: local z (up) ends up pointing sideways,
    # so shift the body by H/2 to keep the footprint centred on (cx, cy)
    a = math.radians(yaw)
    b = Body((cx - math.sin(a) * H / 2, cy + math.cos(a) * H / 2, W / 2), yaw=yaw, roll=90)
    yellow = (0.85, 0.62, 0.12, 1)
    off = H / 2
    b.add("box", pos=(0, 0, off), size=(L / 2, W / 2, H / 2 - 0.2), rgba=yellow, mat="paint")
    b.add("box", pos=(0, W / 2 + 0.01, off + 0.35), size=(L / 2 - 0.6, 0.02, 0.55), rgba=GLASS_BROKEN)
    b.add("box", pos=(0, -W / 2 - 0.01, off + 0.35), size=(L / 2 - 0.6, 0.02, 0.55), rgba=GLASS)
    b.add("box", pos=(L / 2 + 0.01, 0, off + 0.3), size=(0.02, W / 2 - 0.1, 0.7), rgba=GLASS)
    b.add("box", pos=(0, 0, off - H / 2 + 0.25), size=(L / 2, W / 2, 0.08), rgba=(0.3, 0.3, 0.3, 1))
    for ax in (-0.33, 0.33):
        for sy in (-1, 1):
            b.add("cylinder", pos=(ax * L, sy * (W / 2 - 0.15), 0.45), size=(0.5, 0.14), euler=(90, 0, 0), rgba=TIRE)
    b.emit()


# =========================================================================== infrastructure
def streetlight(x, y, facing, state="ok"):
    """facing: yaw of the arm. state: ok, bent, fallen."""
    if state == "fallen":
        b = Body((x, y, 0.08), yaw=facing, pitch=90, roll=90)
    elif state == "bent":
        b = Body((x, y, 0), yaw=facing, roll=rng.uniform(12, 22))
    else:
        b = Body((x, y, 0), yaw=facing)
    b.add("cylinder", pos=(0, 0, 3.0), size=(0.07, 3.0), rgba=(0.3, 0.32, 0.33, 1))
    b.add("cylinder", fromto=(0, 0, 5.9, 1.2, 0, 6.1), size=(0.04,), rgba=(0.3, 0.32, 0.33, 1))
    b.add("box", pos=(1.3, 0, 6.05), size=(0.3, 0.13, 0.06), rgba=(0.25, 0.25, 0.25, 1))
    b.add("box", pos=(1.3, 0, 5.98), size=(0.25, 0.1, 0.02), rgba=(0.95, 0.9, 0.7, 1) if state == "ok" else GLASS_BROKEN)
    b.emit()


def electric_pole(x, y, state="ok", yaw=0):
    if state == "fallen":
        # snapped pole: splintered stub + the upper 5.5 m lying in direction `yaw`
        a = math.radians(yaw)
        WORLD.append(geom("cylinder", pos=(x, y, 0.6), size=(0.13, 0.6), rgba=(0.35, 0.27, 0.18, 1)))
        ex, ey = x + math.cos(a) * 5.5, y + math.sin(a) * 5.5
        WORLD.append(geom("cylinder", fromto=(x + math.cos(a) * 0.3, y + math.sin(a) * 0.3, 0.14, ex, ey, 0.14),
                          size=(0.13,), rgba=(0.35, 0.27, 0.18, 1)))
        b = Body((ex, ey, 0.1), yaw=yaw + 90, pitch=0)
        b.add("box", pos=(0, 0, 0), size=(0.9, 0.06, 0.06), rgba=(0.35, 0.27, 0.18, 1))
        b.emit()
        return
    elif state == "leaning":
        b = Body((x, y, 0), yaw=yaw, roll=14)
    else:
        b = Body((x, y, 0), yaw=yaw)
    b.add("cylinder", pos=(0, 0, 4.2), size=(0.13, 4.2), rgba=(0.35, 0.27, 0.18, 1))
    b.add("box", pos=(0, 0, 7.6), size=(0.06, 0.9, 0.06), rgba=(0.35, 0.27, 0.18, 1))
    for sy in (-0.75, -0.25, 0.25, 0.75):
        b.add("cylinder", pos=(0, sy, 7.72), size=(0.04, 0.06), rgba=(0.85, 0.85, 0.8, 1))
    b.add("cylinder", pos=(0.25, 0.4, 6.6), size=(0.22, 0.35), rgba=STEEL)       # transformer
    b.emit()


def wires(points, sag=0.6):
    """Overhead cables between pole tops (straight segments with a sag point)."""
    for (a, c) in zip(points[:-1], points[1:]):
        for dy in (-0.75, 0.75):
            mid = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2 + dy, (a[2] + c[2]) / 2 - sag)
            WORLD.append(geom("cylinder", fromto=(a[0], a[1] + dy, a[2], *mid), size=(0.012,), rgba=DARK))
            WORLD.append(geom("cylinder", fromto=(*mid, c[0], c[1] + dy, c[2]), size=(0.012,), rgba=DARK))


def jersey_barriers(x0, x1, y0, y1):
    """Row of concrete road barriers (one collider, several segments)."""
    collider(x0, x1, y0, y1, h=1.0)
    horiz = (x1 - x0) >= (y1 - y0)
    n = max(1, int(round(((x1 - x0) if horiz else (y1 - y0)) / 2.0)))
    for k in range(n):
        if horiz:
            px, py = x0 + (k + 0.5) * (x1 - x0) / n, (y0 + y1) / 2
            sz = ((x1 - x0) / n / 2 - 0.05, (y1 - y0) / 2, 0.42)
        else:
            px, py = (x0 + x1) / 2, y0 + (k + 0.5) * (y1 - y0) / n
            sz = ((x1 - x0) / 2, (y1 - y0) / n / 2 - 0.05, 0.42)
        WORLD.append(geom("box", pos=(px, py, 0.42), size=sz, mat="concrete"))
        stripe = (0.95, 0.15, 0.1, 1) if k % 2 else (0.95, 0.95, 0.92, 1)
        WORLD.append(geom("box", pos=(px, py, 0.86), size=(sz[0], sz[1], 0.03), rgba=stripe))


def cone(x, y):
    WORLD.append(geom("cylinder", pos=(x, y, 0.02), size=(0.2, 0.02), rgba=(0.1, 0.1, 0.1, 1)))
    WORLD.append(geom("capsule", fromto=(x, y, 0.05, x, y, 0.55), size=(0.08,), rgba=(0.98, 0.42, 0.05, 1)))
    WORLD.append(geom("cylinder", pos=(x, y, 0.36), size=(0.1, 0.04), rgba=(0.95, 0.95, 0.95, 1)))


def tree(x, y, burnt=False, s=1.0):
    trunk = (0.12, 0.1, 0.09, 1) if burnt else (0.35, 0.25, 0.16, 1)
    WORLD.append(geom("cylinder", pos=(x, y, 1.6 * s), size=(0.16 * s, 1.6 * s), rgba=trunk))
    if burnt:
        WORLD.append(geom("cylinder", fromto=(x, y, 2.4 * s, x + 0.9 * s, y + 0.3, 3.3 * s), size=(0.06 * s,), rgba=trunk))
        WORLD.append(geom("cylinder", fromto=(x, y, 2.0 * s, x - 0.7 * s, y - 0.4, 2.8 * s), size=(0.05 * s,), rgba=trunk))
        return
    for dx, dy, dz, r in ((0, 0, 3.6, 1.3), (0.7, 0.3, 3.1, 0.9), (-0.6, -0.4, 3.2, 0.95), (0.1, 0.5, 4.3, 0.85)):
        WORLD.append(geom("sphere", pos=(x + dx * s, y + dy * s, dz * s), size=(r * s,),
                          rgba=(0.2 + rng.uniform(0, 0.05), 0.34 + rng.uniform(0, 0.06), 0.17, 1)))


def fallen_tree(x0, x1, y, collide=True):
    if collide:
        collider(x0, x1, y - 0.3, y + 0.3, h=1.0)
    WORLD.append(geom("cylinder", fromto=(x0, y, 0.25, x1, y, 0.3), size=(0.22,), rgba=(0.35, 0.25, 0.16, 1)))
    for k in range(4):
        px = x0 + (k + 0.6) * (x1 - x0) / 4
        WORLD.append(geom("sphere", pos=(px, y + rng.uniform(-0.3, 0.3), 0.55), size=(rng.uniform(0.5, 0.8),),
                          rgba=(0.22, 0.36, 0.17, 1)))


# =========================================================================== rescue infrastructure
def tent(x0, x1, y0, y1, ridge="y", colour=(0.93, 0.93, 0.9, 1), cross=True, open_side=None, collide=True):
    """Field tent with a gabled roof. open_side: 'E','W','N','S' leaves that end open."""
    if collide:
        collider(x0, x1, y0, y1, h=2.0)
    cx, cy, w, d = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    b = Body((cx, cy, 0))
    h_wall, h_ridge = 1.6, 2.7
    across = w if ridge == "y" else d
    along = d if ridge == "y" else w
    slope = math.degrees(math.atan2(h_ridge - h_wall, across / 2))
    slab = math.hypot(across / 2, h_ridge - h_wall) / 2
    for s in (-1, 1):
        if ridge == "y":
            b.add("box", pos=(s * across / 4, 0, (h_wall + h_ridge) / 2), size=(slab + 0.05, along / 2 + 0.1, 0.03),
                  euler=(0, s * slope, 0), rgba=colour)
            b.add("box", pos=(s * (w / 2 - 0.02), 0, h_wall / 2), size=(0.02, d / 2, h_wall / 2), rgba=colour)
        else:
            b.add("box", pos=(0, s * across / 4, (h_wall + h_ridge) / 2), size=(along / 2 + 0.1, slab + 0.05, 0.03),
                  euler=(-s * slope, 0, 0), rgba=colour)
            b.add("box", pos=(0, s * (d / 2 - 0.02), h_wall / 2), size=(w / 2, 0.02, h_wall / 2), rgba=colour)
    ends = ("S", "N") if ridge == "y" else ("W", "E")
    for e in ends:
        if e == open_side:
            continue
        if ridge == "y":
            b.add("box", pos=(0, (d / 2 - 0.02) * (1 if e == "N" else -1), h_wall / 2 + 0.3),
                  size=(w / 2, 0.02, h_wall / 2 + 0.3), rgba=colour)
        else:
            b.add("box", pos=((w / 2 - 0.02) * (1 if e == "E" else -1), 0, h_wall / 2 + 0.3),
                  size=(0.02, d / 2, h_wall / 2 + 0.3), rgba=colour)
    if cross:
        red = (0.85, 0.08, 0.08, 1)
        for s in (-1, 1):
            if ridge == "y":
                b.add("box", pos=(s * across / 4, 0, (h_wall + h_ridge) / 2 + 0.035), size=(0.18, 0.55, 0.005),
                      euler=(0, s * slope, 0), rgba=red)
                b.add("box", pos=(s * across / 4, 0, (h_wall + h_ridge) / 2 + 0.035), size=(0.5, 0.18, 0.005),
                      euler=(0, s * slope, 0), rgba=red)
            else:
                b.add("box", pos=(0, s * across / 4, (h_wall + h_ridge) / 2 + 0.035), size=(0.18, 0.5, 0.005),
                      euler=(-s * slope, 0, 0), rgba=red)
                b.add("box", pos=(0, s * across / 4, (h_wall + h_ridge) / 2 + 0.035), size=(0.55, 0.18, 0.005),
                      euler=(-s * slope, 0, 0), rgba=red)
    b.emit()


def cot(x, y, slot_index):
    """Stretcher cot on the triage area; rescued victims are laid here (site = slot)."""
    b = Body((x, y, 0))
    b.add("box", pos=(0, 0, 0.42), size=(0.36, 0.98, 0.03), rgba=(0.2, 0.35, 0.55, 1))
    for sx in (-0.32, 0.32):
        for sy in (-0.88, 0.88):
            b.add("cylinder", fromto=(sx, sy, 0, sx, sy, 0.4), size=(0.02,), rgba=STEEL)
    b.emit()
    WORLD.append(f'<site name="rescue_slot_{slot_index}" pos="{fmt(x, y, 0.6)}" size="0.1" group="5" rgba="0 0 0 0"/>')


def supplies(x0, x1, y0, y1):
    """Rescue equipment and supply storage: crates, pallets, water tanks, generator."""
    collider(x0, x1, y0, y1, h=1.5)
    for k in range(8):
        x, y = rng.uniform(x0 + 0.4, x1 - 0.4), rng.uniform(y0 + 0.4, y1 - 0.4)
        h = rng.choice((0.3, 0.3, 0.6))
        WORLD.append(geom("box", pos=(x, y, h / 2 + 0.12), size=(0.4, 0.3, h / 2), euler=(0, 0, rng.choice((0, 90, 8))),
                          rgba=rng.choice(((0.55, 0.42, 0.25, 1), (0.25, 0.4, 0.25, 1), (0.6, 0.48, 0.3, 1)))))
    WORLD.append(geom("box", pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.06), size=((x1 - x0) / 2 - 0.1, (y1 - y0) / 2 - 0.1, 0.06),
                      rgba=WOOD))
    for k in range(2):
        WORLD.append(geom("cylinder", pos=(x0 + 0.55, y0 + 0.6 + k * 1.1, 0.75), size=(0.45, 0.75), rgba=(0.2, 0.45, 0.75, 1)))
    WORLD.append(geom("box", pos=(x1 - 0.5, y1 - 0.6, 0.45), size=(0.4, 0.55, 0.45), rgba=(0.85, 0.65, 0.1, 1)))


def flag(x, y, colour, cross=False):
    WORLD.append(geom("cylinder", pos=(x, y, 2.0), size=(0.03, 2.0), rgba=STEEL))
    WORLD.append(geom("box", pos=(x + 0.45, y, 3.6), size=(0.45, 0.01, 0.3), rgba=colour))
    if cross:
        WORLD.append(geom("box", pos=(x + 0.45, y + 0.012, 3.6), size=(0.08, 0.005, 0.22), rgba=(0.85, 0.08, 0.08, 1)))
        WORLD.append(geom("box", pos=(x + 0.45, y + 0.012, 3.6), size=(0.22, 0.005, 0.08), rgba=(0.85, 0.08, 0.08, 1)))


def assembly_point(x0, x1, y0, y1):
    """Evacuation assembly area: painted green zone with a sign."""
    WORLD.append(geom("box", pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.007), size=((x1 - x0) / 2, (y1 - y0) / 2, 0.001),
                      rgba=(0.15, 0.55, 0.25, 0.85)))
    for (a, b_, c, d) in ((x0, x1, y0, y0 + 0.15), (x0, x1, y1 - 0.15, y1), (x0, x0 + 0.15, y0, y1), (x1 - 0.15, x1, y0, y1)):
        WORLD.append(geom("box", pos=((a + b_) / 2, (c + d) / 2, 0.009), size=((b_ - a) / 2, (d - c) / 2, 0.001),
                          rgba=(0.92, 0.92, 0.9, 1)))
    sx, sy = x0 + 0.4, y1 - 0.4
    WORLD.append(geom("cylinder", pos=(sx, sy, 1.1), size=(0.04, 1.1), rgba=STEEL))
    WORLD.append(geom("box", pos=(sx, sy - 0.05, 2.25), size=(0.45, 0.02, 0.3), rgba=(0.1, 0.55, 0.25, 1)))
    WORLD.append(geom("box", pos=(sx, sy - 0.075, 2.25), size=(0.12, 0.005, 0.2), rgba=(0.95, 0.95, 0.95, 1)))


# =========================================================================== disaster effects
_lights = []


def fire(x, y, z=0.0, s=1.0, light=True):
    b = Body((x, y, z))
    b.add("cylinder", pos=(0, 0, 0.01), size=(0.8 * s, 0.003), rgba=(1.0, 0.4, 0.08, 0.25))
    for dx, dy, h, col in ((0, 0, 0.55, (1.0, 0.32, 0.04, 0.85)), (0.25, 0.1, 0.4, (1.0, 0.5, 0.08, 0.85)),
                           (-0.2, -0.15, 0.35, (1.0, 0.45, 0.06, 0.85)), (0.02, 0.03, 0.32, (1.0, 0.88, 0.35, 0.95))):
        b.add("ellipsoid", pos=(dx * s, dy * s, h * s * 0.9), size=(0.28 * s, 0.28 * s, h * s), rgba=col, mat="flame")
    for _ in range(5):
        b.add("box", pos=(rng.uniform(-0.6, 0.6) * s, rng.uniform(-0.6, 0.6) * s, 0.05), size=(0.25, 0.06, 0.05),
              euler=(0, 0, rng.uniform(0, 180)), rgba=(0.06, 0.05, 0.05, 1))
    b.emit()
    if light and len(_lights) < 4:
        _lights.append(f'<light pos="{fmt(x, y, z + 1.2 * s)}" diffuse="1 0.45 0.12" specular="0 0 0" '
                       f'attenuation="0.3 0.25 0.12" castshadow="false"/>')


def smoke(x, y, z, height=8.0, drift=(0.35, 0.45), dark=0.18):
    """Rising plume: many overlapping soft puffs that widen, lighten and fade with height."""
    n = 16
    for k in range(n):
        t = k / (n - 1)
        r = 0.45 + 1.5 * t
        g = dark + 0.2 * t
        jx, jy = rng.uniform(-0.4, 0.4) * (0.3 + t), rng.uniform(-0.4, 0.4) * (0.3 + t)
        WORLD.append(geom("ellipsoid", pos=(x + drift[0] * height * t * t + jx, y + drift[1] * height * t * t + jy,
                                            z + 0.6 + height * t),
                          size=(r, r, r * 0.8), rgba=(g, g, g * 0.97, 0.2 - 0.13 * t), mat="smoke"))


def dust(x, y, rx, ry):
    for _ in range(4):
        WORLD.append(geom("ellipsoid", pos=(x + rng.uniform(-rx, rx) * 0.6, y + rng.uniform(-ry, ry) * 0.6, 0.5),
                          size=(rx * rng.uniform(0.5, 0.8), ry * rng.uniform(0.5, 0.8), 0.6),
                          rgba=(0.62, 0.55, 0.45, 0.13), mat="smoke"))


def debris_field(x, y, rx, ry, n):
    """Scattered bricks, planks and rods on open ground (decoration only)."""
    for _ in range(n):
        px, py = x + rng.uniform(-rx, rx), y + rng.uniform(-ry, ry)
        kind = rng.random()
        if kind < 0.5:
            WORLD.append(geom("box", pos=(px, py, 0.05), size=(0.12, 0.06, 0.05), euler=(0, 0, rng.uniform(0, 180)),
                              mat=rng.choice(("brick", "concrete"))))
        elif kind < 0.8:
            WORLD.append(geom("box", pos=(px, py, 0.03), size=(rng.uniform(0.4, 0.9), 0.06, 0.03),
                              euler=(0, 0, rng.uniform(0, 180)), rgba=WOOD))
        else:
            a = rng.uniform(0, math.pi)
            L = rng.uniform(0.5, 1.2)
            WORLD.append(geom("cylinder", fromto=(px, py, 0.02, px + math.cos(a) * L, py + math.sin(a) * L, 0.05),
                              size=(0.015,), rgba=REBAR))


def debris_over(x, y, yaw):
    """A broken slab propped over a spawn point: a victim there is 'beneath debris'."""
    b = Body((x, y, 0), yaw=yaw)
    b.add("box", pos=(0.0, 0.55, 0.5), size=(0.9, 0.75, 0.07), euler=(-22, 0, 0), mat="concrete")
    b.add("box", pos=(0.75, 1.1, 0.25), size=(0.25, 0.2, 0.25), mat="concrete", euler=(0, 0, 20))
    b.add("box", pos=(-0.8, 1.0, 0.18), size=(0.2, 0.25, 0.18), mat="brick", euler=(0, 0, -15))
    b.add("cylinder", fromto=(-0.5, 0.6, 0.5, -0.7, 0.2, 0.95), size=(0.015,), rgba=REBAR)
    b.emit()


# =========================================================================== people
SKIN = [(0.87, 0.68, 0.55, 1), (0.55, 0.38, 0.28, 1), (0.95, 0.8, 0.68, 1), (0.7, 0.5, 0.38, 1), (0.42, 0.3, 0.22, 1)]


def responder(x, y, yaw, role):
    """Standing rescue or medical personnel. People are obstacles: robots must drive around them."""
    collider(x - 0.3, x + 0.3, y - 0.3, y + 0.3, h=1.8)
    jacket, stripe, helmet, pants = {
        "firefighter": ((0.55, 0.45, 0.25, 1), (0.85, 0.88, 0.45, 1), (0.95, 0.8, 0.1, 1), (0.5, 0.42, 0.24, 1)),
        "paramedic": ((0.12, 0.42, 0.25, 1), (0.85, 0.88, 0.45, 1), (0.95, 0.95, 0.95, 1), (0.15, 0.15, 0.2, 1)),
        "rescue": ((0.95, 0.45, 0.05, 1), (0.85, 0.85, 0.85, 1), (0.95, 0.45, 0.05, 1), (0.2, 0.22, 0.3, 1)),
        "police": ((0.1, 0.14, 0.25, 1), (0.85, 0.95, 0.1, 1), (0.1, 0.14, 0.25, 1), (0.1, 0.12, 0.2, 1)),
    }[role]
    b = Body((x, y, 0), yaw=yaw)
    for sy in (-0.1, 0.1):
        b.add("capsule", fromto=(0, sy, 0.1, 0, sy, 0.85), size=(0.075,), rgba=pants)
        b.add("box", pos=(0.05, sy, 0.05), size=(0.12, 0.06, 0.05), rgba=DARK)
    b.add("capsule", fromto=(0, 0, 0.98, 0, 0, 1.38), size=(0.17,), rgba=jacket)
    for z in (1.05, 1.22):
        b.add("cylinder", pos=(0, 0, z), size=(0.178, 0.022), rgba=stripe)
    b.add("sphere", pos=(0, 0, 1.68), size=(0.105,), rgba=SKIN[rng.randrange(len(SKIN))])
    b.add("sphere", pos=(0, 0, 1.73), size=(0.12,), rgba=helmet)
    b.add("cylinder", pos=(0, 0, 1.7), size=(0.15, 0.012), rgba=helmet)
    b.add("capsule", fromto=(0, -0.25, 1.38, 0.15, -0.28, 1.0), size=(0.055,), rgba=jacket)
    b.add("capsule", fromto=(0, 0.25, 1.38, 0.12, 0.3, 0.98), size=(0.055,), rgba=jacket)
    b.emit()


def victim(j, pos, shirt, pants, skin):
    """Mocap victim with a lying pose and a sitting pose; the env shows one at a time.

    Body origin is at the hips, 0.15 m above the ground. The person lies along +y (head)."""
    out = [f'<body name="victim_{j}" mocap="true" pos="{fmt(pos[0], pos[1], 0.15)}">']
    k = 0

    def part(pose, *args, **kw):
        nonlocal k
        k += 1
        out.append("  " + geom(*args, cls="person", name=f"victim_{j}_{pose}_{k}", **kw))

    # lying
    part("lie", "capsule", fromto=(0, -0.28, 0, 0, 0.2, 0), size=(0.16,), rgba=shirt)
    part("lie", "sphere", pos=(0, 0.47, 0.0), size=(0.11,), rgba=skin)
    for s in (-1, 1):
        part("lie", "capsule", fromto=(s * 0.09, -0.36, -0.03, s * 0.12, -1.02, -0.05), size=(0.08,), rgba=pants)
        part("lie", "box", pos=(s * 0.12, -1.1, 0.0), size=(0.05, 0.05, 0.09), rgba=DARK)
        part("lie", "capsule", fromto=(s * 0.23, 0.15, -0.03, s * 0.27, -0.32, -0.05), size=(0.055,), rgba=shirt)
        part("lie", "sphere", pos=(s * 0.27, -0.4, -0.05), size=(0.05,), rgba=skin)
    out.append("  " + geom("box", pos=(0.08, 0.05, 0.165), size=(0.05, 0.07, 0.004), cls="person", name=f"victim_{j}_lie_tag"))
    # sitting (walking wounded): back upright, legs stretched forward (-y)
    for s in (-1, 1):
        part("sit", "capsule", fromto=(s * 0.1, -0.05, -0.05, s * 0.12, -0.8, -0.06), size=(0.08,), rgba=pants)
        part("sit", "box", pos=(s * 0.12, -0.86, 0.02), size=(0.05, 0.05, 0.09), rgba=DARK)
        part("sit", "capsule", fromto=(s * 0.22, 0.08, 0.42, s * 0.24, -0.25, 0.15), size=(0.055,), rgba=shirt)
        part("sit", "sphere", pos=(s * 0.24, -0.3, 0.12), size=(0.05,), rgba=skin)
    part("sit", "capsule", fromto=(0, 0.04, 0.05, 0, 0.1, 0.5), size=(0.16,), rgba=shirt)
    part("sit", "sphere", pos=(0, 0.12, 0.76), size=(0.11,), rgba=skin)
    out.append("  " + geom("box", pos=(0.08, -0.12, 0.4), size=(0.05, 0.004, 0.07), cls="person", name=f"victim_{j}_sit_tag"))
    # severity ring and beacon (coloured at runtime)
    out.append("  " + geom("cylinder", pos=(0, -0.3, -0.142), size=(0.95, 0.003), cls="person", name=f"victim_{j}_ring"))
    out.append("  " + geom("sphere", pos=(0, 0, 1.25), size=(0.1,), cls="person", name=f"victim_{j}_beacon"))
    out.append("</body>")
    WORLD.extend(out)


def spawn(name, x, y, yaw):
    WORLD.append(f'<site name="spawn_{name}" pos="{fmt(x, y, 0.15)}" euler="0 0 {yaw:.4g}" size="0.1" group="5" rgba="0 0 0 0"/>')


# =========================================================================== robots
def robot(i, x, y, accent_mat):
    """Holonomic (mecanum) stretcher UGV.

    Two bodies: the physics body (slide joints + an invisible round collider, so robots
    slide past each other) and a mocap body carrying the visual model, which the env
    turns to face the direction of travel."""
    out = [f'<body name="robot_{i}" pos="{fmt(x, y, 0.3)}" childclass="robot">',
           f'  <joint name="r{i}_x" axis="1 0 0"/>',
           f'  <joint name="r{i}_y" axis="0 1 0"/>',
           f'  <geom name="robot_{i}_col"/>']
    if i == 0:
        out.append('  <camera name="follow" mode="trackcom" pos="0 -6 11" xyaxes="1 0 0 0 0.878 0.479" fovy="50"/>')
    out.append("</body>")
    b = Body((0, 0, 0))
    b.add("box", pos=(0, 0, 0.0), size=(0.34, 0.58, 0.1), mat="metal_dark", cls="rvis")
    b.add("box", pos=(0, 0, 0.0), size=(0.345, 0.4, 0.06), mat=accent_mat, cls="rvis")
    b.add("box", pos=(0, 0.6, -0.02), size=(0.3, 0.03, 0.07), rgba=(0.95, 0.75, 0.1, 1), cls="rvis")
    b.add("box", pos=(0, -0.6, -0.02), size=(0.3, 0.03, 0.07), rgba=(0.95, 0.75, 0.1, 1), cls="rvis")
    b.add("box", pos=(0.2, 0.62, 0.02), size=(0.06, 0.012, 0.03), rgba=(1.0, 0.95, 0.8, 1), cls="rvis")   # headlights
    b.add("box", pos=(-0.2, 0.62, 0.02), size=(0.06, 0.012, 0.03), rgba=(1.0, 0.95, 0.8, 1), cls="rvis")
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.add("cylinder", pos=(sx * 0.42, sy * 0.4, -0.12), size=(0.18, 0.06), euler=(0, 90, 0), rgba=TIRE, cls="rvis")
            b.add("cylinder", pos=(sx * 0.48, sy * 0.4, -0.12), size=(0.1, 0.01), euler=(0, 90, 0), rgba=STEEL, cls="rvis")
    b.add("box", pos=(0, 0.05, 0.13), size=(0.28, 0.5, 0.025), rgba=(0.92, 0.48, 0.12, 1), cls="rvis")
    for sx in (-0.3, 0.3):
        b.add("box", pos=(sx, 0.05, 0.18), size=(0.015, 0.5, 0.02), rgba=STEEL, cls="rvis")
    b.add("cylinder", pos=(0, -0.5, 0.35), size=(0.025, 0.25), rgba=STEEL, cls="rvis")
    b.add("cylinder", pos=(0, -0.5, 0.63), size=(0.08, 0.04), mat="metal_dark", cls="rvis")
    b.add("box", pos=(0, -0.43, 0.5), size=(0.06, 0.03, 0.04), rgba=DARK, cls="rvis")
    b.add("sphere", pos=(0, -0.5, 0.71), size=(0.045,), mat=accent_mat, cls="rvis")
    out.append(f'<body name="robot_{i}_vis" mocap="true" pos="{fmt(x, y, 0.3)}">')
    out += ["  " + l for l in b.lines]
    out.append("</body>")
    WORLD.extend(out)


# =========================================================================== layout
def layout():
    ground_and_roads()

    # ---------- boundary: the site is closed by buildings, barricades and a wrecked bus
    for name, (x0, x1, y0, y1) in {"bound_n": (-17, 17, 16.0, 16.5), "bound_s": (-17, 17, -16.5, -16.0),
                                   "bound_e": (16.0, 16.5, -17, 17), "bound_w": (-16.5, -16.0, -17, 17)}.items():
        collider(x0, x1, y0, y1, h=2.0, name=name)

    # ---------- NW block: residential, alley, roofless ruin, back yard
    building(-15.5, -10.2, 6.0, 11.0, 2, "plaster_beige", 1, name="house_nw")
    ruin_shell(-8.6, -5.5, 6.5, 12.0, door_x0=-7.7, door_x1=-6.3, wall="plaster_rose")
    collider(-5.5, -5.2, 12.0, 16.0, h=1.2, name="garden_wall")
    WORLD.append(geom("box", pos=(-5.35, 14.0, 0.6), size=(0.15, 2.0, 0.6), mat="brick"))
    fallen_tree(-14.0, -11.0, 13.9)
    tree(-15.0, 15.0, s=0.9)
    tree(-6.8, 15.2, burnt=True, s=0.8)
    collider(-15.9, -15.5, 5.0, 12.0, h=1.0)
    WORLD.append(geom("box", pos=(-15.7, 8.5, 0.5), size=(0.2, 3.5, 0.5), mat="brick"))
    debris_field(-9.4, 8.5, 0.3, 2.0, 8)
    dust(-6.3, 11.0, 0.8, 0.8)      # over the collapsed corner, not the interior

    # ---------- NE block: commercial, collapsed building spilling into the cross street
    building(11.0, 15.5, 5.8, 10.0, 2, "plaster_grey", 2, name="shops_ne")
    building(5.5, 9.5, 9.5, 15.5, 3, "plaster_grey", 3, name="collapsed_ne")
    collider(-1.0, 5.5, 10.5, 13.5, h=1.5, name="spill_ne")
    b = Body((2.25, 12.0, 0))
    rubble_pile(b, 0, 0, 3.0, 1.4, 1.3, 40)
    b.emit()
    building(10.5, 15.5, 11.8, 15.5, 2, "metal_clad", 1, name="warehouse_ne")
    fire(7.5, 12.0, 1.6, 1.2)
    smoke(7.5, 12.0, 2.5, height=10)
    dust(4.5, 11.5, 2.5, 1.8)
    debris_over(6.2, 8.75, 0)
    debris_field(8.0, 7.0, 2.4, 1.5, 14)

    # ---------- SE block: parking lot, overturned car, fallen power line, damaged apartments
    building(12.0, 15.5, -15.5, -6.5, 2, "plaster_beige", 2, name="apartments_se")
    building(5.5, 11.0, -15.5, -13.0, 1, "brick", 1, roof="roof", name="shops_se")
    for k in range(5):           # parking bay lines
        WORLD.append(geom("box", pos=(5.6 + k * 2.4, -12.4, 0.006), size=(0.05, 1.0, 0.001), rgba=(0.85, 0.85, 0.8, 1)))
    car(9.0, -9.2, 30, (0.75, 0.75, 0.72, 1), "overturned")
    car(11.0, -12.6, 0, (0.12, 0.11, 0.1, 1), "burnt")
    car(-14.0, -17.6, 0, (0.25, 0.35, 0.55, 1), "parked")       # outside the site
    fire(11.6, -12.4, 0.9, 0.8)
    smoke(11.6, -12.4, 1.6, height=9, drift=(-0.2, 0.5))
    collider(6.5, 12.0, -6.7, -6.3, h=0.6, name="fallen_pole_se")
    electric_pole(12.0, -6.5, state="fallen", yaw=180)
    WORLD.append(geom("cylinder", fromto=(4.9, -6.2, 0.15, 7.0, -7.2, 0.02), size=(0.012,), rgba=DARK))
    WORLD.append(geom("cylinder", fromto=(4.9, -6.8, 0.15, 7.6, -7.6, 0.02), size=(0.012,), rgba=DARK))
    debris_field(8.5, -8.0, 2.5, 1.2, 10)

    # ---------- main street east: crashed car, fallen streetlight, bus on its side (closes the road)
    car(5.6, 1.6, 15, (0.6, 0.12, 0.1, 1), "crashed")
    collider(8.75, 9.25, -3.5, 1.65, h=0.6, name="fallen_light_e")
    streetlight(9.0, -4.4, 90, state="fallen")
    bus_on_side(13.8, 0.0, 90)
    smoke(13.0, 1.0, 2.0, height=7, dark=0.3)
    uplifted_slab(6.8, -2.2, 20)
    crack(4.0, -1.5, 3.5, 15)
    crack(10.5, 2.2, 3.0, -30)
    pothole(7.2, 2.6, 0.45)

    # ---------- intersection: sinkhole, cracked asphalt
    sinkhole(-1.5, 1.5, -1.5, 1.5)
    crack(-2.6, 2.0, 2.5, 60)
    crack(2.3, -2.4, 2.8, -20)
    crack(-1.8, -2.8, 2.2, 160)

    # ---------- cross street north: rubble gap and barricade; south: fallen light and rescue truck
    jersey_barriers(-3.5, -1.0, 15.2, 15.9)
    collider(-4.6, 1.65, -9.25, -8.75, h=0.6, name="fallen_light_s")
    streetlight(-4.4, -9.0, 0, state="fallen")
    rescue_truck(0.0, -14.8, 0)
    jersey_barriers(-3.5, -2.9, -16.0, -13.6)
    jersey_barriers(2.9, 3.5, -16.0, -13.6)
    for cx in (-2.5, -1.5, 1.5, 2.5):
        cone(cx, -12.8)
    crack(0.8, -6.5, 3.0, 80)
    pothole(-1.5, -11.5, 0.5)
    crack(-1.0, 7.5, 2.6, 100)

    # ---------- main street west: emergency staging (fire truck, ambulance), closed by barriers
    fire_truck(-11.8, 2.3, 0)
    ambulance(-12.8, -2.25, 0)
    jersey_barriers(-16.0, -15.4, -3.5, 3.5)
    responder(-15.0, 4.3, 200, "firefighter")
    responder(-14.2, 4.6, 170, "firefighter")
    responder(-9.6, -3.9, 160, "paramedic")

    # ---------- SW block: emergency medical station and evacuation area
    WORLD.append(geom("box", pos=(-10.0, -9.0, 0.008), size=(2.5, 2.0, 0.002), mat="tarp", name="safe_zone"))
    for (a, b_, c, d) in ((-12.5, -7.5, -11.0, -10.85), (-12.5, -7.5, -7.15, -7.0),
                          (-12.5, -12.35, -11.0, -7.0), (-7.65, -7.5, -11.0, -7.0)):
        WORLD.append(geom("box", pos=((a + b_) / 2, (c + d) / 2, 0.011), size=((b_ - a) / 2, (d - c) / 2, 0.001),
                          rgba=(0.85, 0.1, 0.1, 1)))
    slots = [(-11.5, -9.95), (-10.0, -9.95), (-8.5, -9.95), (-11.5, -8.05), (-10.0, -8.05), (-8.5, -8.05)]
    for k, (x, y) in enumerate(slots):
        cot(x, y, k)
    tent(-15.5, -13.0, -11.5, -6.5, ridge="y", open_side=None)
    tent(-12.0, -8.0, -15.5, -12.5, ridge="x", colour=(0.92, 0.92, 0.88, 1))
    flag(-7.6, -12.2, (0.95, 0.95, 0.95, 1), cross=True)
    ambulance(-6.05, -8.0, 90)
    supplies(-7.2, -5.3, -14.6, -11.6)
    assembly_point(-15.5, -12.8, -15.5, -12.7)
    responder(-12.7, -6.0, 90, "paramedic")
    responder(-13.6, -12.2, 45, "paramedic")
    responder(-6.4, -11.2, 120, "rescue")
    responder(-14.3, -14.0, 0, "rescue")
    responder(-3.9, -13.0, 90, "police")
    responder(2.0, 14.3, 270, "police")

    # ---------- street furniture: poles with wires, streetlights (some damaged)
    poles = [(-14.0, 4.6), (-6.0, 4.6), (6.0, 4.6), (14.0, 4.6)]
    for k, (x, y) in enumerate(poles):
        electric_pole(x, y, state="leaning" if k == 2 else "ok", yaw=0)
    wires([(-14.0, 4.6, 7.72), (-6.0, 4.6, 7.72)])
    wires([(6.0, 4.6, 7.72), (14.0, 4.6, 7.72)], sag=1.6)
    streetlight(-10.0, -4.4, 90, "ok")
    streetlight(-4.4, 9.0, 0, "bent")
    streetlight(4.4, 8.5, 180, "ok")
    streetlight(4.4, -10.0, 180, "bent")
    streetlight(12.0, 4.4, 270, "bent")

    # ---------- victim spawn points by situation (the env samples 6 per episode)
    spawn("inside_ruin_a", -7.3, 10.2, 0)
    spawn("inside_ruin_b", -6.8, 8.0, 80)
    spawn("backyard_tree", -12.6, 12.6, 95)
    spawn("alley", -9.4, 9.0, 0)
    spawn("sidewalk_nw", -12.0, 4.4, 90)
    spawn("road_crash", 3.6, -1.9, 160)
    spawn("lot_under_slab", 6.2, 8.4, 180)
    spawn("collapse_edge_ne", 8.6, 8.0, 30)
    spawn("street_gap_n", -2.3, 12.2, 0)
    spawn("spill_edge", 0.6, 9.6, 90)
    spawn("lot_overturned", 7.4, -12.1, 100)
    spawn("power_line_se", 6.2, -7.6, 85)
    spawn("behind_light_e", 11.0, -1.8, 200)
    spawn("shopfront_ne", 11.6, 4.8, 90)
    spawn("back_passage_ne", 13.2, 10.9, 90)
    spawn("road_s", 2.3, -11.0, 0)
    spawn("cross_street_n", 1.6, 6.6, 20)
    spawn("barricade_n", -0.6, 14.6, 90)
    spawn("main_street_w", -6.5, -2.4, 0)
    spawn("sidewalk_se", 9.5, -4.4, 90)
    spawn("lot_parked", 6.0, -10.2, 270)
    spawn("road_far_s", -2.2, -7.0, 180)
    debris_over(-6.8, 8.0, 80)
    debris_over(3.6, -1.9, 160)

    # ---------- victims (positions are overwritten by the env on reset)
    shirts = [(0.2, 0.35, 0.6, 1), (0.7, 0.2, 0.2, 1), (0.3, 0.5, 0.3, 1), (0.85, 0.75, 0.3, 1),
              (0.45, 0.45, 0.5, 1), (0.55, 0.3, 0.55, 1)]
    pants = [(0.2, 0.2, 0.25, 1), (0.3, 0.28, 0.22, 1), (0.25, 0.3, 0.45, 1), (0.15, 0.15, 0.18, 1),
             (0.35, 0.25, 0.2, 1), (0.2, 0.25, 0.35, 1)]
    for j in range(N_VICTIM_MODELS):
        victim(j, (-20 - 2 * j, -20), shirts[j % len(shirts)], pants[(j * 2 + 1) % len(pants)], SKIN[j % len(SKIN)])

    # ---------- rescue robots (robot_0 is the learner)
    robot(0, -14.2, -4.3, "learner")     # beside the station, clear of the triage entrance
    robot(1, -10.2, -5.7, "agent")
    robot(2, -8.5, -5.7, "agent")

    # ---------- surrounding city (scenery only, no colliders)
    for (x0, x1, f, mat, dmg) in ((-28, -18, 4, "plaster_grey", 1), (-16.5, -8.5, 3, "plaster_beige", 2),
                                  (-7.5, -4.5, 3, "brick", 1), (4.5, 12, 4, "plaster_teal", 1), (13, 22, 3, "plaster_rose", 0),
                                  (23, 30, 4, "plaster_grey", 1)):
        building(x0, x1, 17.2, 22.0, f, mat, dmg, solid=False)
    building(-24.0, -17.2, 6, 14, 3, "plaster_grey", 1, solid=False)
    for (y0, y1, f, mat) in ((-15, -6, 2, "plaster_teal"), (6, 14, 3, "plaster_grey")):
        building(17.2, 24.0, y0, y1, f, mat, 1, solid=False)
    for x, y in ((-19.5, -8), (-22, -11.5), (-19, -13.5), (-23, -7)):     # small park west of the station
        tree(x, y, s=rng.uniform(0.9, 1.1))
    building(17.2, 24.0, -26, -17, 2, "brick", 0, solid=False)
    for x, y in ((-12, -19), (-8, -20.5), (6, -19), (11, -20), (15, -18.5), (-20, -19), (-18, -22), (20, -21)):
        tree(x, y, burnt=rng.random() < 0.25, s=rng.uniform(0.9, 1.15))
    for x0, x1 in ((-16.0, -4.0), (4.0, 16.0)):     # low walls marking the site edge
        WORLD.append(geom("box", pos=((x0 + x1) / 2, -16.25, 0.35), size=((x1 - x0) / 2, 0.15, 0.35), mat="concrete"))
    for x0, x1 in ((-16.0, -4.0), (4.0, 16.0)):
        WORLD.append(geom("box", pos=((x0 + x1) / 2, 16.25, 0.4), size=((x1 - x0) / 2, 0.15, 0.4), mat="concrete"))


# =========================================================================== document
HEADER = """<mujoco model="disaster_response_urban">
  <!-- GENERATED by envs/scene_builder.py - edit the builder, not this file. -->
  <compiler angle="degree"/>
  <option timestep="0.01"/>
  <statistic center="0 0 0" extent="20"/>
  <size memory="20M"/>

  <visual>
    <global offwidth="1920" offheight="1200" fovy="40"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.05" zfar="200" shadowclip="1.6" shadowscale="0.5" fogstart="2.4" fogend="6" haze="0.3"/>
    <rgba fog="0.72 0.66 0.58 1" haze="0.78 0.7 0.6 1"/>
    <headlight ambient="0.32 0.31 0.3" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.8 0.72 0.62" rgb2="0.36 0.38 0.44" width="512" height="3072"/>
    <texture name="soil" type="2d" builtin="flat" mark="random" random="0.15" rgb1="0.46 0.42 0.36" markrgb="0.34 0.31 0.27" width="512" height="512"/>
    <texture name="asphalt" type="2d" builtin="flat" mark="random" random="0.25" rgb1="0.25 0.25 0.26" markrgb="0.36 0.35 0.34" width="512" height="512"/>
    <texture name="paving" type="2d" builtin="checker" rgb1="0.6 0.59 0.56" rgb2="0.56 0.55 0.52" width="128" height="128"/>
    <texture name="concrete" type="2d" builtin="flat" mark="random" random="0.1" rgb1="0.6 0.59 0.56" markrgb="0.45 0.44 0.42" width="256" height="256"/>
    <texture name="brick" type="2d" builtin="flat" mark="edge" rgb1="0.56 0.31 0.24" markrgb="0.42 0.38 0.34" width="64" height="64"/>
    <texture name="plaster_beige" type="2d" builtin="flat" mark="random" random="0.06" rgb1="0.78 0.72 0.6" markrgb="0.68 0.62 0.5" width="256" height="256"/>
    <texture name="plaster_grey" type="2d" builtin="flat" mark="random" random="0.06" rgb1="0.66 0.66 0.64" markrgb="0.55 0.55 0.53" width="256" height="256"/>
    <texture name="plaster_teal" type="2d" builtin="flat" mark="random" random="0.06" rgb1="0.5 0.62 0.6" markrgb="0.42 0.52 0.5" width="256" height="256"/>
    <texture name="plaster_rose" type="2d" builtin="flat" mark="random" random="0.06" rgb1="0.74 0.58 0.52" markrgb="0.64 0.48 0.42" width="256" height="256"/>
    <texture name="metal_clad" type="2d" builtin="gradient" rgb1="0.62 0.64 0.66" rgb2="0.5 0.52 0.54" width="64" height="64"/>
    <texture name="roof" type="2d" builtin="flat" mark="random" random="0.12" rgb1="0.32 0.31 0.3" markrgb="0.42 0.4 0.38" width="256" height="256"/>
    <texture name="tarp" type="2d" builtin="checker" rgb1="0.88 0.88 0.85" rgb2="0.83 0.83 0.8" width="64" height="64"/>

    <material name="soil" texture="soil" texrepeat="40 40" texuniform="true"/>
    <material name="asphalt" texture="asphalt" texrepeat="1 1" texuniform="true"/>
    <material name="paving" texture="paving" texrepeat="1.5 1.5" texuniform="true"/>
    <material name="concrete" texture="concrete" texrepeat="1 1" texuniform="true"/>
    <material name="brick" texture="brick" texrepeat="4 4" texuniform="true"/>
    <material name="plaster_beige" texture="plaster_beige" texrepeat="0.5 0.5" texuniform="true"/>
    <material name="plaster_grey" texture="plaster_grey" texrepeat="0.5 0.5" texuniform="true"/>
    <material name="plaster_teal" texture="plaster_teal" texrepeat="0.5 0.5" texuniform="true"/>
    <material name="plaster_rose" texture="plaster_rose" texrepeat="0.5 0.5" texuniform="true"/>
    <material name="metal_clad" texture="metal_clad" texrepeat="3 1" texuniform="true" specular="0.4"/>
    <material name="roof" texture="roof" texrepeat="1 1" texuniform="true"/>
    <material name="tarp" texture="tarp" texrepeat="1 1" texuniform="true"/>
    <material name="paint" specular="0.6" shininess="0.7"/>
    <material name="smoke" specular="0" shininess="0" emission="0.35"/>
    <material name="flame" specular="0" shininess="0" emission="1"/>
    <material name="metal_dark" rgba="0.2 0.21 0.23 1" specular="0.5" shininess="0.6"/>
    <material name="learner" rgba="0.62 0.3 0.92 1" specular="0.4" shininess="0.5"/>
    <material name="agent" rgba="0.15 0.48 0.95 1" specular="0.4" shininess="0.5"/>
  </asset>

  <default>
    <default class="collider">
      <geom type="box" group="3" rgba="0.9 0.2 0.9 0.35"/>
    </default>
    <default class="deco">
      <geom contype="0" conaffinity="0" density="0" group="1"/>
    </default>
    <default class="person">
      <geom contype="0" conaffinity="0" group="1"/>
    </default>
    <default class="robot">
      <joint type="slide" damping="5"/>
      <geom type="cylinder" size="0.55 0.18" mass="5" group="3" rgba="0.9 0.2 0.9 0.35" friction="0.3"/>
    </default>
    <default class="rvis">
      <geom contype="0" conaffinity="0" density="0" group="1"/>
    </default>
  </default>
"""

CAMERAS = """
    <camera name="overview" pos="0 -34.5 30" xyaxes="1 0 0 0 0.69 0.72" fovy="45"/>
    <camera name="isometric" pos="-25 -27 25" xyaxes="0.707 -0.707 0 0.42 0.42 0.805" fovy="45"/>
    <camera name="triage" pos="-4 -16 8" xyaxes="0.83 0.55 0 -0.3 0.46 0.83" fovy="50"/>
    <camera name="ruins" pos="-3 -1 9" xyaxes="0.8 0.6 0 -0.42 0.56 0.71" fovy="50"/>
    <camera name="collapse" pos="12 -2 9" xyaxes="0.83 0.55 0 -0.36 0.54 0.76" fovy="55"/>
"""

FOOTER = """
  <actuator>
    <velocity name="r0_vx" joint="r0_x" kv="50" ctrlrange="-2.5 2.5"/>
    <velocity name="r0_vy" joint="r0_y" kv="50" ctrlrange="-2.5 2.5"/>
    <velocity name="r1_vx" joint="r1_x" kv="50" ctrlrange="-2.5 2.5"/>
    <velocity name="r1_vy" joint="r1_y" kv="50" ctrlrange="-2.5 2.5"/>
    <velocity name="r2_vx" joint="r2_x" kv="50" ctrlrange="-2.5 2.5"/>
    <velocity name="r2_vy" joint="r2_y" kv="50" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
"""


def build():
    WORLD.clear()
    _lights.clear()
    layout()
    sun = ('<light name="sun" directional="true" pos="0 0 30" dir="0.45 0.55 -1" diffuse="0.78 0.7 0.58" '
           'specular="0.15 0.15 0.15" castshadow="true"/>')
    body = ["  <worldbody>", "    " + sun] + ["    " + l for l in _lights] + [CAMERAS.rstrip()] \
        + ["    " + l for l in WORLD] + ["  </worldbody>"]
    xml = HEADER + "\n".join(body) + "\n" + FOOTER
    with open(OUT_PATH, "w") as f:
        f.write(xml)
    return OUT_PATH


if __name__ == "__main__":
    print("wrote", build())