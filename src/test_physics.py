"""
tests/test_physics.py — Unit tests for core physics modules.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_physics.py
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vector import Vec2
from src.ball import Ball
from src.spring import Spring
from src.world import World
from src.collision import resolve_ball_ball, resolve_ball_wall


# ── Vec2 ──────────────────────────────────────────────────────────────────
def test_vec2_magnitude():
    v = Vec2(3, 4)
    assert abs(v.magnitude() - 5.0) < 1e-9

def test_vec2_dot():
    assert Vec2(2, 3).dot(Vec2(4, 5)) == 23.0

def test_vec2_normalized():
    n = Vec2(0, 7).normalized()
    assert abs(n.magnitude() - 1.0) < 1e-9
    assert abs(n.y - 1.0) < 1e-9

def test_vec2_zero_normalized():
    n = Vec2(0, 0).normalized()
    assert n.x == 0 and n.y == 0

def test_vec2_arithmetic():
    a, b = Vec2(1, 2), Vec2(3, 4)
    assert (a + b).x == 4 and (a + b).y == 6
    assert (a - b).x == -2
    assert (a * 3).x == 3 and (a * 3).y == 6
    assert (a / 2).x == 0.5


# ── Ball ──────────────────────────────────────────────────────────────────
def test_ball_euler_integration():
    """F=m*a: 10N on 2kg → a=5 m/s². Starting at rest, after 1s v=5, x=5."""
    b = Ball(Vec2(0, 0), Vec2(0, 0), mass=2.0)
    # One big step: apply 10N force, integrate with dt=1
    b.apply_force(Vec2(10, 0))
    b.integrate_euler(1.0)
    assert abs(b.vel.x - 5.0) < 1e-9, f"Expected v=5, got {b.vel.x}"
    assert abs(b.pos.x - 5.0) < 1e-9, f"Expected x=5, got {b.pos.x}"

def test_ball_kinetic_energy():
    """KE = ½ m v²"""
    b = Ball(Vec2(0, 0), Vec2(6, 8), mass=2.0)   # |v| = 10
    assert abs(b.kinetic_energy - 100.0) < 1e-9

def test_ball_momentum():
    b = Ball(Vec2(0, 0), Vec2(3, 4), mass=5.0)
    p = b.momentum
    assert abs(p.x - 15.0) < 1e-9 and abs(p.y - 20.0) < 1e-9

def test_ball_pinned_does_not_move():
    b = Ball(Vec2(100, 100), Vec2(50, 50), mass=1.0)
    b.pinned = True
    b.apply_force(Vec2(1000, 1000))
    b.integrate_euler(0.1)
    assert b.pos.x == 100 and b.pos.y == 100


# ── Spring ────────────────────────────────────────────────────────────────
def test_spring_extension():
    a = Ball(Vec2(0, 0), mass=1.0)
    b = Ball(Vec2(200, 0), mass=1.0)
    sp = Spring(a, b, k=100, natural_length=100)
    assert abs(sp.extension - 100.0) < 1e-9

def test_spring_potential_energy():
    """PE = ½ k x²  →  ½ · 100 · 100² = 500000"""
    a = Ball(Vec2(0, 0), mass=1.0)
    b = Ball(Vec2(200, 0), mass=1.0)
    sp = Spring(a, b, k=100, natural_length=100)
    assert abs(sp.potential_energy - 500000.0) < 1e-9

def test_spring_force_direction():
    """Spring force on A should point toward B when stretched."""
    a = Ball(Vec2(0, 0), mass=1.0)
    b = Ball(Vec2(200, 0), mass=1.0)
    sp = Spring(a, b, k=100, natural_length=100, damping=0)
    sp.apply_forces()
    # Force on A should be positive x (toward B)
    assert a._force.x > 0, f"Expected rightward force on A, got {a._force.x}"
    # Force on B should be negative x (toward A), Newton's 3rd law
    assert b._force.x < 0, f"Expected leftward force on B, got {b._force.x}"

def test_spring_forces_equal_opposite():
    """Newton's 3rd law: |F_a| == |F_b| at rest (no damping)."""
    a = Ball(Vec2(0, 0), mass=1.0)
    b = Ball(Vec2(200, 0), mass=1.0)
    sp = Spring(a, b, k=100, natural_length=100, damping=0)
    sp.apply_forces()
    assert abs(a._force.x + b._force.x) < 1e-9


# ── Collisions ───────────────────────────────────────────────────────────
def test_elastic_collision_momentum():
    """Total momentum conserved in elastic collision."""
    a = Ball(Vec2(0, 0), Vec2(100, 0), mass=2.0, radius=20, restitution=1.0)
    b = Ball(Vec2(30, 0), Vec2(-50, 0), mass=3.0, radius=20, restitution=1.0)
    p_before = a.momentum.x + b.momentum.x
    resolve_ball_ball(a, b)
    p_after = a.momentum.x + b.momentum.x
    assert abs(p_after - p_before) < 1e-6, (
        f"Momentum not conserved: {p_before:.4f} → {p_after:.4f}"
    )

def test_collision_no_overlap_after_resolve():
    """Balls should not overlap after resolution."""
    a = Ball(Vec2(0, 0), Vec2(50, 0), mass=1.0, radius=20, restitution=0.9)
    b = Ball(Vec2(30, 0), Vec2(-50, 0), mass=1.0, radius=20, restitution=0.9)
    resolve_ball_ball(a, b)
    dist = a.pos.distance_to(b.pos)
    assert dist >= a.radius + b.radius - 0.1, (
        f"Balls still overlapping: dist={dist:.2f}, min={a.radius+b.radius:.2f}"
    )

def test_wall_bounce():
    """Ball moving left should bounce right when it hits left wall."""
    b = Ball(Vec2(5, 300), Vec2(-200, 0), mass=1.0, radius=10, restitution=1.0)
    resolve_ball_wall(b, 800, 600)
    assert b.vel.x > 0, f"Ball should bounce right, got vx={b.vel.x}"
    assert b.pos.x >= b.radius, "Ball inside wall after resolve"


# ── World ────────────────────────────────────────────────────────────────
def test_world_gravity():
    """Ball should accelerate downward under gravity."""
    w = World(800, 600, gravity=1000)
    b = Ball(Vec2(400, 100), Vec2(0, 0), mass=1.0)
    w.add_ball(b)
    w.step(0.1)
    assert b.vel.y > 0, "Ball should have downward velocity after gravity step"
    assert b.pos.y > 100, "Ball should have moved downward"

def test_world_energy_spring_conservation():
    """
    Undamped spring-mass with no gravity: energy should be conserved
    within 0.5% over 500 steps (dt=0.001s).
    """
    w = World(4000, 4000, gravity=0, integrator="euler")
    anchor = Ball(Vec2(2000, 2000), mass=1.0, radius=5)
    anchor.pinned = True
    w.add_ball(anchor)
    bob = Ball(Vec2(2050, 2000), Vec2(0, 80), mass=1.0, radius=10)
    w.add_ball(bob)
    w.add_spring(Spring(anchor, bob, k=100, natural_length=50, damping=0))

    for _ in range(500):
        w.step(0.001)

    e0 = w.te_history[0]
    ef = w.te_history[-1]
    drift_pct = abs(ef - e0) / max(abs(e0), 1) * 100
    assert drift_pct < 0.5, f"Energy drift too large: {drift_pct:.3f}%"

def test_world_paused_no_motion():
    """Paused world should not move balls."""
    w = World(800, 600, gravity=500)
    b = Ball(Vec2(400, 300), Vec2(0, 0), mass=1.0)
    w.add_ball(b)
    w.paused = True
    w.step(0.1)
    assert b.pos.y == 300, "Ball moved while world was paused"


# ── runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # Vec2
        test_vec2_magnitude, test_vec2_dot, test_vec2_normalized,
        test_vec2_zero_normalized, test_vec2_arithmetic,
        # Ball
        test_ball_euler_integration, test_ball_kinetic_energy,
        test_ball_momentum, test_ball_pinned_does_not_move,
        # Spring
        test_spring_extension, test_spring_potential_energy,
        test_spring_force_direction, test_spring_forces_equal_opposite,
        # Collisions
        test_elastic_collision_momentum,
        test_collision_no_overlap_after_resolve,
        test_wall_bounce,
        # World
        test_world_gravity, test_world_energy_spring_conservation,
        test_world_paused_no_motion,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {t.__name__}  →  {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}  →  EXCEPTION: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
