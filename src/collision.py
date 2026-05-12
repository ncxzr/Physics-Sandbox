"""
collision.py — Collision detection and impulse-based resolution.

Ball–ball elastic/inelastic collision
──────────────────────────────────────
Using conservation of momentum and kinetic energy (generalised for
coefficient of restitution e):

    1D resolved along the collision normal n̂:

        v1' = u1 - (1+e)·m2/(m1+m2) · (u1 - u2) · [n̂] · n̂
        v2' = u2 + (1+e)·m1/(m1+m2) · (u1 - u2) · [n̂] · n̂

    where [n̂] denotes the component projected onto n̂.

For e = 1 (perfectly elastic):
    reduces to standard elastic collision formulae.

Ball–wall reflection
─────────────────────
    Velocity component normal to wall is reversed and scaled by e_wall.
    Tangential component is unchanged (frictionless walls).

Positional correction (depenetration)
──────────────────────────────────────
    After velocity update, objects are nudged apart to prevent sinking,
    proportional to the overlap depth.
"""

from __future__ import annotations
from src.vector import Vec2
from src.ball import Ball

# ── constants ────────────────────────────────────────────────────────────
WALL_RESTITUTION = 0.80   # energy retained on wall bounce


def resolve_ball_wall(ball: Ball, width: int, height: int) -> None:
    """
    Reflect ball off bounding-box walls.
    Applies restitution and corrects position to prevent tunnelling.
    """
    e = ball.restitution * WALL_RESTITUTION   # combined restitution

    # ── left / right walls ───────────────────────────────────────────────
    if ball.pos.x - ball.radius < 0:
        ball.pos = Vec2(ball.radius, ball.pos.y)
        ball.vel = Vec2(-ball.vel.x * e, ball.vel.y)

    elif ball.pos.x + ball.radius > width:
        ball.pos = Vec2(width - ball.radius, ball.pos.y)
        ball.vel = Vec2(-ball.vel.x * e, ball.vel.y)

    # ── top / bottom walls ───────────────────────────────────────────────
    if ball.pos.y - ball.radius < 0:
        ball.pos = Vec2(ball.pos.x, ball.radius)
        ball.vel = Vec2(ball.vel.x, -ball.vel.y * e)

    elif ball.pos.y + ball.radius > height:
        ball.pos = Vec2(ball.pos.x, height - ball.radius)
        ball.vel = Vec2(ball.vel.x, -ball.vel.y * e)


def resolve_ball_ball(a: Ball, b: Ball) -> bool:
    """
    Detect and resolve collision between two balls.

    Returns True if a collision occurred.

    Algorithm
    ---------
    1. Check overlap: distance < r_a + r_b
    2. Compute collision normal n̂ = (pos_b - pos_a).normalised()
    3. Compute relative velocity along normal
    4. If separating (v_rel > 0), skip (already moving apart)
    5. Apply impulse scalar J using combined restitution
    6. Update velocities:  v += ±J/m · n̂
    7. Positional correction: push objects apart by overlap amount
    """
    diff = b.pos - a.pos
    dist = diff.magnitude()
    min_dist = a.radius + b.radius

    # ── broad-phase check ────────────────────────────────────────────────
    if dist >= min_dist or dist < 1e-9:
        return False

    # ── collision normal ─────────────────────────────────────────────────
    n_hat = diff / dist   # unit vector from a → b

    # ── relative velocity along normal ───────────────────────────────────
    rel_vel = b.vel - a.vel
    v_rel_n = rel_vel.dot(n_hat)   # scalar; positive = separating

    if v_rel_n > 0:
        return False   # already separating — no impulse needed

    # ── impulse scalar (coefficient of restitution formulation) ──────────
    # J = -(1 + e) · v_rel_n / (1/m_a + 1/m_b)
    e = min(a.restitution, b.restitution)
    inv_mass_sum = (1.0 / a.mass) + (1.0 / b.mass)
    J = -(1.0 + e) * v_rel_n / inv_mass_sum

    # ── velocity update ──────────────────────────────────────────────────
    if not a.pinned:
        a.vel = a.vel - n_hat * (J / a.mass)
    if not b.pinned:
        b.vel = b.vel + n_hat * (J / b.mass)

    # ── positional correction (prevents sinking / overlap accumulation) ──
    overlap = min_dist - dist
    correction_fraction = 0.5   # split correction between both objects
    correction = n_hat * (overlap * correction_fraction)

    if not a.pinned:
        a.pos = a.pos - correction
    if not b.pinned:
        b.pos = b.pos + correction

    return True


def check_all_collisions(
    balls: list[Ball], width: int, height: int
) -> int:
    """
    Check every ball pair for collisions and resolve them.
    Also resolves wall collisions for each ball.

    Returns total number of ball–ball collisions this frame.
    """
    collision_count = 0

    # ── ball–ball (O(n²) brute force — fine for small n) ─────────────────
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            if resolve_ball_ball(balls[i], balls[j]):
                collision_count += 1

    # ── wall collisions ───────────────────────────────────────────────────
    for ball in balls:
        resolve_ball_wall(ball, width, height)

    return collision_count
