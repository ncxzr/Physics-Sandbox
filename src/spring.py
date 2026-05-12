"""
spring.py — Damped spring connecting two Ball objects.

Physics model
─────────────
Hooke's law (restoring force):
    F_spring = -k · (|r| - L₀) · r̂

where:
    k   = spring constant  (N/m)
    |r| = current length   (m)
    L₀  = natural length   (m)
    r̂   = unit vector from ball_a to ball_b

Damping force (velocity-proportional viscous damping):
    F_damp = -c · (v_b - v_a) · r̂  · r̂    (projected onto spring axis)

where:
    c = damping coefficient (N·s/m)

Critical damping: c_crit = 2·sqrt(k·m_reduced)
    m_reduced = (m_a · m_b) / (m_a + m_b)

Equal and opposite forces are applied to each ball (Newton's 3rd law).
"""

from __future__ import annotations
from src.vector import Vec2
from src.ball import Ball
import math


class Spring:
    """
    A Hooke's-law spring with optional viscous damping.

    Parameters
    ----------
    ball_a, ball_b : endpoints
    k              : spring constant (N/m)  — stiffness
    natural_length : rest length (pixels); if None, uses current separation
    damping        : damping coefficient c (N·s/m)
    """

    def __init__(
        self,
        ball_a: Ball,
        ball_b: Ball,
        k: float = 200.0,
        natural_length: float = None,
        damping: float = 5.0,
        color: tuple = (180, 180, 100),
    ):
        self.ball_a = ball_a
        self.ball_b = ball_b
        self.k = k
        self.damping = damping
        self.color = color

        if natural_length is None:
            self.natural_length = ball_a.pos.distance_to(ball_b.pos)
        else:
            self.natural_length = natural_length

    # ── physics ──────────────────────────────────────────────────────────
    def apply_forces(self) -> None:
        """
        Compute spring + damping forces and accumulate onto each ball.

        Steps:
        1. Compute displacement vector r = pos_b - pos_a
        2. Extension x = |r| - L₀
        3. Spring force magnitude: F_s = k · x  (along r̂)
        4. Relative velocity projected onto spring axis: v_rel = (v_b - v_a)·r̂
        5. Damping force magnitude: F_d = c · v_rel  (along r̂)
        6. Total force on a = +(F_s + F_d)·r̂;  on b = -(F_s + F_d)·r̂
        """
        r = self.ball_b.pos - self.ball_a.pos        # displacement vector
        length = r.magnitude()

        if length < 1e-6:                             # avoid division by zero
            return

        r_hat = r / length                            # unit vector a→b

        # ── Hooke's law ──────────────────────────────────────────────────
        extension = length - self.natural_length      # positive = stretched
        f_spring_mag = self.k * extension             # F = k·x

        # ── viscous damping ──────────────────────────────────────────────
        rel_vel = self.ball_b.vel - self.ball_a.vel
        v_rel_proj = rel_vel.dot(r_hat)               # projection onto axis
        f_damp_mag = self.damping * v_rel_proj

        # total force magnitude along r̂ (on ball_a toward ball_b)
        total = f_spring_mag + f_damp_mag

        force_on_a = r_hat * total
        force_on_b = r_hat * (-total)

        if not self.ball_a.pinned:
            self.ball_a.apply_force(force_on_a)
        if not self.ball_b.pinned:
            self.ball_b.apply_force(force_on_b)

    # ── derived quantities ────────────────────────────────────────────────
    @property
    def current_length(self) -> float:
        return self.ball_a.pos.distance_to(self.ball_b.pos)

    @property
    def extension(self) -> float:
        """Positive = stretched, negative = compressed."""
        return self.current_length - self.natural_length

    @property
    def potential_energy(self) -> float:
        """PE = ½·k·x²"""
        x = self.extension
        return 0.5 * self.k * x * x

    @property
    def critical_damping(self) -> float:
        """c_crit = 2·sqrt(k·m_reduced)"""
        m_a = self.ball_a.mass
        m_b = self.ball_b.mass
        m_r = (m_a * m_b) / (m_a + m_b)
        return 2 * math.sqrt(self.k * m_r)

    @property
    def damping_ratio(self) -> float:
        """ζ = c / c_crit  (0=undamped, 1=critically damped, >1=overdamped)"""
        c_crit = self.critical_damping
        if c_crit < 1e-9:
            return 0.0
        return self.damping / c_crit

    def __repr__(self) -> str:
        return (
            f"Spring(k={self.k:.1f}, L₀={self.natural_length:.1f}, "
            f"c={self.damping:.1f}, ζ={self.damping_ratio:.2f})"
        )
