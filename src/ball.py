"""
ball.py — Rigid-body ball with mass, position, velocity, and force accumulation.

Physics model
─────────────
Newton's 2nd law:  F = m·a  →  a = F/m
Euler integration: v(t+dt) = v(t) + a·dt
                   x(t+dt) = x(t) + v(t+dt)·dt   (semi-implicit Euler)

Kinetic energy:    KE = ½·m·|v|²
Momentum:          p  = m·v
"""

from __future__ import annotations
import math
from src.vector import Vec2


class Ball:
    """
    A 2D circular rigid body.

    Parameters
    ----------
    pos     : initial position (pixels)
    vel     : initial velocity (pixels/s)
    mass    : kg  (also scales radius for visual clarity)
    radius  : pixels (if None, derived from mass)
    color   : RGB tuple for rendering
    restitution : coefficient of restitution  0=perfectly inelastic, 1=elastic
    """

    _id_counter = 0

    def __init__(
        self,
        pos: Vec2,
        vel: Vec2 = None,
        mass: float = 1.0,
        radius: float = None,
        color: tuple = (100, 200, 255),
        restitution: float = 0.85,
    ):
        Ball._id_counter += 1
        self.id = Ball._id_counter

        self.pos = pos.copy()
        self.vel = vel.copy() if vel else Vec2(0, 0)
        self.mass = max(mass, 0.1)          # guard against zero mass
        self.radius = radius if radius else self._mass_to_radius(self.mass)
        self.color = color
        self.restitution = restitution

        # force accumulator — reset each frame after integration
        self._force = Vec2(0, 0)

        # trajectory trail (list of Vec2 positions)
        self.trail: list[Vec2] = []
        self.max_trail = 80

        # pinned balls don't move (used for spring anchors)
        self.pinned = False

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _mass_to_radius(mass: float) -> float:
        """Visual radius scales with cube-root of mass (density = const)."""
        return max(8.0, 8.0 * (mass ** (1 / 3)))

    # ── force accumulation ───────────────────────────────────────────────
    def apply_force(self, force: Vec2) -> None:
        """Add a force to be applied this timestep. Call before integrate()."""
        self._force = self._force + force

    def clear_forces(self) -> None:
        self._force = Vec2(0, 0)

    # ── numerical integration ────────────────────────────────────────────
    def integrate_euler(self, dt: float) -> None:
        """
        Semi-implicit (symplectic) Euler integration.

        Update order: velocity first, then position.
        This conserves energy better than explicit Euler for oscillatory systems.
        """
        if self.pinned:
            return
        acceleration = self._force / self.mass   # a = F/m
        self.vel = self.vel + acceleration * dt   # v ← v + a·dt
        self.pos = self.pos + self.vel * dt       # x ← x + v·dt

        self._record_trail()
        self.clear_forces()

    def integrate_rk4(self, dt: float, force_fn) -> None:
        """
        Runge-Kutta 4th-order integration for higher accuracy.

        force_fn(pos, vel) → Vec2  must return the net force at a given state.

        RK4 stages (for dx/dt = v,  dv/dt = F(x,v)/m):
            k1 = f(t,        y)
            k2 = f(t + dt/2, y + dt/2 · k1)
            k3 = f(t + dt/2, y + dt/2 · k2)
            k4 = f(t + dt,   y + dt   · k3)
            y_new = y + dt/6 · (k1 + 2k2 + 2k3 + k4)
        """
        if self.pinned:
            return

        def derivatives(pos: Vec2, vel: Vec2):
            f = force_fn(pos, vel)
            return vel, f / self.mass   # (dx/dt, dv/dt)

        p0, v0 = self.pos, self.vel

        dp1, dv1 = derivatives(p0, v0)
        dp2, dv2 = derivatives(p0 + dp1 * (dt / 2), v0 + dv1 * (dt / 2))
        dp3, dv3 = derivatives(p0 + dp2 * (dt / 2), v0 + dv2 * (dt / 2))
        dp4, dv4 = derivatives(p0 + dp3 * dt,        v0 + dv3 * dt)

        self.pos = p0 + (dp1 + 2 * dp2 + 2 * dp3 + dp4) * (dt / 6)
        self.vel = v0 + (dv1 + 2 * dv2 + 2 * dv3 + dv4) * (dt / 6)

        self._record_trail()
        self.clear_forces()

    # ── physics quantities ───────────────────────────────────────────────
    @property
    def kinetic_energy(self) -> float:
        """KE = ½·m·v²"""
        return 0.5 * self.mass * self.vel.magnitude_sq()

    @property
    def momentum(self) -> Vec2:
        """p = m·v"""
        return self.vel * self.mass

    @property
    def speed(self) -> float:
        return self.vel.magnitude()

    # ── trail ────────────────────────────────────────────────────────────
    def _record_trail(self):
        self.trail.append(self.pos.copy())
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def clear_trail(self):
        self.trail.clear()

    def __repr__(self) -> str:
        return (
            f"Ball(id={self.id}, m={self.mass:.2f}kg, "
            f"pos={self.pos}, vel={self.vel})"
        )
