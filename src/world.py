"""
world.py — The simulation world: manages all objects, applies forces, steps time.

Responsibilities
────────────────
* Hold the lists of Ball and Spring objects.
* Each simulation step:
    1. Apply gravity to all free balls.
    2. Apply spring forces.
    3. Integrate equations of motion.
    4. Resolve collisions.
* Track global energy and momentum for diagnostics.
* Maintain an energy–time history for optional matplotlib plots.
"""

from __future__ import annotations
import time
from src.vector import Vec2
from src.ball import Ball
from src.spring import Spring
from src.collision import check_all_collisions

# Gravity constant  (pixels/s² — calibrated so 1 px ≈ 1 cm gives g≈980)
DEFAULT_GRAVITY = 500.0   # pixels/s²  (downward = +y in screen coords)


class World:
    """
    Container and stepper for the 2D physics simulation.

    Parameters
    ----------
    width, height : simulation boundary in pixels
    gravity       : downward acceleration (pixels/s²)
    integrator    : 'euler' or 'rk4'
    """

    def __init__(
        self,
        width: int,
        height: int,
        gravity: float = DEFAULT_GRAVITY,
        integrator: str = "euler",
    ):
        self.width = width
        self.height = height
        self.gravity = gravity        # adjustable at runtime
        self.integrator = integrator  # 'euler' or 'rk4'

        self.balls: list[Ball] = []
        self.springs: list[Spring] = []

        self.paused = False
        self.sim_time = 0.0           # total elapsed simulation time (s)

        # ── energy history for plotting ───────────────────────────────────
        self.time_history: list[float] = []
        self.ke_history: list[float] = []
        self.pe_history: list[float] = []
        self.te_history: list[float] = []

        self._max_history = 3000      # ~50 s at 60 fps

    # ── object management ─────────────────────────────────────────────────
    def add_ball(self, ball: Ball) -> Ball:
        self.balls.append(ball)
        return ball

    def remove_ball(self, ball: Ball) -> None:
        self.balls = [b for b in self.balls if b is not ball]
        # also remove any springs connected to this ball
        self.springs = [
            s for s in self.springs
            if s.ball_a is not ball and s.ball_b is not ball
        ]

    def add_spring(self, spring: Spring) -> Spring:
        self.springs.append(spring)
        return spring

    def clear(self) -> None:
        """Remove all objects and reset time."""
        self.balls.clear()
        self.springs.clear()
        self.sim_time = 0.0
        self.time_history.clear()
        self.ke_history.clear()
        self.pe_history.clear()
        self.te_history.clear()
        Ball._id_counter = 0

    # ── simulation step ───────────────────────────────────────────────────
    def step(self, dt: float) -> None:
        """
        Advance the simulation by dt seconds.

        Steps:
        ① Apply gravity forces to all free balls.
        ② Apply spring forces (Hooke's law + damping).
        ③ Integrate equations of motion.
        ④ Resolve collisions (ball–ball and ball–wall).
        ⑤ Record energy history.
        """
        if self.paused or dt <= 0:
            return

        # clamp dt to avoid instability on large frames (e.g. on window focus)
        dt = min(dt, 1 / 30)

        # ① gravity — F_grav = m · g · ĵ  (ĵ points down in screen coords)
        g_vec = Vec2(0, self.gravity)
        for ball in self.balls:
            if not ball.pinned:
                ball.apply_force(g_vec * ball.mass)

        # ② spring forces
        for spring in self.springs:
            spring.apply_forces()

        # ③ integration
        if self.integrator == "rk4":
            self._integrate_rk4(dt)
        else:
            self._integrate_euler(dt)

        # ④ collisions
        check_all_collisions(self.balls, self.width, self.height)

        # ⑤ bookkeeping
        self.sim_time += dt
        self._record_energy()

    # ── integrators ───────────────────────────────────────────────────────
    def _integrate_euler(self, dt: float) -> None:
        """Semi-implicit Euler for all balls."""
        for ball in self.balls:
            ball.integrate_euler(dt)

    def _integrate_rk4(self, dt: float) -> None:
        """
        RK4 for each ball.

        force_fn receives the ball's hypothetical pos/vel and re-evaluates
        gravity + spring forces at that state.  Springs currently use the
        stored state — a full coupled RK4 would require re-evaluating all
        forces simultaneously; this per-ball approach is a practical compromise.
        """
        g_vec = Vec2(0, self.gravity)

        for ball in self.balls:
            if ball.pinned:
                continue

            def force_fn(pos: Vec2, vel: Vec2) -> Vec2:
                # gravity contribution
                f = g_vec * ball.mass
                # spring contributions (using current stored positions of partners)
                for spring in self.springs:
                    if spring.ball_a is ball:
                        r = spring.ball_b.pos - pos
                    elif spring.ball_b is ball:
                        r = pos - spring.ball_a.pos
                        r = -r
                    else:
                        continue
                    length = r.magnitude()
                    if length < 1e-6:
                        continue
                    r_hat = r / length
                    ext = length - spring.natural_length
                    f_s = r_hat * (spring.k * ext)
                    # damping along spring axis
                    if spring.ball_a is ball:
                        rel_v = spring.ball_b.vel - vel
                    else:
                        rel_v = spring.ball_a.vel - vel
                        rel_v = -rel_v
                    v_proj = rel_v.dot(r_hat)
                    f_d = r_hat * (spring.damping * v_proj)
                    f = f + f_s + f_d
                return f

            ball.clear_forces()   # already accumulated above; RK4 re-derives
            ball.integrate_rk4(dt, force_fn)

    # ── global quantities ──────────────────────────────────────────────────
    @property
    def total_kinetic_energy(self) -> float:
        return sum(b.kinetic_energy for b in self.balls)

    @property
    def total_spring_potential_energy(self) -> float:
        return sum(s.potential_energy for s in self.springs)

    @property
    def total_gravitational_pe(self) -> float:
        """PE_grav = m·g·h  (h measured from bottom of screen)."""
        return sum(
            b.mass * self.gravity * (self.height - b.pos.y)
            for b in self.balls
        )

    @property
    def total_energy(self) -> float:
        return (
            self.total_kinetic_energy
            + self.total_spring_potential_energy
            + self.total_gravitational_pe
        )

    @property
    def total_momentum(self) -> Vec2:
        """System linear momentum p = Σ mᵢvᵢ"""
        px = sum(b.momentum.x for b in self.balls)
        py = sum(b.momentum.y for b in self.balls)
        return Vec2(px, py)

    # ── energy history ─────────────────────────────────────────────────────
    def _record_energy(self) -> None:
        if len(self.time_history) >= self._max_history:
            self.time_history.pop(0)
            self.ke_history.pop(0)
            self.pe_history.pop(0)
            self.te_history.pop(0)

        ke = self.total_kinetic_energy
        pe = self.total_spring_potential_energy + self.total_gravitational_pe
        self.time_history.append(self.sim_time)
        self.ke_history.append(ke)
        self.pe_history.append(pe)
        self.te_history.append(ke + pe)
