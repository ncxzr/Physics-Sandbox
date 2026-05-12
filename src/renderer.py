"""
renderer.py — Pure rendering layer.  No physics calculations here.

Draws:
  * Simulation background and grid
  * Ball trails
  * Spring connections (colour-coded by tension)
  * Ball circles with velocity arrows
  * HUD overlay (physics values)
  * Drag preview arrow
  * Pause / mode indicators
"""

from __future__ import annotations
import pygame
import math
from src.vector import Vec2
from src.ball import Ball
from src.spring import Spring
from src.world import World

# ── colour palette ─────────────────────────────────────────────────────────
BG_COLOR        = (12,  14,  20)   # near-black background
GRID_COLOR      = (25,  30,  45)   # subtle grid
TEXT_COLOR      = (200, 210, 230)  # main HUD text
ACCENT_COLOR    = (90,  160, 255)  # highlight / title
WARN_COLOR      = (255, 140,  60)  # warnings / pinned
VEL_ARROW_COLOR = (100, 255, 150)  # velocity arrows
SPRING_SLACK    = (80,  180,  80)  # spring at natural length
SPRING_TENSION  = (255, 100,  60)  # spring stretched
SPRING_COMPRESS = (80,  120, 255)  # spring compressed
TRAIL_BASE      = (60,   80, 120)  # trail start color


class Renderer:
    """
    Stateless drawing helper — call draw() each frame.

    Parameters
    ----------
    screen  : pygame.Surface to draw onto
    font_sm, font_md, font_lg : pre-loaded pygame.font.Font objects
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()

        pygame.font.init()
        self.font_sm = pygame.font.SysFont("Courier New", 13)
        self.font_md = pygame.font.SysFont("Courier New", 15, bold=True)
        self.font_lg = pygame.font.SysFont("Courier New", 20, bold=True)

        self.show_velocity_arrows = True
        self.show_trails = True
        self.show_grid = True
        self.show_spring_labels = False

    # ── main draw call ─────────────────────────────────────────────────────
    def draw(
        self,
        world: World,
        drag_start: Vec2 | None = None,
        drag_current: Vec2 | None = None,
        hovered_ball: Ball | None = None,
        selected_ball: Ball | None = None,
        mode: str = "spawn",             # 'spawn', 'spring', 'pin'
        integrator_label: str = "Euler",
    ) -> None:
        self.screen.fill(BG_COLOR)

        if self.show_grid:
            self._draw_grid()

        # ── springs ────────────────────────────────────────────────────────
        for spring in world.springs:
            self._draw_spring(spring)

        # ── balls ──────────────────────────────────────────────────────────
        for ball in world.balls:
            if self.show_trails:
                self._draw_trail(ball)
            self._draw_ball(ball, hovered_ball, selected_ball)
            if self.show_velocity_arrows:
                self._draw_velocity_arrow(ball)

        # ── drag preview ────────────────────────────────────────────────────
        if drag_start and drag_current:
            self._draw_drag_arrow(drag_start, drag_current, mode)

        # ── spring-mode connector preview ───────────────────────────────────
        if mode == "spring" and selected_ball and drag_current:
            pygame.draw.line(
                self.screen,
                (150, 150, 60),
                selected_ball.pos.to_int_tuple(),
                drag_current.to_int_tuple(),
                2,
            )

        # ── HUD ───────────────────────────────────────────────────────────
        self._draw_hud(world, integrator_label, mode)

        if world.paused:
            self._draw_paused_overlay()

    # ── grid ──────────────────────────────────────────────────────────────
    def _draw_grid(self) -> None:
        step = 60
        for x in range(0, self.width, step):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, step):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.width, y))

    # ── spring ────────────────────────────────────────────────────────────
    def _draw_spring(self, spring: Spring) -> None:
        ext = spring.extension
        max_ext = spring.natural_length * 0.5 or 1

        # colour encodes tension (red=stretched, blue=compressed, green=natural)
        t = max(-1.0, min(1.0, ext / max_ext))
        if t > 0:
            color = self._lerp_color(SPRING_SLACK, SPRING_TENSION, t)
        else:
            color = self._lerp_color(SPRING_SLACK, SPRING_COMPRESS, -t)

        a = spring.ball_a.pos.to_int_tuple()
        b = spring.ball_b.pos.to_int_tuple()

        # draw spring as a zig-zag line
        self._draw_zigzag(a, b, color, coils=10, amplitude=8)

        if self.show_spring_labels:
            mid = Vec2(
                (spring.ball_a.pos.x + spring.ball_b.pos.x) / 2,
                (spring.ball_a.pos.y + spring.ball_b.pos.y) / 2,
            )
            lbl = self.font_sm.render(
                f"k={spring.k:.0f}  ζ={spring.damping_ratio:.2f}",
                True, TEXT_COLOR
            )
            self.screen.blit(lbl, mid.to_int_tuple())

    def _draw_zigzag(
        self,
        a: tuple,
        b: tuple,
        color: tuple,
        coils: int = 10,
        amplitude: int = 8,
    ) -> None:
        """Draw a zigzag line simulating a spring coil."""
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return

        # perpendicular unit vector
        px = -dy / length
        py = dx / length

        pts = [(ax, ay)]
        n = coils * 2
        for i in range(1, n):
            t = i / n
            cx = ax + dx * t
            cy = ay + dy * t
            side = amplitude if i % 2 == 1 else -amplitude
            pts.append((int(cx + px * side), int(cy + py * side)))
        pts.append((bx, by))

        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 2)

    # ── ball ──────────────────────────────────────────────────────────────
    def _draw_trail(self, ball: Ball) -> None:
        n = len(ball.trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = i / n
            c = self._lerp_color(BG_COLOR, ball.color, alpha * 0.4)
            p1 = ball.trail[i - 1].to_int_tuple()
            p2 = ball.trail[i].to_int_tuple()
            thickness = max(1, int(alpha * 3))
            pygame.draw.line(self.screen, c, p1, p2, thickness)

    def _draw_ball(
        self,
        ball: Ball,
        hovered: Ball | None,
        selected: Ball | None,
    ) -> None:
        pos = ball.pos.to_int_tuple()
        r = int(ball.radius)

        # outer glow for hovered / selected
        if ball is selected:
            pygame.draw.circle(self.screen, WARN_COLOR, pos, r + 5, 2)
        elif ball is hovered:
            pygame.draw.circle(self.screen, ACCENT_COLOR, pos, r + 3, 1)

        # main body
        pygame.draw.circle(self.screen, ball.color, pos, r)
        # bright rim
        pygame.draw.circle(self.screen, self._lighten(ball.color, 60), pos, r, 2)

        # pinned indicator
        if ball.pinned:
            pygame.draw.line(
                self.screen, WARN_COLOR,
                (pos[0] - r, pos[1] - r), (pos[0] + r, pos[1] + r), 2
            )
            pygame.draw.line(
                self.screen, WARN_COLOR,
                (pos[0] + r, pos[1] - r), (pos[0] - r, pos[1] + r), 2
            )

        # mass label
        lbl = self.font_sm.render(f"{ball.mass:.1f}", True, (20, 20, 30))
        self.screen.blit(lbl, (pos[0] - 8, pos[1] - 6))

    def _draw_velocity_arrow(self, ball: Ball) -> None:
        if ball.speed < 5:
            return
        scale = 0.12
        end = ball.pos + ball.vel * scale
        self._draw_arrow(
            ball.pos.to_int_tuple(),
            end.to_int_tuple(),
            VEL_ARROW_COLOR,
            head_size=8,
        )

    def _draw_drag_arrow(
        self, start: Vec2, end: Vec2, mode: str
    ) -> None:
        color = (255, 80, 80) if mode == "spawn" else ACCENT_COLOR
        self._draw_arrow(
            start.to_int_tuple(), end.to_int_tuple(), color, head_size=12
        )
        # preview ball ghost
        if mode == "spawn":
            pygame.draw.circle(self.screen, color, start.to_int_tuple(), 14, 2)

    # ── arrow helper ──────────────────────────────────────────────────────
    def _draw_arrow(
        self,
        start: tuple,
        end: tuple,
        color: tuple,
        head_size: int = 10,
        width: int = 2,
    ) -> None:
        if start == end:
            return
        pygame.draw.line(self.screen, color, start, end, width)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux   # perpendicular

        tip = end
        base_x = tip[0] - ux * head_size
        base_y = tip[1] - uy * head_size
        left  = (int(base_x + px * head_size * 0.5), int(base_y + py * head_size * 0.5))
        right = (int(base_x - px * head_size * 0.5), int(base_y - py * head_size * 0.5))
        pygame.draw.polygon(self.screen, color, [tip, left, right])

    # ── HUD ──────────────────────────────────────────────────────────────
    def _draw_hud(
        self, world: World, integrator_label: str, mode: str
    ) -> None:
        # ── top-left panel ─────────────────────────────────────────────
        panel_x = 14
        y = 14
        line_h = 18

        def txt(s, color=TEXT_COLOR, font=None):
            nonlocal y
            f = font or self.font_sm
            surf = f.render(s, True, color)
            self.screen.blit(surf, (panel_x, y))
            y += line_h

        txt("INTERACTIVE PHYSICS SANDBOX", ACCENT_COLOR, self.font_md)
        y += 4

        ke   = world.total_kinetic_energy
        pe   = world.total_spring_potential_energy + world.total_gravitational_pe
        te   = ke + pe
        mom  = world.total_momentum

        txt(f"t  = {world.sim_time:7.2f} s")
        txt(f"KE = {ke:9.1f} J")
        txt(f"PE = {pe:9.1f} J")
        txt(f"TE = {te:9.1f} J")
        txt(f"|p| = {mom.magnitude():7.1f} kg·m/s")
        y += 4
        txt(f"g  = {world.gravity:.1f} px/s²")
        txt(f"Balls: {len(world.balls)}   Springs: {len(world.springs)}")
        txt(f"Integrator: {integrator_label}", ACCENT_COLOR)

        # ── bottom-left controls legend ────────────────────────────────
        controls = [
            "─── CONTROLS ─────────────────",
            "LMB drag  : spawn & launch ball",
            "RMB       : delete ball",
            "S + click : spring mode",
            "P + click : pin/unpin ball",
            "SPACE     : pause / resume",
            "R         : reset",
            "↑/↓       : gravity",
            "G/H       : spring constant",
            "D/F       : damping",
            "M/N       : ball mass",
            "E         : toggle integrator",
            "V         : velocity arrows",
            "T         : trails on/off",
        ]
        cy = self.height - len(controls) * 16 - 10
        for line in controls:
            surf = self.font_sm.render(line, True, (100, 115, 140))
            self.screen.blit(surf, (14, cy))
            cy += 16

        # ── mode indicator ────────────────────────────────────────────
        mode_labels = {"spawn": "MODE: SPAWN BALL", "spring": "MODE: ADD SPRING", "pin": "MODE: PIN BALL"}
        mode_colors = {"spawn": (100, 255, 150), "spring": (255, 200, 60), "pin": WARN_COLOR}
        label = mode_labels.get(mode, mode.upper())
        color = mode_colors.get(mode, TEXT_COLOR)
        mode_surf = self.font_md.render(label, True, color)
        self.screen.blit(
            mode_surf, (self.width - mode_surf.get_width() - 14, 14)
        )

    def _draw_paused_overlay(self) -> None:
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 60))
        self.screen.blit(surf, (0, 0))
        lbl = self.font_lg.render("⏸  PAUSED  (SPACE to resume)", True, WARN_COLOR)
        self.screen.blit(
            lbl,
            (self.width // 2 - lbl.get_width() // 2, self.height // 2 - 16),
        )

    # ── colour utilities ──────────────────────────────────────────────────
    @staticmethod
    def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
        t = max(0.0, min(1.0, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    @staticmethod
    def _lighten(color: tuple, amount: int) -> tuple:
        return tuple(min(255, c + amount) for c in color)
