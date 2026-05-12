"""
main.py

Interaction modes
─────────────────
  SPAWN (default) : Left click and drag
  SPRING          : Click first ball, click second → creates a spring
  PIN             :can pin ball at any state

"""

import sys
import time
import random
import pygame

from src.vector import Vec2
from src.ball import Ball
from src.spring import Spring
from src.world import World, DEFAULT_GRAVITY
from src.renderer import Renderer
from src.plotter import plot_energy, save_energy_plot

# window settings 
WIDTH, HEIGHT = 1200, 780
FPS_TARGET    = 60
TITLE         = "Interactive Physics Sandbox"

# spawn parameters
LAUNCH_SCALE  = 3.0   
BALL_COLORS   = [
    (100, 200, 255), (255, 140,  80), (120, 255, 160),
    (255, 110, 170), (200, 150, 255), (255, 230,  80),
    (80,  220, 220), (255, 160, 100),
]


def random_color() -> tuple:
    return random.choice(BALL_COLORS)


def run():
    pygame.init()
    screen  = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    clock   = pygame.time.Clock()

    world    = World(WIDTH, HEIGHT, gravity=DEFAULT_GRAVITY, integrator="euler")
    renderer = Renderer(screen)

    # mutable simulation parameters
    spawn_mass        = 1.0
    spring_k          = 200.0
    spring_damping    = 5.0
    integrator_label  = "Euler"

    # ── interaction state 
    mode              = "spawn"    # 'spawn' | 'spring' | 'pin'
    drag_start        = None       # Vec2 — where LMB was pressed
    drag_current      = None       # Vec2 — current mouse pos during drag
    spring_first_ball = None       # first ball selected in spring mode
    hovered_ball      = None
    selected_ball     = None       # first ball in spring connection

    def get_ball_at(pos: Vec2) -> Ball | None:
        """Return the topmost ball whose radius covers pos."""
        for ball in reversed(world.balls):
            if ball.pos.distance_to(pos) <= ball.radius + 4:
                return ball
        return None

    def make_demo_scene():
        """Populate an interesting starting scene."""
        # Pendulum: pinned anchor + heavy bob
        anchor = Ball(Vec2(WIDTH // 2, 120), mass=0.1, radius=8,
                      color=(200, 200, 200), restitution=0.3)
        anchor.pinned = True
        bob    = Ball(Vec2(WIDTH // 2 + 180, 280), mass=3.0,
                      color=BALL_COLORS[0], restitution=0.6)
        world.add_ball(anchor)
        world.add_ball(bob)
        world.add_spring(Spring(anchor, bob, k=150, natural_length=220, damping=2))

        # Chain of 3 masses
        cx = WIDTH // 4
        prev = Ball(Vec2(cx, 100), mass=0.1, radius=8, color=(200,200,200))
        prev.pinned = True
        world.add_ball(prev)
        for i in range(3):
            cur = Ball(Vec2(cx + (i + 1) * 80, 100 + (i + 1) * 80),
                       mass=1.5, color=BALL_COLORS[2 + i % 3])
            world.add_ball(cur)
            world.add_spring(Spring(prev, cur, k=300, natural_length=100, damping=4))
            prev = cur

        # A few free balls for collision demo
        for _ in range(4):
            pos = Vec2(
                random.randint(800, 1100),
                random.randint(200, 600),
            )
            vel = Vec2(random.uniform(-200, 200), random.uniform(-100, 100))
            b = Ball(pos, vel, mass=random.uniform(0.8, 2.5), color=random_color())
            world.add_ball(b)

    make_demo_scene()

    # ── main loop ──────────────────────────────────────────────────────────
    running   = True
    prev_time = time.perf_counter()

    while running:
        now = time.perf_counter()
        dt  = now - prev_time
        prev_time = now

        mouse_px = pygame.mouse.get_pos()
        mouse_pos = Vec2(*mouse_px)
        hovered_ball = get_ball_at(mouse_pos)

        # ── events ────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ── mouse button down ─────────────────────────────────────────
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = Vec2(*event.pos)

                if event.button == 1:   # left click
                    if mode == "spawn":
                        drag_start = pos.copy()

                    elif mode == "spring":
                        clicked = get_ball_at(pos)
                        if clicked:
                            if spring_first_ball is None:
                                spring_first_ball = clicked
                                selected_ball     = clicked
                            else:
                                if clicked is not spring_first_ball:
                                    world.add_spring(Spring(
                                        spring_first_ball, clicked,
                                        k=spring_k,
                                        damping=spring_damping,
                                    ))
                                spring_first_ball = None
                                selected_ball     = None

                    elif mode == "pin":
                        clicked = get_ball_at(pos)
                        if clicked:
                            clicked.pinned = not clicked.pinned
                            clicked.clear_trail()

                elif event.button == 3:  # right click — delete
                    target = get_ball_at(Vec2(*event.pos))
                    if target:
                        world.remove_ball(target)

            # ── mouse button up ───────────────────────────────────────────
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and mode == "spawn" and drag_start:
                    end_pos = Vec2(*event.pos)
                    # launch velocity = reverse of drag direction × scale
                    launch_vel = (drag_start - end_pos) * LAUNCH_SCALE
                    b = Ball(
                        drag_start.copy(),
                        vel=launch_vel,
                        mass=spawn_mass,
                        color=random_color(),
                    )
                    world.add_ball(b)
                    drag_start   = None
                    drag_current = None

            # ── mouse motion ──────────────────────────────────────────────
            elif event.type == pygame.MOUSEMOTION:
                if drag_start:
                    drag_current = Vec2(*event.pos)
                if mode == "spring" and spring_first_ball:
                    drag_current = Vec2(*event.pos)

            # ── keyboard ──────────────────────────────────────────────────
            elif event.type == pygame.KEYDOWN:
                k = event.key

                # simulation control
                if k == pygame.K_SPACE:
                    world.paused = not world.paused

                elif k == pygame.K_r:
                    world.clear()
                    spring_first_ball = None
                    selected_ball     = None
                    drag_start        = None
                    drag_current      = None
                    make_demo_scene()

                # gravity
                elif k == pygame.K_UP:
                    world.gravity = min(world.gravity + 50, 2000)
                elif k == pygame.K_DOWN:
                    world.gravity = max(world.gravity - 50, 0)

                # spring constant
                elif k == pygame.K_g:
                    spring_k = max(20, spring_k - 20)
                elif k == pygame.K_h:
                    spring_k = min(2000, spring_k + 20)

                # damping
                elif k == pygame.K_d:
                    spring_damping = max(0, spring_damping - 1)
                elif k == pygame.K_f:
                    spring_damping = min(100, spring_damping + 1)

                # spawn mass
                elif k == pygame.K_m:
                    spawn_mass = min(spawn_mass * 1.5, 20.0)
                elif k == pygame.K_n:
                    spawn_mass = max(spawn_mass / 1.5, 0.2)

                # integrator toggle
                elif k == pygame.K_e:
                    if world.integrator == "euler":
                        world.integrator = "rk4"
                        integrator_label = "RK4"
                    else:
                        world.integrator = "euler"
                        integrator_label = "Euler"

                # display toggles
                elif k == pygame.K_v:
                    renderer.show_velocity_arrows = not renderer.show_velocity_arrows
                elif k == pygame.K_t:
                    renderer.show_trails = not renderer.show_trails

                # mode switches
                elif k == pygame.K_s:
                    mode = "spring" if mode != "spring" else "spawn"
                    spring_first_ball = None
                    selected_ball     = None
                elif k == pygame.K_p:
                    mode = "pin" if mode != "pin" else "spawn"

                # ESC → back to spawn mode
                elif k == pygame.K_ESCAPE:
                    mode = "spawn"
                    spring_first_ball = None
                    selected_ball     = None
                    drag_start        = None

                # energy plot (C = show, X = save PNG)
                elif k == pygame.K_c:
                    was_paused = world.paused
                    world.paused = True
                    plot_energy(world)   # blocks until window closed
                    world.paused = was_paused

                elif k == pygame.K_x:
                    save_energy_plot(world, "graphs/energy_plot.png")

        # ── physics step ──────────────────────────────────────────────────
        world.step(dt)

        # ── render ────────────────────────────────────────────────────────
        renderer.draw(
            world,
            drag_start=drag_start,
            drag_current=drag_current,
            hovered_ball=hovered_ball,
            selected_ball=selected_ball,
            mode=mode,
            integrator_label=integrator_label,
        )

        pygame.display.flip()
        clock.tick(FPS_TARGET)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()
