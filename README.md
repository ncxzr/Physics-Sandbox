# Interactive Physics Sandbox

A 2D physics simulation built in Python for a bachelor-level mechanics and mathematical modeling portfolio.

The project focuses on:
* classical mechanics
* numerical simulation
* collision physics
* spring dynamics

---

## Features

* Gravity simulation
* Elastic collisions
* Spring mechanics with damping
* Real-time physics updates
* Energy and momentum calculations
* Euler and RK4 integration methods
* Interactive object spawning

---

## Physics Concepts

Newton’s Second Law:
F=ma

Hooke’s Law:

genui{"math_block_widget_always_prefetch_v2":{"content":"F=-kx"}}

Momentum Conservation:
m_1u_1+m_2u_2=m_1v_1+m_2v_2

---

## Numerical Methods

The simulation uses:

* Semi-Implicit Euler Integration
* Runge–Kutta 4 (RK4)

for stable real-time motion simulation.

---

## Screenshots

### Gravity Simulation


![Gravity](screenshots/gravity.png)

### Collision Physics



![Collisions](screenshots/collison.png)

### Spring Mechanics



![Springs](screenshots/springs.png)

---

## Demo GIF



![Demo](screenshots/demo.gif)

---

## Project Structure

```text
physics_sandbox/
│
├── main.py
├── requirements.txt
└── src/
    ├── vector.py
    ├── ball.py
    ├── spring.py
    ├── collision.py
    ├── world.py
    ├── renderer.py
    └── plotter.py
```

---




## Technologies Used

* Python
* pygame
* numpy
* matplotlib
