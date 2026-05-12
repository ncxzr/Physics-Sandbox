"""
vector.py — 2D Vector class for physics calculations.

Provides clean vector arithmetic used throughout the simulation.
All physics quantities (position, velocity, force, momentum) are Vec2 objects.
"""

import math


class Vec2:
    """
    Immutable-style 2D vector supporting standard arithmetic operations.
    
    Attributes:
        x (float): Horizontal component.
        y (float): Vertical component.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    # ── arithmetic ──────────────────────────────────────────────────────
    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __repr__(self) -> str:
        return f"Vec2({self.x:.3f}, {self.y:.3f})"

    # ── vector operations ────────────────────────────────────────────────
    def dot(self, other: "Vec2") -> float:
        """Scalar dot product: a · b = ax*bx + ay*by"""
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        """Euclidean length: |v| = sqrt(x² + y²)"""
        return math.hypot(self.x, self.y)

    def magnitude_sq(self) -> float:
        """Squared magnitude — avoids sqrt when only comparison is needed."""
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec2":
        """Unit vector in same direction; returns zero vector if magnitude is 0."""
        mag = self.magnitude()
        if mag < 1e-12:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def distance_to(self, other: "Vec2") -> float:
        return (other - self).magnitude()

    def copy(self) -> "Vec2":
        return Vec2(self.x, self.y)

    # ── convenience ──────────────────────────────────────────────────────
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> tuple[int, int]:
        return (int(self.x), int(self.y))
