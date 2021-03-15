import dataclasses
import numpy as np
from typing import Optional, List, Tuple, Any, Dict, Union, Callable

#@title Point Class


@dataclasses.dataclass(order=True, frozen=True)
class Point:
    """A class representing a point in 2D space.

    Comes with some convenience functions.
    """
    x: float
    y: float

    def sum(self):
        return self.x + self.y

    def l2norm(self):
        """Computes the L2 norm of the point."""
        return np.sqrt(self.x * self.x + self.y * self.y)

    def __add__(self, other: 'Point'):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point'):
        return Point(self.x - other.x, self.y - other.y)

    def normal_sample_around(self, scale: float):
        """Samples a point around the current point based on some noise."""
        new_coords = np.random.normal(dataclasses.astuple(self), scale)
        new_coords = new_coords.astype(np.float32)
        return Point(*new_coords)

    def is_close_to(self, other: 'Point', diff: float = 1e-4):
        """Determines if one point is close to another."""
        point_diff = self - other
        if abs(point_diff.x) <= diff and abs(point_diff.y) <= diff:
            return True
        else:
            return False

# # Intersection code.
# See Sedgewick, Robert, and Kevin Wayne. Algorithms. , 2011.
# Chapter 6.1 on Geometric Primitives
# https://algs4.cs.princeton.edu/91primitives/


def on_segment(a: Point, b: Point, c: Point):
    x1, x2, x3 = a.x, b.x, c.x

    y1, y2, y3 = a.y, b.y, c.y

    if x1 == x2:
        on_and_between = (x3 == x2) and (y1 <= y3 <= y2)
    else:
        slope = (y2 - y1) / (x2 - x1)

        pt3_on = (y3 - y1) == slope * (x3 - x1)

        pt3_between = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
        on_and_between = pt3_on and pt3_between

    return on_and_between


def _check_counter_clockwise(a: Point, b: Point, c: Point):
    """Checks if 3 points are counter clockwise to each other."""
    slope_AB_numerator = (b.y - a.y)
    slope_AB_denominator = (b.x - a.x)
    slope_AC_numerator = (c.y - a.y)
    slope_AC_denominator = (c.x - a.x)
    return (slope_AC_numerator * slope_AB_denominator >= \
            slope_AB_numerator * slope_AC_denominator)


def intersect(segment_1: Tuple[Point, Point], segment_2: Tuple[Point, Point]):
    """Checks if two line segments intersect."""
    a, b = segment_1
    c, d = segment_2

    if on_segment(a, b, c) or on_segment(a, b, d) or on_segment(c, d, a) or on_segment(c, d, b):
        return True

    # Checking if there is an intersection is equivalent to:
    # Exactly one counter clockwise path to D (from A or B) via C.
    AC_ccw_CD = _check_counter_clockwise(a, c, d)
    BC_ccw_CD = _check_counter_clockwise(b, c, d)
    toD_via_C = AC_ccw_CD != BC_ccw_CD

    # AND
    # Exactly one counterclockwise path from A (to C or D) via B.
    AB_ccw_BC = _check_counter_clockwise(a, b, c)
    AB_ccw_BD = _check_counter_clockwise(a, b, d)

    fromA_via_B = AB_ccw_BC != AB_ccw_BD

    return toD_via_C and fromA_via_B


# Test the points.
z1 = Point(0.4, 0.1)
assert z1.is_close_to(z1)
assert z1.is_close_to(Point(0.5, 0.0), 1.0)
assert not z1.is_close_to(Point(5.0, 0.0), 1.0)
z2 = Point(0.1, 0.1)
z3 = z1 - z2
assert isinstance(z3, Point)
assert z3.is_close_to(Point(0.3, 0.0))
assert isinstance(z3.normal_sample_around(0.1), Point)


# Some simple tests to ensure everything is working.
assert not intersect((Point(1, 0), Point(1, 1)), (Point(0,0), Point(0, 1))), \
  'Parallel lines detected as intersecting.'
assert not intersect((Point(0, 0), Point(1, 0)), (Point(0,1), Point(1, 1))), \
  'Parallel lines detected as intersecting.'
assert intersect((Point(3, 5), Point(1, 1)), (Point(2, 2), Point(0, 1))), \
  'Lines that intersect not detected.'
assert not intersect((Point(0, 0), Point(2, 2)), (Point(3, 3), Point(5, 1))), \
  'Lines that do not intersect detected as intersecting'
assert intersect((Point(0, .5), Point(0, -.5)), (Point(.5, 0), Point(-.5, 0.))), \
  'Lines that intersect not detected.'