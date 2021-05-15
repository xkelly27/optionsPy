import math
import numpy
import pandas
from turtle import Turtle, Screen

def f(ox, oy, c):
    if c == 0:
        return
    f(ox + 40, oy + 30, c - 1)

    t = Turtle()
    t.penup()
    t.setpos(ox, oy)
    t.pendown()
    t.left(36.86)
    t.forward(50)

    f(ox + 40, oy - 30, c - 1)

    t = Turtle()
    t.penup()
    t.setpos(ox, oy)
    t.pendown()
    t.right(36.86)
    t.forward(50)


f(0, 0, 5)
input()