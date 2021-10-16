# Numeric calculation libraries

import numpy as np
from scipy.integrate import odeint

# Symbolic calculation library

import sympy as smp

# Graph libraries

import plotly.graph_objects as go

# Define all the symbols needed to represent the problem.

# time and acceleration due to gravity constant (g)

t, g = smp.symbols('t g')

# Masses of the two particles of the pendulum

M1, M2 = smp.symbols('M1 M2')

# Lenghts of the two massless bars that joins the particles

l1, l2 = smp.symbols('l1 l2')

# Define theta1 and theta2 as functions of time.

theta1 = smp.Function(r'\theta_1')(t)
theta2 = smp.Function(r'\theta_2')(t)

# Define first and second derivatives of theta1 and theta2.

# First derivatives

theta1_d = smp.diff(theta1, t)
theta2_d = smp.diff(theta2, t)

# Second derivatives

theta1_dd = smp.diff(theta1_d, t)
theta2_dd = smp.diff(theta2_d, t)

# Define the (x-y) position of the particles of the pendulum.

# Position of particle 1 with mass M1

x1 = l1 * smp.sin(theta1)
y1 = -l1 * smp.cos(theta1)

# Position of particle 2 with mass M2

x2 = x1 + (l2 * smp.sin(theta2))
y2 = y1 - (l2 * smp.cos(theta2))

# Define the velocity of the particles.

# Velocity of particle 1

vx1 = smp.diff(x1, t)
vy1 = smp.diff(y1, t)

# Net velocity of particle 1

v1 = smp.sqrt((vx1**2) + (vy1**2))

# Velocity of particle 2

vx2 = smp.diff(x2, t)
vy2 = smp.diff(y2, t)

# Net velocity of particle 2

v2 = smp.sqrt((vx2**2) + (vy2**2))

# Define the Langranian for the system L = K - V
# K = Kinetic energy = 1/2 m v^2
# V = Potential energy = m g y
# The Langranian of the system is the sum of the Langranians of the particles.

# Langranian of particle 1

L1 = (smp.Rational(1, 2) * M1 * (v1**2)) - (M1 * g * y1)

# Langranian of particle 2

L2 = (smp.Rational(1, 2) * M2 * (v2**2)) - (M2 * g * y2)

# Langranian of the system

L = (L1 + L2).simplify()

# Define the equations of motion by applying the Euler-Lagrange equations for theta1 and theta2

# Equation for theta1

Eq1 = smp.Eq(smp.diff(smp.diff(L, theta1_d), t) - smp.diff(L, theta1), 0).simplify()

# Equation for theta2

Eq2 = smp.Eq(smp.diff(smp.diff(L, theta2_d), t) - smp.diff(L, theta2), 0).simplify()

# Solve the equations for the second derivatives

sols_dd = smp.solve([Eq1, Eq2], (theta1_dd, theta2_dd), simplify=True, rational=True)

"""Convert the system of two second order ODEs into a system of four first order ODEs
  frac{d theta1}{dt} = z_{1}
  frac{dz_{1}}{dt} = frac{d^{2}theta1}{dt^{2}} = ...
  frac{d theta2}{dt} = z_{2}
  frac{dz_{2}}{dt} = frac{d^{2}theta2}{dt^{2}} = ...
and transform the symbolic equations into numeric functions."""

# z1

dtheta1_dt = smp.lambdify(theta1_d, theta1_d)

# dz1/dt

dz1_dt = smp.lambdify((t, g, M1, M2, l1, l2, theta1, theta2, theta1_d, theta2_d), sols_dd[theta1_dd])

# z2

dtheta2_dt = smp.lambdify(theta2_d, theta2_d)

# dz2/dt

dz2_dt = smp.lambdify((t, g, M1, M2, l1, l2, theta1, theta2, theta1_d, theta2_d), sols_dd[theta2_dd])

# Define a vector \vec{S} = (theta, w1, z, w2) and a function that takes \vec{S} and t and returns frac{d\vec{S}}{dt} in order to solve the system of ODEs in python.


def dS_dt(S, t, g, M1, M2, l1, l2):
    theta1, z1, theta2, z2 = S
    return [dtheta1_dt(z1), dz1_dt(t, g, M1, M2, l1, l2, theta1, theta2, z1, z2),
            dtheta2_dt(z2), dz2_dt(t, g, M1, M2, l1, l2, theta1, theta2, z1, z2)]


# Define all the needed parameters and initial conditions and solve the system ODEs with the **odeint** method.

# Constant parameters

g = 9.8
M1 = 3
M2 = 5
l1 = 2.3
l2 = 0.7

# Initial conditions
# iv : initial velocity
# ip : initial position

# Initial conditions for theta1

ip_theta1 = np.pi
iv_theta1 = 0

# Initial conditions for theta2

ip_theta2 = 0.1 * np.pi
iv_theta2 = 0

# Initial vector

s0 = [ip_theta1, iv_theta1, ip_theta2, iv_theta2]

# it : initial time
# ft : final time
# steps: number of steps

it = 0
ft = 40
steps = 1001

t = np.linspace(it, ft, steps)

# Solve the system of equations

ans = odeint(dS_dt, y0=s0, t=t, args=(g, M1, M2, l1, l2))

# Define a function that returns the position of the particles of the pendulum.


def x1_y1_x2_y2(t, theta1, theta2, l1, l2):
    return(l1 * np.sin(theta1), -l1 * np.cos(theta1), (l1 * np.sin(theta1)) + (l2 * np.sin(theta2)), (-l1 * np.cos(theta1)) - (l2 * np.cos(theta2)))


# Calculate the positions.

X1, Y1, X2, Y2 = x1_y1_x2_y2(t, ans.T[0], ans.T[2], l1, l2)

# Make an animation.
# Use your prefered library for animations.

my_frames = []
for i in range(len(X1) - 1):
    my_frames.append(go.Frame(data=[go.Scatter(x=[0, X1[i + 1]], y=[0, Y1[i + 1]]), go.Scatter(x=[X1[i + 1], X2[i + 1]], y=[Y1[i + 1], Y2[i + 1]])]))

fig = go.Figure(
    data=[go.Scatter(x=[0, X1[0]], y=[0, Y1[0]]), go.Scatter(x=[X1[0], X2[0]], y=[Y1[0], Y2[0]])],
    layout=go.Layout(
        xaxis=dict(range=[-l1 - l2 - 0.3, l1 + l2 + 0.3]),
        yaxis=dict(range=[-l1 - l2 - 0.3, l1 + l2 + 0.3]),
        title="Double Pendulum",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": ft, "redraw": False}, "fromcurrent": True, "transition": {"duration": 1 / len(t[t <= 1])}}])])]
    ),
    frames=my_frames
)

fig.update_layout(width=800, height=800)

fig.show()
