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

# Mas, lenght and k constant of the spring

m, k, l = smp.symbols('m k l')

# Define theta and z as functions of time.

theta = smp.Function(r'\theta')(t)
z = smp.Function('z')(t)

# Define first and second derivatives of theta and z.

# First derivatives

theta_d = smp.diff(theta, t)
z_d = smp.diff(z, t)

# Second derivatives

theta_dd = smp.diff(theta_d, t)
z_dd = smp.diff(z_d, t)

# Define the (x-y) position of the particle of the pendulum.

x = (l + z) * smp.sin(theta)
y = - (l + z) * smp.cos(theta)

# Define the velocity of the particle.

vx = smp.diff(x, t)
vy = smp.diff(y, t)

# Net velocity of the particle

v = smp.sqrt((vx**2) + (vy**2))

# Define the Langranian for the system L = K - V
# K = Kinetic energy = 1/2 m v^2
# V = Potential energy = m g y

L = ((smp.Rational(1, 2) * m * (v**2)) - ((m * g * y) + (smp.Rational(1, 2) * k * (z**2)))).simplify()

# Define the equations of motion by applying the Euler-Lagrange equations for theta and z

# Equation for theta

Eq1 = smp.Eq(smp.diff(smp.diff(L, theta_d), t) - smp.diff(L, theta), 0).simplify()

# Equation for z

Eq2 = smp.Eq(smp.diff(smp.diff(L, z_d), t) - smp.diff(L, z), 0).simplify()

# Solve the equations for the second derivatives

sols_dd = smp.solve([Eq1, Eq2], (theta_dd, z_dd), simplify=True, rational=True)

"""Convert the system of two second order ODEs into a system of four first order ODEs
  frac{d theta}{dt} = w_{1}
  frac{dw_{1}}{dt} = frac{d^{2}theta}{dt^{2}} = ...
  frac{dz}{dt} = w_{2}
  frac{dw_{2}}{dt} = frac{d^{2}z}{dt^{2}} = ...
and transform the symbolic equations into numeric functions."""

# w1

dtheta_dt = smp.lambdify(theta_d, theta_d)

# dw1/dt

dw1_dt = smp.lambdify((t, g, m, k, l, theta, z, theta_d, z_d), sols_dd[theta_dd])

# w2

dz_dt = smp.lambdify(z_d, z_d)

# dw2/dt

dw2_dt = smp.lambdify((t, g, m, k, l, theta, z, theta_d, z_d), sols_dd[z_dd])

# Define a vector \vec{S} = (theta, w1, z, w2) and a function that takes \vec{S} and t and returns frac{d\vec{S}}{dt} in order to solve the system of ODEs in python.


def dS_dt(S, t, g, m, k, l):
    theta, w1, z, w2 = S
    return [dtheta_dt(w1), dw1_dt(t, g, m, k, l, theta, z, w1, w2),
            dz_dt(w2), dw2_dt(t, g, m, k, l, theta, z, w1, w2)]


# Define all the needed parameters and initial conditions and solve the system ODEs with the **odeint** method.

# Constant parameters

g = 9.8
m = 30
k = 27
l = 4

# Initial conditions
# iv : initial velocity
# ip : initial position

# Initial conditions for theta

ip_theta = np.pi
iv_theta = 0.5

# Initial conditions for z

ip_z = 2
iv_z = 0.3

# Initial vector

s0 = [ip_theta, iv_theta, ip_z, iv_z]

# it : initial time
# ft : final time
# steps: number of steps

it = 0
ft = 50
steps = 1001

t = np.linspace(it, ft, steps)

# Solve the system of equations

ans = odeint(dS_dt, y0=s0, t=t, args=(g, m, k, l))

# Define a function that returns the position of the particle of the pendulum.


def x_y(t, theta, z, l):
    return((l + z) * np.sin(theta), -(l + z) * np.cos(theta))


# Calculate the positions.

X, Y = x_y(t, ans.T[0], ans.T[2], l)

# Make an animation.
# Use your prefered library for animations.

my_frames = []
for i in range(len(X)-1):
    my_frames.append(go.Frame(data=[go.Scatter(x=[0, X[i + 1]], y=[0, Y[i + 1]])]))

fig = go.Figure(
    data=[go.Scatter(x=[0, X[0]], y=[0, Y[0]])],
    layout=go.Layout(
        xaxis=dict(range=[np.min(X) - 1, np.max(X) + 1]),
        yaxis=dict(range=[np.min(Y) - 1, np.max(Y) + 1]),
        title="Springed Pendulum",
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
