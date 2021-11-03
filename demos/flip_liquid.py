""" FLIP simulation for liquids

A liquid block collides with a rotated BOX and falls into a liquid pool.
"""

from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


DOMAIN = dict(x=64, y=64, bounds=Box[0:64, 0:64])
GRAVITY = math.tensor([0, -9.81])
DT = 0.1
BOX = Box[1:35, 30:33].rotated(-20)
BOX_MASK = CenteredGrid(BOX, **DOMAIN)
ACCESSIBLE_MASK = field.stagger(CenteredGrid(~BOX, **DOMAIN), math.minimum, extrapolation.ZERO)
_BOX_POINTS = field.nonzero(BOX_MASK).with_color(color='#000000')  # only for plotting

particles = field.nonzero(CenteredGrid(union(Box[15:30, 50:60], Box[:, :5]), **DOMAIN)).split_elements(8) * (0, 0)
velocity = particles @ ACCESSIBLE_MASK
pressure = CenteredGrid(0, **DOMAIN)
scene = particles & _BOX_POINTS * (0, 0)  # only for plotting

for _ in view('scene,velocity,pressure', display='scene', play=False, namespace=globals()).range():
    div_free_velocity, _, occupied = flip.make_incompressible(velocity + DT * GRAVITY, DOMAIN, particles, ACCESSIBLE_MASK)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity)
    particles = advect.runge_kutta_4(particles, div_free_velocity, DT, accessible=ACCESSIBLE_MASK, occupied=occupied)
    particles = flip.respect_boundaries(particles, DOMAIN, [BOX])
    velocity = particles @ ACCESSIBLE_MASK
    scene = particles & _BOX_POINTS * (0, 0)
