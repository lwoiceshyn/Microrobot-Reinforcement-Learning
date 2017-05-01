# myapp.py
from workspace import *

import numpy as np
import time
import math
from threading import Thread

from bokeh.client import push_session
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, Button
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, curdoc

from functools import partial
from tornado import gen


doc = curdoc()

# open a session to keep local doc in sync with server
session = push_session(doc)

particle_source = ColumnDataSource(data=dict(x=[0], y=[0]))
agent_source = ColumnDataSource(data=dict(x=[0], y=[0]))


# @gen.coroutine
def update(particles, agents):
    particle_source.data = particles
    agent_source.data = agents

#------------------------------------------------------------
global box
global Fx
global Fy

box = WorkSpace()
dt = 1. / 1000

def button_reset():
    print('Reset')
    box.reset(preset='A')

box.reset(preset='A')


def animate():
    while True:
        box.step(9, dt)
        doc.add_next_tick_callback(partial(update, particles=box.particles, agents=box.agents))
        time.sleep(dt)

# set up sliders
button = Button(label='Reset')
button.on_click(button_reset)
Fx = Slider(title="Fx", value=0.0, start=-10.0, end=10.0, step=0.1)
Fy = Slider(title="Fy", value=0.0, start=-10.0, end=10.0, step=0.1)

def update_data(attrname, old, new):
    # get current slider values
    box.Fx = Fx.value
    box.Fy = Fy.value

for w in [Fx, Fy]:
    w.on_change('value', update_data)

inputs = widgetbox(button, Fx, Fy)

# setup plot
p1 = figure(plot_width=600, plot_height=600, x_range=(0,10), y_range=(0,10))

particle_source = ColumnDataSource(data=box.particles)
agent_source = ColumnDataSource(data=box.agents)

particles = p1.circle('x', 'y', source=particle_source, radius=box.size[1])
agents = p1.circle('x', 'y', source=agent_source, radius=box.size[0], color="red")

# put the button and plot in a layout and add to the document
doc.add_root(row(inputs, p1))

thread = Thread(target=animate)
thread.daemon = True
thread.start()

# session.show(doc) # open the document in a browser

session.show()

session.loop_until_closed() # run forever