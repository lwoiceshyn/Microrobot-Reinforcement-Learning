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


class Render():
    def __init__(self, box):
        self.doc = curdoc()
        self.box = box

        # open a session to keep local doc in sync with server
        self.session = push_session(self.doc)

        self.particle_source = ColumnDataSource(data=dict(x=[0], y=[0]))
        self.agent_source = ColumnDataSource(data=dict(x=[0], y=[0]))

        # @gen.coroutine

        # setup plot
        self.p = figure(plot_width=600, plot_height=600, x_range=(0,10), y_range=(0,10))

        self.particle_source = ColumnDataSource(data=box.particles)
        self.agent_source = ColumnDataSource(data=box.agents)

        self.particles = self.p.circle('x', 'y', source=self.particle_source, radius=box.size[1])
        self.agents = self.p.circle('x', 'y', source=self.agent_source, radius=box.size[0], color="red")

        # put the button and plot in a layout and add to the document
        self.doc.add_root(row(self.p))

        # self.thread2 = Thread(target=self.animate)
        # self.thread2.daemon = True
        # self.thread2.start()
        # self.animate()
        self.doc.add_periodic_callback(self.update,50)
        # session.show(doc) # open the document in a browser
        # self.animate()
        self.session.show()

        thread = Thread(target=self.session.loop_until_closed)
        thread.daemon=True
        thread.start()

    def update(self):
        self.particle_source.data = self.box.particles
        self.agent_source.data = self.box.agents
        self.done = True
    
    #------------------------------------------------------------

    def animate(self):
        while True:
            # self.done = False
            self.doc.add_next_tick_callback(partial(self.update, particles=self.box.particles, agents=self.box.agents))
            # self.doc.add_periodic_callback(partial(self.update, particles=self.box.particles, agents=self.box.agents),50)