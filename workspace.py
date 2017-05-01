import numpy as np
import time
import math
from scipy.spatial.distance import pdist, squareform
from render import *

class WorkSpace():
    def __init__(self,
                 N = 50,
                 bounds = [0, 10, 0, 10],
                 particle_size = 0.1,
                 agent_size = 0.2,
                 M = 0.5,
                 particle_density = 7850,
                 agent_M = 1,
                 Fx = 0.0,
                 Fy = 0.0,
                 fluid_viscosity=1,
                 fluid_density=1420,
                 dt=0.001):
        self.init_state = self.gen_states(N, particle_size, agent_size)
        # self.M = (4/3*math.pi*size**3) * particle_density * np.ones(self.init_state.shape[0])
        self.M = M * np.ones(self.init_state.shape[0])
        # self.M[0] = (4/3*math.pi*agent_size**3) * particle_density
        self.size = particle_size * np.ones(self.init_state.shape[0])
        self.size[0] = agent_size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.Fx = Fx
        self.Fy = Fy
        self.fluid_viscosity = fluid_viscosity
        self.fluid_density = fluid_density
        self.particles = dict(x=self.state[1:,0], y=self.state[1:,1])
        self.agents = dict(x=[self.state[0,0]], y=[self.state[0,1]])
        self.stepcounter = 0
        self.dt = dt
        self.isrendered = False
        self.reward = 0.
        self.actions = 3
        self.touches = 1
        self.notouches = 1
        self.steps = 0


    def reset(self, preset='random'):
        if preset == 'A':
            side = np.random.randint(2)
            if side == 0:
                self.state = np.array([[1., 1., 0., 0.], [9., 1., 0., 0.]])
            else:
                self.state = np.array([[9., 1., 0., 0.], [1., 1., 0., 0.]])
            self.size = np.ones((2)) * 0.1
            self.reward = 0.
            self.stepcounter = 0
            self.Fx = 0
            self.Fy = 0
            return self.state
        elif preset == 'random':
            self.state = self.gen_states(50, 0.5, 0.8)
        else:
            print(preset + ' is not a valid preset')
            return

        self.reward = 0.
        self.stepcounter = 0
        self.Fx = 0
        self.Fy = 0
    
    def gen_states(self, N, particle_size, agent_size):
        init_state = np.zeros((N,4))
        n = 0
        while n < N:
            pos = np.random.random((4))
            pos *= 10
            if n == 0:
                size = agent_size
            else:
                size = particle_size
            overlap = False
            for i in range(0,N):
                d = pdist((pos[:2], init_state[i,:2]))
                if d < size * 2:
                    overlap = True
            if not overlap:
                init_state[n] = pos
                n += 1

        init_state[:, 2:] = 0
        return init_state
    def resolve_collisions(self):
        # update positions by adding velocities * dt
        self.state[:, :2] += self.dt * self.state[:, 2:]

        # find pairs of particles undergoing a collision
        predicted = self.state[:, :2] + self.dt * self.state[:, 2:]

        # D = squareform(pdist(self.state[:, :2]))
        self.pd = pdist(predicted)
        D = squareform(self.pd)
        overlap = D - self.size - self.size[1]
        ind1, ind2 = np.where(overlap < 0)
        unique = (ind1 > ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        # ref. https://github.com/zjost/pycollide/blob/master/pycollide/pycollide.py
        # ref http://vobarian.com/collisions/2dcollisions2.pdf
        for i1, i2 in zip(ind1, ind2):
            
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vectors
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]
            d_o = pdist((r1, r2))
            # velocity vectors
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # normal vector
            n_hat_p = r2 - r1
            n_hat = n_hat_p / np.linalg.norm(n_hat_p)

            # tangent vector
            t_hat = [-n_hat[1], n_hat[0]]

            # projected velocity vectors
            v1n = np.dot(v1, n_hat)
            v1t = np.dot(v1, t_hat)
            v2n = np.dot(v2, n_hat)
            v2t = np.dot(v2, t_hat)

            v1n_p = (v1n*(m1 - m2) + 2*m2*v2n) / (m1 + m2)
            v2n_p = (v2n*(m2 - m1) + 2*m1*v1n) / (m1 + m2)

            v1_p = np.array([v1n_p, v1t])
            v2_p = np.array([v2n_p, v2t])

            # Define conversion matrix
            A11 = np.dot(np.array([1, 0]), n_hat)
            A12 = np.dot(np.array([1, 0]), t_hat)
            A21 = np.dot(np.array([0, 1]), n_hat)
            A22 = np.dot(np.array([0, 1]), t_hat)
            A = np.array([[A11, A12], [A21, A22]])

            v1_new = np.dot(A, v1_p)
            v2_new = np.dot(A, v2_p)

            predicted1 = r1 + self.dt * v1_new
            predicted2 = r2 + self.dt * v2_new

            d_p = pdist((predicted1, predicted2))

            self.state[i1, 2:] = v1_new
            self.state[i2, 2:] = v2_new

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size[crossed_x1]
        self.state[crossed_x2, 0] = self.bounds[1] - self.size[crossed_x2]

        self.state[crossed_y1, 1] = self.bounds[2] + self.size[crossed_y1]
        self.state[crossed_y2, 1] = self.bounds[3] - self.size[crossed_y2]

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1

    def resolve_drag(self):
        # ref (2.27) from Micro-Scale Mobile Robotics - Eric Diller and Metin Sitti
        # approximation for a sphere at low Reynolds number
        F_dx = self.size * 6 * math.pi * self.fluid_viscosity * self.state[:,2]
        F_dy = self.size * 6 * math.pi * self.fluid_viscosity * self.state[:,3]
        self.state[:,2] -= F_dx / self.M[0] * self.dt
        self.state[:,3] -= F_dy / self.M[0] * self.dt
    
    def resolve_forces(self):
        self.state[0, 2] += self.Fx / self.M[0] * self.dt
        self.state[0, 3] += self.Fy / self.M[0] * self.dt
    
    def step(self, action, dt):
        self.time_elapsed += self.dt
        if action == 1:
            self.Fx = 10.
        elif action == 0:
            self.Fx = -10.

        self.Fx = self.clamp(self.Fx, 10, -10)
        for i in range(int(dt/0.001)):
            self.iter()
            if self.done:
                print(self.touches/(self.touches+self.notouches))
                self.steps = 0
                break
            # else:
                # self.reward = -0.01
        self.steps += 1
        return self.state, self.reward, self.done

    def iter(self, dt=0.01):
        done = False
        self.dt = dt
        """ Step once by dt seconds """
        

        # take care of particle and wall collisions
        self.resolve_collisions()
        
        # add force
        self.resolve_forces()

        # add drag
        self.resolve_drag()

        # setup dictionaries
        if len(self.state[1:,0]) == 1:
            self.particles = dict(x=[self.state[1,0]], y=[self.state[1,1]])
        else:
            self.particles = dict(x=self.state[1:,0], y=self.state[1:,1])
        self.agents = dict(x=[self.state[0,0]], y=[self.state[0,1]])

        # compute reward
        # self.reward = 1/np.min(self.pd)
        # self.reward -= 0.0001

        reward = 0.5/self.pd
        # reward = 0
        self.reward = reward
        
        if np.min(self.pd) < 0.5:
            self.reward = 1. + (200-self.steps)
            # print('TOUCH!')
            self.touches += 1
            self.done = True
        else:
            # compute if done
            if self.stepcounter > (10/self.dt): # 3 seconds
                self.done = True
                # print('no touch...')
                self.notouches +=1
                self.reward = -1.
            else:
                self.done = False

        # update step counter
        self.stepcounter += 1
        
        if self.isrendered:
            time.sleep(self.dt)
        # print(self.Fx)
        

    def render(self):
        if not self.isrendered:
            self.renderer = Render(self)
            self.isrendered = True

    def clamp(self, val, upbound, lobound):
        if val > upbound:
            return upbound
        elif val < lobound:
            return lobound
        else:
            return val
