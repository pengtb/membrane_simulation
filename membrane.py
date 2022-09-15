import mesa
from scheduler import SimultaneousActivation, SimpleSimultaneousActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from lipids import Lipid_allinter, Lipid_simple
import numpy as np
from datacollector import JumpDataCollector, JumpModelDataCollector
from force import attractive_force, repulsive_force, friction_force, pull_force, constant_velocity, zero_acceleration
import jax.numpy as jnp
from jit_functions import *

def initialize_shape_positions(N, init_shape, **kwargs):
    if init_shape == 'circle':
        radius = kwargs.get('radius', 6)
        positions = [np.array([np.cos(2 * np.pi / N * i), np.sin(2 * np.pi / N * i)]) * radius for i in range(N)]
    elif init_shape == 'square':
        assert np.sqrt(N) % 1 == 0
        edge_size = np.sqrt(N)
        positions = [np.array([i // edge_size, i % edge_size]) for i in range(N)] - np.array([edge_size / 2, edge_size / 2])
    elif init_shape == 'oval':
        radius = kwargs.get('radius', 6)
        long_radius = kwargs.get('long_radius', 8)
        positions = [np.array([np.cos(2 * np.pi / N * i) * long_radius, np.sin(2 * np.pi / N * i) * radius]) for i in range(N)]
    elif init_shape == 'polygon':
        distance = kwargs.get('distance', 0.375)
        angle = 2 * np.pi / N
        radius = distance / 2 / np.sin(angle / 2)
        positions = [np.array([np.cos(2 * np.pi / N * i), np.sin(2 * np.pi / N * i)]) * radius for i in range(N)]
    else:
        raise ValueError('Unknown init_shape')
    return positions

# area of a polygon
def polygon_area(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class Membrane(mesa.Model):
    def __init__(self, N, lipid=Lipid_allinter, **kwargs):
        # initialize model constants
        self.N = N
        self.r0 = kwargs.get('r0', 1)
        self.k1 = kwargs.get('k1', 1)
        self.k2 = kwargs.get('k2', 2)
        self.k3 = kwargs.get('k3', 10)
        self.dt = kwargs.get('dt', 0.1)
        self.power = kwargs.get('power', 1)
        space_radius = kwargs.get('space_radius', 100)
        init_shape = kwargs.get('init_shape', 'circle')
        self.update_area = kwargs.get('update_area', True)
        self.update_neighbor = kwargs.get('update_neighbor', True)
        self.neighbor_distance_cutoff = kwargs.get('neighbor_distance_cutoff', 3*self.r0)
        self.update_perimeter = kwargs.get('update_perimeter', True)
        self.update_rg = kwargs.get('update_rg', True)
        self.jump_step = kwargs.get('jump_step', 1)

        # initialize space
        self.space = ContinuousSpace(space_radius, space_radius, True, -space_radius, -space_radius)
        # initialize schedule
        self.schedule = SimultaneousActivation(self)
        # initialize positions of lipids
        if init_shape == 'random':
            positions = [(self.random.randrange(space_radius), self.random.randrange(space_radius)) for i in range(N)]
        else:
            positions = initialize_shape_positions(N, init_shape, **kwargs)
        self.init_pos = np.array(positions)
        # create lipids
        for i in range(N):
            a = lipid(i, self)
            self.schedule.add(a)
            # put lipids in space
            self.space.place_agent(a, positions[i])
        
        # reporters
        model_reporters = {}
        if self.update_area:
            model_reporters['area'] = 'area'
        if self.update_perimeter:
            model_reporters['perimeter'] = 'perimeter'
        if self.update_rg:
            model_reporters['rg'] = 'rg'
        if self.update_area or self.update_rg:
            model_reporters['center_x'] = lambda m: m.center[0]
            model_reporters['center_y'] = lambda m: m.center[1]
        # self.datacollector.model_reporters = model_reporters
        self.datacollector = JumpDataCollector(agent_reporters={"pos_x":lambda a: a.pos[0], 
                                                            'pos_y':lambda a: a.pos[1], 
                                                            'vx':lambda a: a.velocity[0], 
                                                            'vy':lambda a: a.velocity[1], 
                                                            'fx':lambda a: a.total_force[0], 
                                                            'fy':lambda a: a.total_force[1]},
                                                model_reporters=model_reporters,
                                                jump_step=self.jump_step)

        # batch run
        self.running = True
        
        # initialize area
        if self.update_area:
            self.area = polygon_area(self.init_pos[:,0], self.init_pos[:,1])
            self.area0 = self.area
            self.center = np.mean(positions, axis=0)

        # initialize neighborhood matrix
        self.neighborhood = np.eye(N, k=1, dtype=bool) + np.eye(N, k=-1, dtype=bool) + \
            np.eye(N, k=N-1, dtype=bool) + np.eye(N, k=-N+1, dtype=bool)

        # initialize perimeter
        if self.update_perimeter:
            self.perimeter = np.sum(np.linalg.norm(np.roll(self.init_pos, -1, axis=0) - self.init_pos, axis=-1))
        
        # initialize rg
        if self.update_rg:
            if not hasattr(self, 'center'):
                self.center = np.mean(positions, axis=0)
            self.rg = np.linalg.norm(self.init_pos[0] - self.center, axis=-1)

    def step(self, **kwargs):
        # speed limit
        vlim = kwargs.get('vlim', None)
        if vlim is not None:
            for i in range(self.N):
                v = self.schedule.agents[i].velocity
                v_size = np.linalg.norm(v)
                if v_size > vlim:
                    self.schedule.agents[i].velocity = v / v_size * vlim

        # run a step
        aristotelian = kwargs.get('aristotelian', False)
        self.schedule.step(update_neighbor=self.update_neighbor,  aristotelian=aristotelian, **kwargs)
        self.datacollector.collect(self)

        lipids = self.schedule.agents
        # get all lipids' positions
        positions = np.stack([lipid.pos for lipid in lipids], axis=0)

        # update area
        if self.update_area:
            self.area = polygon_area(positions[:, 0], positions[:, 1])
            self.center = np.mean(positions, axis=0)

        # udpate neighborhood matrix
        if self.update_neighbor:
            # consider lipids within the distance cutoff as neighbors
            distance_cutoff = self.neighbor_distance_cutoff
            # distance matrix
            distance_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            # get neighbors
            self.neighborhood = distance_matrix <= distance_cutoff
        
        # update perimeter
        if self.update_perimeter:
            self.perimeter = np.sum(np.linalg.norm(np.roll(positions, -1, axis=0) - positions, axis=-1))

        # update rg
        if self.update_rg:
            self.center = np.mean(positions, axis=0)
            self.rg = np.linalg.norm(np.linalg.norm(positions - self.center, axis=-1), axis=0) / np.sqrt(self.N)
            
    def stopall(self):
        for i in range(self.N):
            self.schedule.agents[i].stop()

    # enlarge the area
    def enlarge(self, velocity_factor=1.):
        for i in range(self.N):
            self.schedule.agents[i].push_out(velocity_factor)

# model with step method vectorized with jax
class Membrane_jax(mesa.Model):
    def __init__(self, N, lipid=Lipid_simple, **kwargs):
        # initialize model constants
        self.N = N
        self.r0 = kwargs.get('r0', 1)
        # forces constants
        self.k1 = kwargs.get('k1', None)
        self.k2 = kwargs.get('k2', None)
        self.k3 = kwargs.get('k3', None)
        self.epsilon = kwargs.get('epsilon', None)
        self.power = kwargs.get('power', 1)
        # time step
        self.dt = kwargs.get('dt', 0.1)
        # create lipids
        self.agent_ids = np.arange(self.N)
        self.lipids = [lipid(idx, self) for idx in self.agent_ids]
        self.mass = jnp.asarray([agent.mass for agent in self.lipids], dtype=jnp.float64)

        # initialize positions of lipids
        init_shape = kwargs.get('init_shape', 'circle')
        if init_shape == 'random':
            space_radius = kwargs.get('space_radius', 100)
            positions = [(self.random.randrange(space_radius), self.random.randrange(space_radius)) for i in range(N)]
        else:
            positions = initialize_shape_positions(N, **kwargs)
        # place lipids
        self.agent_positions = jnp.asarray(positions, dtype=jnp.float64)
        self.init_pos = self.agent_positions

        # initialize neighborhood matrix
        self.all_neighbor = kwargs.get('all_neighbor', False)
        self.num_neighbor = kwargs.get('num_neighbor', 1)
        if not self.all_neighbor:
            self.agent_neighborhood = jnp.zeros((N, N), dtype=bool)
            for idx in range(1, self.num_neighbor+1):
                self.agent_neighborhood |= jnp.eye(N, k=idx, dtype=bool)
                self.agent_neighborhood |= jnp.eye(N, k=-idx, dtype=bool)
                self.agent_neighborhood |= jnp.eye(N, k=N-idx, dtype=bool)
                self.agent_neighborhood |= jnp.eye(N, k=idx-N, dtype=bool)
        else:
            self.agent_neighborhood = jnp.ones((N, N), dtype=bool) ^ jnp.eye(N, k=0, dtype=bool)
        self.distance_matrix = DistanceMatrix(self.agent_positions)
        
        # initialize velocities
        self.velocities = jnp.zeros((N, 2), dtype=jnp.float64)

        # initialize schedule
        self.schedule = SimpleSimultaneousActivation(self)

        # initialize output status
        self.init_status(**kwargs)

        # if model is simple
        self.simple = kwargs.get('simple', True)
        
    def init_status(self, **kwargs):
        # status output options
        self.update_area = kwargs.get('update_area', True)
        self.update_perimeter = kwargs.get('update_perimeter', True)
        self.update_rg = kwargs.get('update_rg', True)
        self.update_r_mcc = kwargs.get('update_r_mcc', True)
        self.jump_step = kwargs.get('jump_step', 1)

        # initialize area
        if self.update_area:
            self.area = PolygonArea(self.agent_positions)
            self.area0 = self.area
            self.center = jnp.mean(self.agent_positions, axis=0)
        # initialize perimeter
        if self.update_perimeter:
            self.perimeter = Perimeter(self.agent_positions)
        # initialize rg
        if self.update_rg:
            if not hasattr(self, 'center'):
                self.center = jnp.mean(self.agent_positions, axis=0)
            self.rg = RadiusofGyration(self.agent_positions, self.center, self.N)
        # initilize r_mcc
        if self.update_r_mcc:
            if not hasattr(self, 'distance_matrix'):
                self.distance_matrix = DistanceMatrix(self.agent_positions)
            self.r_mcc = self.distance_matrix.max() / 2

        # reporters
        model_reporters = {}
        if self.update_area:
            model_reporters['area'] = 'area'
        if self.update_perimeter:
            model_reporters['perimeter'] = 'perimeter'
        if self.update_rg:
            model_reporters['rg'] = 'rg'
        if self.update_area or self.update_rg:
            model_reporters['center_x'] = lambda m: m.center[0]
            model_reporters['center_y'] = lambda m: m.center[1]
        if self.update_r_mcc:
            model_reporters['r_mcc'] = 'r_mcc'
        
        agent_reporters = {}
        agent_reporters['pos_x'] = lambda m: m.agent_positions[:,0]
        agent_reporters['pos_y'] = lambda m: m.agent_positions[:,1]
        agent_reporters['vx'] = lambda m: m.velocities[:,0]
        agent_reporters['vy'] = lambda m: m.velocities[:,1]
        agent_reporters['fx'] = lambda m: m.forces[:,0]
        agent_reporters['fy'] = lambda m: m.forces[:,1]

        self.datacollector = JumpModelDataCollector(self,
                                                    model_reporters=model_reporters,
                                                    agent_reporters=agent_reporters,
                                                    jump_step=self.jump_step)

    def step(self, **kwargs):
        self.schedule.step(**kwargs)

        # speed limit
        vlim = kwargs.get('vlim', None)
        if vlim is not None:
            v_size = jnp.linalg.norm(self.velocities, axis=-1)
            v_size_cliped = jnp.minimum(v_size, vlim)
            self.velocities = self.velocities * v_size_cliped[:, None] / v_size[:]
        vmin = kwargs.get('vmin', None)
        if vmin is not None:
            self.velocities = LowestVelocity(self.velocities, vmin)
            
        # run a step
        # update neighborhood
        self.update_neighborhood(**kwargs)

        # update forces
        total_forces = self.update_forces(**kwargs)

        # update positions
        acceleration = total_forces / self.mass[:, None]
        c_velocity = kwargs.get('constant_velocity', None)
        if c_velocity is not None:
            if (not self.simple) | (self.schedule.steps == 1):
                self.zero_acc_arr = jnp.asarray(zero_acceleration(0, self.N))
                self.constant_velocity = jnp.asarray(constant_velocity(0, self.N, c_velocity))
            self.velocities = self.velocities * self.zero_acc_arr + self.constant_velocity
            acceleration = acceleration * self.zero_acc_arr
        c_acceleration = kwargs.get('constant_acceleration', None)
        if c_acceleration is not None:
            acceleration = acceleration * zero_acceleration(0, self.N) + constant_velocity(0, self.N, c_acceleration)

        movement = self.velocities * self.dt + acceleration * self.dt ** 2 / 2
        self.agent_positions = self.agent_positions + movement

        # update velocities
        self.velocities = self.velocities + acceleration * self.dt

        # update other status
        self.update_status()

        # collect data
        self.datacollector.collect(self)
            
    def stopall(self):
        self.velocities = jnp.zeros((self.N, 2), dtype=jnp.float64)

    def update_status(self):
        # update area
        if self.update_area:
            self.area = PolygonArea(self.agent_positions)
            self.center = jnp.mean(self.agent_positions, axis=0)
        # update perimeter
        if self.update_perimeter:
            self.perimeter = Perimeter(self.agent_positions)
        # update rg
        if self.update_rg:
            if not self.update_area:
                self.center = jnp.mean(self.agent_positions, axis=0)
            self.rg = RadiusofGyration(self.agent_positions, self.center, self.N)
        # update r_mcc
        if self.update_r_mcc:
            if not hasattr(self, 'distance_matrix'):
                self.distance_matrix = DistanceMatrix(self.agent_positions)
            self.r_mcc = self.distance_matrix.max() / 2
    
    def update_neighborhood(self, **kwargs):
        self.update_neighbor = kwargs.get('update_neighbor', True)
        self.neighbor_distance_cutoff = kwargs.get('neighbor_distance_cutoff', 2.5*self.r0)
        # distance matrix
        self.distance_matrix = DistanceMatrix(self.agent_positions)
        # udpate neighborhood matrix
        if self.update_neighbor:
            # consider lipids within the distance cutoff as neighbors
            distance_cutoff = self.neighbor_distance_cutoff
            # get neighbors
            self.agent_neighborhood = self.distance_matrix <= distance_cutoff

    def update_forces(self, **kwargs):
        # only consider neighbors
        relative_positions = RelativePositions(self.agent_positions)
        neighbors = self.agent_neighborhood[:, :, None].astype(jnp.float64)
        other_forces = jnp.zeros((self.N, 2), dtype=jnp.float64)
        distance_matrixs = self.distance_matrix[:,:,None]
        if self.k1 is not None:
            # attractive force
            attr_force = AttractiveForce(relative_positions, neighbors, distance_matrixs, k=self.k1, r0=self.r0, power=self.power).sum(axis=0)
            other_forces += attr_force
        if self.k2 is not None:
            # repulsive force
            repl_force = RepulsiveForce(relative_positions, neighbors, distance_matrixs, k=self.k2, r0=self.r0, power=self.power).sum(axis=0)
            other_forces += repl_force
        if self.epsilon is not None:
            # vdW force
            vdw_force = vdWForce(relative_positions, neighbors, distance_matrixs, k=self.epsilon, r0=self.r0).sum(axis=0)
            other_forces += vdw_force
        # pull force
        pull_force_factor = kwargs.get('pull_force_factor', 1e-5)
        if pull_force_factor is not None:
            pull_f = pull_force(0, self.N, pull_force_factor)
            if (not self.simple) | (self.schedule.steps == 1):
                self.pull_f = jnp.asarray(pull_f)
            other_forces += self.pull_f
        # friction force
        friction_force_factor = kwargs.get('friction_force_factor', 0)
        if friction_force_factor is not None:
            fric_force = FrictionForce(self.velocities, other_forces, force=friction_force_factor)
            total_forces = other_forces + fric_force
        else:
            total_forces = other_forces        
        self.forces = total_forces

        return total_forces

# model with step method vectorized 
class Membrane_vec(mesa.Model):
    def __init__(self, N, lipid=Lipid_simple, **kwargs):
        # initialize model constants
        self.N = N
        self.r0 = kwargs.get('r0', 1)
        # forces constants
        self.k1 = kwargs.get('k1', 1)
        self.k2 = kwargs.get('k2', 2)
        self.k3 = kwargs.get('k3', 10)
        self.power = kwargs.get('power', 1)
        # time step
        self.dt = kwargs.get('dt', 0.1)
        # create lipids
        self.agent_ids = np.arange(self.N)
        self.lipids = [lipid(idx, self) for idx in self.agent_ids]
        self.mass = np.asarray([agent.mass for agent in self.lipids], dtype=np.float32)

        # initialize positions of lipids
        init_shape = kwargs.get('init_shape', 'circle')
        if init_shape == 'random':
            space_radius = kwargs.get('space_radius', 100)
            positions = [(self.random.randrange(space_radius), self.random.randrange(space_radius)) for i in range(N)]
        else:
            positions = initialize_shape_positions(N, init_shape, **kwargs)
        # place lipids
        self.agent_positions = np.asarray(positions, dtype=np.float32)
        self.init_pos = self.agent_positions

        # initialize neighborhood matrix
        self.agent_neighborhood = np.eye(N, k=1, dtype=bool) + np.eye(N, k=-1, dtype=bool) + \
            np.eye(N, k=N-1, dtype=bool) + np.eye(N, k=-N+1, dtype=bool)
        self.distance_matrix = np.linalg.norm(self.agent_positions[:, None, :] - self.agent_positions[None, :, :], axis=-1)
        
        # initialize velocities
        self.velocities = np.zeros((N, 2), dtype=np.float32)

        # initialize schedule
        self.schedule = SimpleSimultaneousActivation(self)

        # initialize output status
        self.init_status(**kwargs)
        
    def init_status(self, **kwargs):
        # status output options
        self.update_area = kwargs.get('update_area', True)
        self.update_perimeter = kwargs.get('update_perimeter', True)
        self.update_rg = kwargs.get('update_rg', True)
        self.update_r_mcc = kwargs.get('update_r_mcc', True)
        self.jump_step = kwargs.get('jump_step', 1)

        # initialize area
        if self.update_area:
            self.area = polygon_area(self.init_pos[:,0], self.init_pos[:,1])
            self.area0 = self.area
            self.center = np.mean(self.init_pos, axis=0)
        # initialize perimeter
        if self.update_perimeter:
            self.perimeter = np.sum(np.linalg.norm(np.roll(self.init_pos, -1, axis=0) - self.init_pos, axis=-1))
        # initialize rg
        if self.update_rg:
            if not hasattr(self, 'center'):
                self.center = np.mean(self.init_pos, axis=0)
            self.rg = np.linalg.norm(self.init_pos[0] - self.center, axis=-1)
        # initilize r_mcc
        if self.update_r_mcc:
            if not hasattr(self, 'distance_matrix'):
                self.distance_matrix = np.linalg.norm(self.init_pos[:, None, :] - self.init_pos[None, :, :], axis=-1)
            self.r_mcc = self.distance_matrix.max() / 2

        # reporters
        model_reporters = {}
        if self.update_area:
            model_reporters['area'] = 'area'
        if self.update_perimeter:
            model_reporters['perimeter'] = 'perimeter'
        if self.update_rg:
            model_reporters['rg'] = 'rg'
        if self.update_area or self.update_rg:
            model_reporters['center_x'] = lambda m: m.center[0]
            model_reporters['center_y'] = lambda m: m.center[1]
        if self.update_r_mcc:
            model_reporters['r_mcc'] = 'r_mcc'
        
        agent_reporters = {}
        agent_reporters['pos_x'] = lambda m: m.agent_positions[:,0]
        agent_reporters['pos_y'] = lambda m: m.agent_positions[:,1]
        agent_reporters['vx'] = lambda m: m.velocities[:,0]
        agent_reporters['vy'] = lambda m: m.velocities[:,1]
        agent_reporters['fx'] = lambda m: m.forces[:,0]
        agent_reporters['fy'] = lambda m: m.forces[:,1]

        self.datacollector = JumpModelDataCollector(self,
                                                    model_reporters=model_reporters,
                                                    agent_reporters=agent_reporters,
                                                    jump_step=self.jump_step)

    def step(self, **kwargs):
        self.schedule.step(**kwargs)

        # speed limit
        vlim = kwargs.get('vlim', None)
        if vlim is not None:
            v_size = np.linalg.norm(self.velocities, axis=-1)
            v_size_cliped = np.minimum(v_size, vlim)
            self.velocities = self.velocities * v_size_cliped[:, None] / v_size[:]
            
        # run a step
        # update neighborhood
        self.update_neighborhood(**kwargs)

        # update forces
        total_forces = self.update_forces(**kwargs)

        # update positions
        acceleration = total_forces / self.mass[:, None]
        movement = self.velocities * self.dt + acceleration * self.dt ** 2 / 2
        self.agent_positions = self.agent_positions + movement

        # update velocities
        self.velocities = self.velocities + acceleration * self.dt

        # update other status
        self.update_status()

        # collect data
        self.datacollector.collect(self)
            
    def stopall(self):
        self.velocities = np.zeros((self.N, 2), dtype=np.float32)

    def update_status(self):
        # update area
        if self.update_area:
            self.area = polygon_area(self.agent_positions[:, 0], self.agent_positions[:, 1])
            self.center = np.mean(self.agent_positions, axis=0)
        # update perimeter
        if self.update_perimeter:
            self.perimeter = np.sum(np.linalg.norm(np.roll(self.agent_positions, -1, axis=0) - self.agent_positions, axis=-1))
        # update rg
        if self.update_rg:
            self.center = np.mean(self.agent_positions, axis=0)
            self.rg = np.linalg.norm(np.linalg.norm(self.agent_positions - self.center, axis=-1), axis=0) / np.sqrt(self.N)
        # update r_mcc
        if self.update_r_mcc:
            if not hasattr(self, 'distance_matrix'):
                self.distance_matrix = np.linalg.norm(self.agent_positions[:, None, :] - self.agent_positions[None, :, :], axis=-1)
            self.r_mcc = self.distance_matrix.max() / 2

    def update_neighborhood(self, **kwargs):
        self.update_neighbor = kwargs.get('update_neighbor', True)
        self.neighbor_distance_cutoff = kwargs.get('neighbor_distance_cutoff', 2.5*self.r0)
        # udpate neighborhood matrix
        if self.update_neighbor:
            # consider lipids within the distance cutoff as neighbors
            distance_cutoff = self.neighbor_distance_cutoff
            # distance matrix
            self.distance_matrix = np.linalg.norm(self.agent_positions[:, None, :] - self.agent_positions[None, :, :], axis=-1)
            # get neighbors
            self.agent_neighborhood = self.distance_matrix <= distance_cutoff

    def update_forces(self, **kwargs):
        # forces
        relative_positions = self.agent_positions[:, None, :] - self.agent_positions[None, :, :]
        # only consider neighbors
        relative_positions = relative_positions * self.agent_neighborhood[:, :, None].astype(np.float32)
        # attractive force
        attr_force = attractive_force(r0=self.r0, k=self.k1, power=self.power, vec=relative_positions, distance=self.distance_matrix).sum(axis=0)
        # repulsive force
        repl_force = repulsive_force(r0=self.r0, k=self.k2, power=self.power, vec=relative_positions, distance=self.distance_matrix).sum(axis=0)
        # pull force
        pull_force_factor = kwargs.get('pull_force_factor', 1e-5)
        pull_f = pull_force(0, self.N, pull_force_factor)
        # friction force
        friction_force_factor = kwargs.get('friction_force_factor', 0)
        other_forces = attr_force + repl_force + pull_f
        fric_force = friction_force(self.velocities, other_forces, force=friction_force_factor)
        # total force
        total_forces = other_forces + fric_force
        self.forces = total_forces

        return total_forces