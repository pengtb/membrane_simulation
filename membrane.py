import mesa
from scheduler import SimultaneousActivation, SimpleSimultaneousActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from lipids import Lipid_allinter, Lipid_simple
import numpy as np
from datacollector import JumpDataCollector, JumpModelDataCollector
from force import attractive_force, repulsive_force, friction_force, pull_force, constant_velocity, zero_acceleration
import jax.numpy as jnp
import jax
from jit_functions import *
import time

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

def initialize_actin_positions(actin_r0, num_lines=6, nuc_r0=25.8, cell_r0=59.68, match_end=False):
    # actin positions on one line
    sing_actin_pos_x = np.arange(nuc_r0, cell_r0, actin_r0*2)
    if match_end:
        sing_actin_pos_x = sing_actin_pos_x + cell_r0 - sing_actin_pos_x[-1]
    sing_actin_pos_y = np.zeros_like(sing_actin_pos_x)
    sing_actin_pos = np.stack([sing_actin_pos_x, sing_actin_pos_y], axis=1)
    # angle to rotate
    rotation_angle = 2 * np.pi / num_lines
    rotation_vec = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]]) # (2, 2)
    # rotate to get all actin positions
    rotated_actin_positions = [sing_actin_pos]
    for _ in range(1, num_lines):
        sing_actin_pos = np.matmul(sing_actin_pos, rotation_vec)
        rotated_actin_positions.append(sing_actin_pos)
    actin_positions = np.concatenate(rotated_actin_positions, axis=0)
    return actin_positions

# area of a polygon
def polygon_area(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# neighborhood matrix
def neighborhood_matrix(N, num_neighbor):
    agent_neighborhood = jnp.zeros((N, N), dtype=bool)
    for idx in range(1, num_neighbor+1):
        agent_neighborhood |= jnp.eye(N, k=idx, dtype=bool)
        agent_neighborhood |= jnp.eye(N, k=-idx, dtype=bool)
        agent_neighborhood |= jnp.eye(N, k=N-idx, dtype=bool)
        agent_neighborhood |= jnp.eye(N, k=idx-N, dtype=bool)
    return agent_neighborhood

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
        # seed
        if kwargs.get('prob', True):
            self.seed = kwargs.get('seed', 0)
            self.prng = jax.random.PRNGKey(self.seed)
        # initialize model constants
        self.N = N
        self.r0 = kwargs.get('r0', 1)
        # forces constants
        self.k1 = kwargs.get('k1', None)
        self.k2 = kwargs.get('k2', None)
        self.k = kwargs.get('k', None)
        self.angle_penalty = kwargs.get('angle_penalty', None)
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
            self.distance = kwargs.get('distance')
            positions = initialize_shape_positions(N, **kwargs)
        # place lipids
        self.agent_positions = jnp.asarray(positions, dtype=jnp.float64)
        self.init_pos = self.agent_positions

        # initialize neighborhood matrix
        self.init_neighborhood(self.N, **kwargs)
        
        # initialize velocities
        self.velocities = jnp.zeros((N, 2), dtype=jnp.float64)

        # initialize schedule
        self.schedule = SimpleSimultaneousActivation(self)

        # initialize output status
        self.init_status(**kwargs)

        # if model is simple
        self.simple = kwargs.get('simple', True)

        # model as a string
        self.string = kwargs.get('string', False)
        if self.string:
            self.string_neighborhood = neighborhood_matrix(N, 1)
            
        # initialize angles between edges
        edge_vectors = EdgeVectors(self.init_pos)
        self.init_angles = CalcIncludedAngle(edge_vectors)
   
    def init_neighborhood(self, N, **kwargs):
        self.all_neighbor = kwargs.get('all_neighbor', True)
        self.num_neighbor = kwargs.get('num_neighbor', 1)
        if not self.all_neighbor:
            self.agent_neighborhood = neighborhood_matrix(N, self.num_neighbor)
        else:
            self.agent_neighborhood = jnp.ones((N, N), dtype=bool) ^ jnp.eye(N, k=0, dtype=bool)
        self.distance_matrix = DistanceMatrix(self.agent_positions)
        
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
        # initialize other agents status
        self.velocities = jnp.zeros((self.N, 2), dtype=jnp.float64)
        self.forces = jnp.zeros((self.N, 2), dtype=jnp.float64)

        # model reporters
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
        self.datacollector.collect(self)
        
    def step(self, **kwargs):
        self.schedule.step(**kwargs)
        # run a step
        # update neighborhood
        self.update_neighborhood(**kwargs)
        # update forces
        total_forces = self.update_forces(**kwargs)
        # update accelerations
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
        # update velocities
        nextstep_velocities = self.velocities + acceleration * self.dt
        # speed limit
        vmin = kwargs.get('vmin', None)
        if vmin is not None:
            nextstep_velocities = ZeroLowestVelocity(nextstep_velocities, vmin)
        # update positions
        movement = (self.velocities + nextstep_velocities) / 2 * self.dt
        vlim = kwargs.get('vlim', None)
        if vlim is not None:
            movement_size = SizeVector(movement)
            if (movement_size > vlim * self.dt).any():
                maxmove_idx = jnp.argmax(movement_size, axis=0)
                clip_dt = ClipTimeStep(self.velocities[maxmove_idx], acceleration[maxmove_idx], movement[maxmove_idx], vlim*self.dt, self.dt, kwargs.get('min_dt', 1e-16))
                print('Movement too large, clip timestep to: ', clip_dt)
                nextstep_velocities = self.velocities + acceleration * clip_dt
                movement = (self.velocities + nextstep_velocities) / 2 * clip_dt
                self.clip_dt = clip_dt
            else:
                self.clip_dt = self.dt
        self.agent_positions = self.agent_positions + movement
        self.velocities = nextstep_velocities
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
        self.update_neighbor = kwargs.get('update_neighbor', False)
        self.neighbor_distance_cutoff = kwargs.get('neighbor_distance_cutoff', 1.25*0.375)
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

        # string neighbors
        if not self.string:
            string_neighbors = neighbors
        else:
            string_neighbors = self.string_neighborhood[:, :, None].astype(jnp.float64)
        
        # string force
        if self.k1 is not None:
            # attractive force
            attr_force = AttractiveForce(relative_positions, string_neighbors, distance_matrixs, k=self.k1, r0=self.r0, power=self.power).sum(axis=0)
            other_forces += attr_force
        if self.k2 is not None:
            # repulsive force
            repl_force = RepulsiveForce(relative_positions, string_neighbors, distance_matrixs, k=self.k2, r0=self.r0, power=self.power).sum(axis=0)
            other_forces += repl_force
        if self.k is not None:
            string_force = StringForce(relative_positions, string_neighbors, distance_matrixs, k=self.k, d0=self.r0*2).sum(axis=0)
            other_forces += string_force
        # vdW force
        if self.epsilon is not None:
            vdw_force = vdWForce(relative_positions, neighbors, distance_matrixs, k=self.epsilon, d0=self.r0*2).sum(axis=0)
            other_forces += vdw_force
        # pull force
        pull_force_factor = kwargs.get('pull_force_factor', 1e-5)
        if pull_force_factor is not None:
            pull_f = pull_force(0, self.N, pull_force_factor)
            if (not self.simple) | (self.schedule.steps == 1):
                self.pull_f = jnp.asarray(pull_f)
            other_forces += self.pull_f
        # angle penalty
        if self.angle_penalty is not None:
            edge_vectors = EdgeVectors(self.agent_positions)
            angles = CalcIncludedAngle(edge_vectors)
            angle_diffs = angles - self.init_angles
            # angle_penaltys = AnglePenalty(angle_diffs, edge_vectors, self.angle_penalty)
            angle_penaltys = AnglePenalty(self.agent_positions, angle_diffs, self.angle_penalty)
            other_forces += angle_penaltys
        # additional force
        additional_force = kwargs.get('additional_force', None)
        if additional_force is not None:
            other_forces += additional_force
        # friction force
        friction_force_factor = kwargs.get('friction_force_factor', None)
        if friction_force_factor is not None:
            fric_force = FrictionForce(self.velocities, other_forces, force=friction_force_factor)
            total_forces = other_forces + fric_force
        else:
            total_forces = other_forces        
        self.forces = total_forces

        return total_forces

    def membrane_growth(self, **kwargs):
        """add new lipids to the string system"""
        # distances between lipids on the string
        edge_vectors = EdgeVectors(self.agent_positions)
        string_distances = EdgeLengths(edge_vectors).squeeze()
        distance_threshold = kwargs.get('distance_threshold', 0.75)
        if distance_threshold is not None:
            if jnp.all(string_distances < distance_threshold):
                return None
        
        # number of new lipids
        max_added_perstep = kwargs.get('max_added_perstep', 2)
        sort_dists = jnp.sort(string_distances) 
        sort_dists_idxs = jnp.argsort(string_distances)
        if max_added_perstep is not None:
            max_dists = sort_dists[-max_added_perstep:]
            max_dists_idxs = sort_dists_idxs[-max_added_perstep:]
        else:
            max_dists = sort_dists
            max_dists_idxs = sort_dists_idxs
        if hasattr(self, 'prng'):
            power = kwargs.get('power', 2)
            probs = PlaceLipidProb(max_dists, distance_threshold, self.distance*2, power=power)
            add_pos_idxs = jnp.where(jax.random.bernoulli(self.prng, probs))[0]
            add_pos_idxs = max_dists_idxs[add_pos_idxs]
            self.prng = jax.random.split(self.prng, 1)[0]
        else:
            add_pos_idxs = max_dists_idxs[max_dists >= distance_threshold]
        
        # add new lipids
        num_add_lipids = len(add_pos_idxs)
        if num_add_lipids:
            self.N += num_add_lipids
            # add new positions
            if kwargs.get('no_overlap'):
                pos_vec = edge_vectors[add_pos_idxs]
                pos_veclen = string_distances[add_pos_idxs]
                new_pos_vectors = NooverlappNewLipidPos(pos_vec, pos_veclen, self.distance)
                if jnp.isnan(new_pos_vectors).any():
                    print('Cannot keep the distance between lipids, placing new lipids in the middle instead.')
                    new_pos_vectors = edge_vectors[add_pos_idxs]
            else:
                new_pos_vectors = edge_vectors[add_pos_idxs]
            
            add_lipids_pos = self.agent_positions[add_pos_idxs] + 0.5 * new_pos_vectors
            self.agent_positions = jnp.insert(self.agent_positions, add_pos_idxs+1, add_lipids_pos, axis=0)
            # add new velocities
            add_lipids_vel = self.velocities[add_pos_idxs] + 0.5 * EdgeVectors(self.velocities)[add_pos_idxs]
            self.velocities = jnp.insert(self.velocities, add_pos_idxs+1, add_lipids_vel, axis=0)
            # update string neighborhood
            self.string_neighborhood = neighborhood_matrix(self.N, 1)
            self.agent_neighborhood = neighborhood_matrix(self.N, self.num_neighbor)
            # add lipids
            lipid = kwargs.get('lipid', Lipid_simple)
            self.lipids = self.lipids + [lipid(idx, self) for idx in range(self.N-num_add_lipids, self.N)]
            self.mass = jnp.asarray([agent.mass for agent in self.lipids], dtype=jnp.float64)
            # update angles
            if self.angle_penalty is not None:
                self.init_angles = CalcIncludedAngle(EdgeVectors(self.agent_positions))
            return self.N
            
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
    
class Cytoskeleton(Membrane_jax):
    def __init__(self, num_lipids, num_actins=None, num_lines=6, **kwargs):
        super().__init__(num_lipids, lipid=Lipid_simple, **kwargs)
        # initialize constants
        self.actin_mass = kwargs.get('actin_mass', 1)
        self.actin_r0 = self.r0 if kwargs.get('actin_r0', None) is None else kwargs.get('actin_r0')
        self.actin_dist = self.actin_r0 * 2 if kwargs.get('actin_dist', None) is None else kwargs.get('actin_dist')
        
        self.nuc_r0 = kwargs.get('nuc_r0', np.power(0.08, 1/3) * 59.68)
        actin_pos = jnp.asarray(initialize_actin_positions(self.actin_r0, num_lines, self.nuc_r0, cell_r0=kwargs.get('cell_r0', 59.68)))
        other_actin_pos = actin_pos[len(actin_pos)//num_lines:, :]
        sing_actin_pos = jnp.asarray(initialize_actin_positions(self.actin_r0, 1, self.nuc_r0, cell_r0=kwargs.get('sing_cell_r0', 59), match_end=True))
        self.actin_pos = jnp.concatenate((sing_actin_pos, other_actin_pos), axis=0)
        
        self.num_actins = num_actins if num_actins is not None else len(self.actin_pos)
        self.num_singline_actins = self.num_actins // num_lines
        
        self.actin_vel = np.zeros((self.num_actins, 2), dtype=np.float64)
        self.actin_vel[:self.num_singline_actins,0] = kwargs.get('actin_vel', 1e-3)
        self.actin_vel = jnp.asarray(self.actin_vel)
        self.max_actin_vel = kwargs.get('max_actin_vel', None)
        # status report
        agent_reporter = {}
        agent_reporter['actin_pos_x'] = lambda a: a.actin_pos[:,0]
        agent_reporter['actin_pos_y'] = lambda a: a.actin_pos[:,1]
        self.actin_datacollector = JumpModelDataCollector(self, 
                                                          agent_reporters=agent_reporter, 
                                                          jump_step=self.jump_step)
        self.actin_datacollector.collect(self)
    
    def update_cytoskeleton_force(self, agent_positions, actin_pos, **kwargs):
        relative_pos = PairRelativePositions(agent_positions, actin_pos) # shape: (num_actins, num_lipids, 2)
        distance_mat = PairDistanceMatrix(agent_positions, actin_pos)[:, :, None] # shape: (num_actins, num_lipids, 1)
        neighborhood_mat = jnp.ones_like(distance_mat, dtype=jnp.float64) # shape: (num_actins, num_lipids, 1)
        add_force = vdWForce(relative_pos, neighborhood_mat, distance_mat, k=self.epsilon, d0=self.r0+self.actin_r0)
        add_force = add_force.sum(axis=0) # shape: (num_lipids, 2)
        if kwargs.get('cyto_string', False):
            d0 = self.r0+self.actin_r0
            string_neighborhood_mat = jnp.where(distance_mat < d0, 1, 0) # shape: (num_actins, num_lipids, 1)
            string_force = StringForce(relative_pos, string_neighborhood_mat, distance_mat, k=self.k, d0=d0).sum(axis=1) # shape: (num_lipids, 2)
            add_force = add_force + string_force
        return add_force
    
    def step(self, **kwargs):
        # update actin velocity
        actin_vel = self.actin_vel
        if self.max_actin_vel is not None:
            if self.actin_vel[0,0] < self.max_actin_vel:
                actin_vel_update = kwargs.get('actin_vel_update', None)
                if actin_vel_update is not None:
                    # vel_diff = self.actin_vel[0,0] - self.velocities[:,0].max()
                    # terminal_dist = self.agent_positions[:,0].max() - self.actin_pos[0,0]
                    # if terminal_dist > self.actin_vel[0,0] * self.dt:
                    # if vel_diff <= actin_vel_update:
                    if self.velocities[:,0].max() >= self.actin_vel[0,0]:
                        actin_vel = np.zeros_like(self.actin_vel, dtype=np.float64)
                        actin_vel[:self.num_singline_actins,0] = self.actin_vel[0,0] + actin_vel_update
                        actin_vel = jnp.asarray(actin_vel)
                        print('Warming up actin velocity to {}'.format(self.actin_vel[0,0]))
        # timestep
        if hasattr(self, 'clip_dt'):
            actin_dt = self.clip_dt
        else:
            actin_dt = self.dt
        # update actin positions
        actin_pos = self.actin_pos + actin_vel * actin_dt
        # add new actin if needed
        if actin_pos[0,0] - self.nuc_r0 > self.actin_dist:
            new_actin_pos = (actin_pos[0,0] - self.actin_dist, 0)
            actin_pos = jnp.concatenate((jnp.array(new_actin_pos, dtype=jnp.float64)[None, :], actin_pos), axis=0)
            actin_vel = jnp.concatenate((self.actin_vel[[0], :], self.actin_vel), axis=0)
            
        # forces on membrane
        prev_add_force = self.update_cytoskeleton_force(self.agent_positions, self.actin_pos, **kwargs)
        add_force = self.update_cytoskeleton_force(self.agent_positions, actin_pos, **kwargs)
        # force limit: actin not move when forces are too large
        force_lim = kwargs.get('force_lim', None)
        if force_lim is not None:
            maxforce = SizeVector(add_force).max()
            if maxforce <= force_lim:
                if len(actin_pos) > len(self.actin_pos):
                    self.num_actins += 1
                    self.num_singline_actins += 1
                # if (len(actin_pos) > len(self.actin_pos)) or (self.velocities[:,0].max() >= self.actin_vel[0,0]):
                self.actin_vel = actin_vel
                self.actin_pos = actin_pos
                # update membrane according to actin positions
                super().step(additional_force=add_force, **kwargs)
            else:
                print('Exceeding force limit: {}, waiting for lipids movement'.format(maxforce))
                # update membrane according to actin positions
                super().step(additional_force=prev_add_force, **kwargs)
        else:
            if len(actin_pos) > len(self.actin_pos):
                self.num_actins += 1
                self.num_singline_actins += 1
            self.actin_pos = actin_pos
            self.actin_vel = actin_vel
            # update membrane according to actin positions
            super().step(additional_force=add_force, **kwargs)
        # report pos of actins
        self.actin_datacollector.collect(self)