import mesa
import numpy as np
from force import attractive_force, repulsive_force, friction, pull, pressure

# lipid with interaction with all other lipids
class Lipid_allinter(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)
        self.mass = 1.
        # initial velocity
        self.velocity = np.array([0., 0.])

    def step(self, aristotelian=False, **kwargs):
        # get all other lipids
        lipids = self.model.schedule.agents
        # get all other lipids' positions
        positions = [lipid.pos for lipid in lipids if lipid != self]
        # get all other lipids' forces
        attr_forces = np.array([attractive_force(pos, self.pos, self.model.r0, self.model.k1) for pos in positions])
        rpls_forces = np.array([repulsive_force(pos, self.pos, self.model.r0, self.model.k2) for pos in positions])
        forces = np.sum(attr_forces + rpls_forces, axis=0)
        friction_force_factor = kwargs.get('friction_force_factor', 5.)
        fric_force = friction(self.velocity, forces, friction_force_factor)
        # get total force
        self.total_force = forces + fric_force
        # get acceleration
        acceleration = self.total_force / self.mass
        if not aristotelian:
            # get movement
            self.movement = (self.velocity + acceleration * self.model.dt / 2) * self.model.dt
            # get delta velocity
            self.delta_velocity = acceleration * self.model.dt
        else:
            # self.velocity = acceleration
            self.velocity = acceleration / 5
            # self.velocity = np.sign(acceleration) * np.log(np.abs(acceleration) + 1)
            self.movement = self.velocity * self.model.dt

    def advance(self, aristotelian=False, **kwargs) -> None:
        # update position
        self.pos += self.movement
        if not aristotelian:
            # update velocity
            self.velocity += self.delta_velocity

    def stop(self):
        # reset velocity to zero
        self.velocity = np.array([0., 0.])

# lipid that only interacts with neighbors
class Lipid_neighbor(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)
        self.mass = 1.
        # initial velocity
        self.velocity = np.array([0., 0.])
        # initial neighbor indexes
        if hasattr(self.model, 'neighborhood'):
            self.neighbor_indexes = np.arange(self.model.N, dtype=np.int32)[self.model.neighborhood[self.unique_id]]
        else:
            neighbor_indexes = np.array([self.unique_id-1, self.unique_id+1])
            # make sure neighbor indexes are in range
            self.neighbor_indexes = np.mod(neighbor_indexes, self.model.N)

    def push_out(self, velocity_factor=1., epision=1e-6):
        # get positions of all lipids
        positions = np.stack([lipid.pos for lipid in self.model.schedule.agents], axis=0)
        # get center of all lipids
        center = np.mean(positions, axis=0)
        # get direction of velocity
        direction = self.pos - center
        direction = direction / np.linalg.norm(direction + epision)
        # get velocity
        self.velocity = direction * velocity_factor

    def step(self, **kwargs):
        # get neighbor lipids
        neighbor_lipids = [self.model.schedule.agents[index] for index in self.neighbor_indexes]

        # get neighbor lipids' positions
        positions = [lipid.pos for lipid in neighbor_lipids]
        # get neighbor lipids' forces
        attr_forces = np.array([attractive_force(pos, self.pos, self.model.r0, self.model.k1, self.model.power) for pos in positions])
        rpls_forces = np.array([repulsive_force(pos, self.pos, self.model.r0, self.model.k2, self.model.power) for pos in positions])
        forces = np.sum(attr_forces + rpls_forces, axis=0)
        # pull
        pull_force_factor = kwargs.get('pull_force_factor', None)
        if pull_force_factor is not None:
            pull_force = pull(self.unique_id, self.model.N, pull_force_factor)
            forces += pull_force
        # get friction force
        friction_force_factor = kwargs.get('friction_force_factor', 1e-1)
        fric_force = friction(self.velocity, forces, friction_force_factor)
        # get total force
        self.total_force = forces + fric_force

        # get acceleration
        acceleration = self.total_force / self.mass
        aristotelian = kwargs.get('aristotelian', False)
        if not aristotelian:
            # get movement
            self.movement = (self.velocity + acceleration * self.model.dt / 2) * self.model.dt
            # get delta velocity
            self.delta_velocity = acceleration * self.model.dt
        else:
            # self.velocity = acceleration
            self.velocity = acceleration / 5
            # self.velocity = np.sign(acceleration) * np.log(np.abs(acceleration) + 1)
            self.movement = self.velocity * self.model.dt

    def advance(self, update_neighbor=False, **kwargs) -> None:
        # update position
        self.pos += self.movement
        aristotelian = kwargs.get('aristotelian', False)
        if not aristotelian:
            # update velocity
            self.velocity += self.delta_velocity

        # update neighbor indexes
        if update_neighbor:
            self.neighbor_indexes = np.arange(self.model.N, dtype=np.int32)[self.model.neighborhood[self.unique_id]]

    def stop(self):
        # reset velocity to zero
        self.velocity = np.array([0., 0.])

# lipid that is only affected by its area
class Lipid_area(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)
        self.mass = 1.
        # initial velocity
        self.velocity = np.array([0., 0.])

    # push the lipid away from the center
    def push_out(self, velocity_factor=1., epision=1e-6):
        # get positions of all lipids
        positions = np.stack([lipid.pos for lipid in self.model.schedule.agents], axis=0)
        # get center of all lipids
        center = np.mean(positions, axis=0)
        # get direction of velocity
        direction = self.pos - center
        direction = direction / np.linalg.norm(direction + epision)
        # get velocity
        self.velocity = direction * velocity_factor

    def step(self, aristotelian=False, **kwargs):
        # get number of lipids
        n_lipids = self.model.N
        # get neighbor indexes
        neighbor_indexes = np.array([self.unique_id-1, self.unique_id+1])
        # make sure neighbor indexes are in range
        neighbor_indexes = np.mod(neighbor_indexes, n_lipids)
        # get neighbor lipids
        neighbor_lipids = [self.model.schedule.agents[index] for index in neighbor_indexes]

        # get neighbor lipids' & own positions
        positions = np.stack([lipid.pos for lipid in [neighbor_lipids[0], self, neighbor_lipids[1]]], axis=0)
        # get presssure
        forces = pressure(positions[0], positions[1], positions[2], self.model.area, self.model.area0, self.model.center, self.model.k3)
        # pull
        pull_force_factor = kwargs.get('pull_force_factor', None)
        if pull_force_factor is not None:
            pull_force = pull(self.unique_id, self.model.N, pull_force_factor)
            forces += pull_force
        # get friction force
        friction_force_factor = kwargs.get('friction_force_factor', 1e-1)
        fric_force = friction(self.velocity, forces, friction_force_factor)
        # get total force
        self.total_force = forces + fric_force

        # get acceleration
        acceleration = self.total_force / self.mass
        if not aristotelian:
            # get movement
            self.movement = (self.velocity + acceleration * self.model.dt / 2) * self.model.dt
            # get delta velocity
            self.delta_velocity = acceleration * self.model.dt
        else:
            self.velocity = acceleration
            # self.velocity = acceleration / 5
            # self.velocity = np.sign(acceleration) * np.log(np.abs(acceleration) + 1)
            self.movement = self.velocity * self.model.dt

    def advance(self, aristotelian=False, **kwargs) -> None:
        # update position
        self.pos += self.movement
        if not aristotelian:
            # update velocity
            self.velocity += self.delta_velocity

    def stop(self):
        # reset velocity to zero
        self.velocity = np.array([0., 0.])

# lipid that is affected by its area & neighboring lipids
class Lipid_AN(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)
        self.mass = 1.
        # initial velocity
        self.velocity = np.array([0., 0.])

    # push the lipid away from the center
    def push_out(self, velocity_factor=1., epision=1e-6):
        # get positions of all lipids
        positions = np.stack([lipid.pos for lipid in self.model.schedule.agents], axis=0)
        # get center of all lipids
        center = np.mean(positions, axis=0)
        # get direction of velocity
        direction = self.pos - center
        direction = direction / np.linalg.norm(direction + epision)
        # get velocity
        self.velocity = direction * velocity_factor

    def step(self, aristotelian=False, **kwargs):
        # get number of lipids
        n_lipids = self.model.N
        # get neighbor indexes
        neighbor_indexes = np.array([self.unique_id-1, self.unique_id+1])
        # make sure neighbor indexes are in range
        neighbor_indexes = np.mod(neighbor_indexes, n_lipids)
        # get neighbor lipids
        neighbor_lipids = [self.model.schedule.agents[index] for index in neighbor_indexes]
        # get neighbor poistions
        neighbor_positions = np.stack([lipid.pos for lipid in neighbor_lipids], axis=0)
        # get neighbor lipids' forces
        attr_forces = np.array([attractive_force(pos, self.pos, self.model.r0, self.model.k1, self.model.power) for pos in neighbor_positions])
        rpls_forces = np.array([repulsive_force(pos, self.pos, self.model.r0, self.model.k2, self.model.power) for pos in neighbor_positions])
        forces = np.sum(attr_forces + rpls_forces, axis=0)
        # get neighbor lipids' & own positions
        positions = np.stack([lipid.pos for lipid in [neighbor_lipids[0], self, neighbor_lipids[1]]], axis=0)
        # get presssure
        forces += pressure(positions[0], positions[1], positions[2], self.model.area, self.model.area0, self.model.center, self.model.k3)
        # pull
        pull_force_factor = kwargs.get('pull_force_factor', None)
        if pull_force_factor is not None:
            pull_force = pull(self.unique_id, self.model.N, pull_force_factor)
            forces += pull_force
        # get friction force
        friction_force_factor = kwargs.get('friction_force_factor', 1e-1)
        fric_force = friction(self.velocity, forces, friction_force_factor)
        # get total force
        self.total_force = forces + fric_force

        # get acceleration
        acceleration = self.total_force / self.mass
        if not aristotelian:
            # get movement
            self.movement = (self.velocity + acceleration * self.model.dt / 2) * self.model.dt
            # get delta velocity
            self.delta_velocity = acceleration * self.model.dt
        else:
            self.velocity = acceleration
            # self.velocity = acceleration / 5
            # self.velocity = np.sign(acceleration) * np.log(np.abs(acceleration) + 1)
            self.movement = self.velocity * self.model.dt

    def advance(self, aristotelian=False, **kwargs) -> None:
        # update position
        self.pos += self.movement
        if not aristotelian:
            # update velocity
            self.velocity += self.delta_velocity

    def stop(self):
        # reset velocity to zero
        self.velocity = np.array([0., 0.])

# super simple lipids
class Lipid_simple(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)
        self.mass = 1.