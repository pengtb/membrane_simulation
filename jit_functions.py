import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

@jit
def DistanceMatrix(positions):
    return jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

@jit
def RelativePositions(positions):
    related_positions = positions[:, None, :] - positions[None, :, :]
    return related_positions

@jit
def AttractiveForce(relative_positions, neighborhood, distances, k=1, power=1, r0=1):
    d0 = 2 * r0
    magnitude =  jnp.abs(k * (distances - d0) ** power) * jnp.maximum(0, jnp.sign(distances - d0))
    direction = jnp.nan_to_num(relative_positions / distances, copy=False)
    force = magnitude * direction
    return force * neighborhood

@jit
def RepulsiveForce(relative_positions, neighborhood, distances, k=1, power=1, r0=1):
    d0 = 2 * r0
    magnitude =  jnp.abs(k * (distances - d0) ** power) * jnp.maximum(0, jnp.sign(d0 - distances))
    direction = - jnp.nan_to_num(relative_positions / distances, copy=False)
    force = magnitude * direction
    return force * neighborhood

@jit
def StringForce(relative_positions, neighborhood, distances, k=1, r0=1):
    d0 = 2 * r0
    magnitude =  jnp.abs(k * (distances - d0))
    direction = jnp.nan_to_num(relative_positions / distances, copy=False)
    force = magnitude * direction
    return force * neighborhood

@jit 
def vdWForce(relative_positions, neighborhood, distances, k=1, r0=1):
    d0 = 2 * r0
    # magnitude = 4*k*(-12*d0**12/(distances **13 + episilon) + 6*d0**6/(distances **7+ episilon))
    magnitude = 4*k*(-12*d0**12/(distances **13) + 6*d0**6/(distances **7))
    # direction = relative_positions / (distances + episilon)
    direction = relative_positions / (distances)
    force = magnitude * direction
    return jnp.nan_to_num(force * neighborhood, copy=False)

@jit
def FrictionForce(velocity, other_force, force=20., threshold=1e-4):
    v_size = jnp.linalg.norm(velocity, ord=2, axis=-1)
    force_vec = - force * velocity / (v_size.reshape(-1,1))
    force_vec = jnp.nan_to_num(force_vec, copy=False)
    # for those too slow
    other_force_size = jnp.linalg.norm(other_force, ord=2, axis=-1, keepdims=True)
    other_force_direction = other_force / (other_force_size)
    other_force_direction = jnp.nan_to_num(other_force_direction, copy=False)
    static_force = - jnp.minimum(other_force_size, force) * other_force_direction
    force_vec = jnp.where((v_size <= threshold)[:,None], static_force, force_vec)
    return force_vec

@jit
def PolygonArea(positions):
    x, y = positions.T
    return 0.5*jnp.abs(jnp.dot(x,jnp.roll(y,1))-jnp.dot(y,jnp.roll(x,1)))

@jit
def Perimeter(positions):
    return jnp.sum(jnp.linalg.norm(jnp.roll(positions, -1, axis=0) - positions, axis=-1))

@jit
def RadiusofGyration(positions, center, N):
    return jnp.linalg.norm(jnp.linalg.norm(positions - center, axis=-1), axis=0) / jnp.sqrt(N)

@jit
def ZeroLowestVelocity(velocities, vmin):
    v_size = jnp.linalg.norm(velocities, axis=-1)
    return jnp.where(v_size[:, None] < vmin, jnp.zeros_like(velocities), velocities)

@jit
def EdgeVectors(positions):
    # calculate the vector of each edge
    edge_vectors = jnp.roll(positions, -1, axis=0) - positions
    return edge_vectors

@jit 
def EdgeLengths(edge_vectors):
    # calculate the length of each edge
    edge_lens = jnp.linalg.norm(edge_vectors, axis=-1, keepdims=True)
    return edge_lens

@jit
def CalcIncludedAngle(edge_vectors):
    """calculate the angle between each two edges in a polygon using the dot product"""
    # calculate the dot product of each two edges
    next_edges = jnp.roll(edge_vectors, -1, axis=0)
    dot_products = jnp.einsum('ij,ij->i', edge_vectors, next_edges)
    # calculate the cross product of each two edges
    cross_products = jnp.cross(edge_vectors, next_edges)

    # calculate the angle between each two edges
    angles = jnp.arctan2(cross_products, dot_products)
    return angles

@jit
def AnglePenalty(angle_diffs, edge_vectors, penalty_constant=1):
    """"use differences of angles to calculate the penalty on each node"""
    # angle penalty
    angle_penaltys = angle_diffs * penalty_constant
    # direction vertical to the edge
    oneside_directions = jnp.array([edge_vectors[:,1], -edge_vectors[:,0]]).T
    # normalize the direction
    oneside_directions = jnp.nan_to_num(oneside_directions / jnp.linalg.norm(oneside_directions, axis=-1, keepdims=True), copy=False)
    # one side penalty
    oneside_penaltys = jnp.einsum('ij,i->ij', oneside_directions, angle_penaltys)
    # another side penalty
    another_directions = jnp.roll(oneside_directions, -1, axis=0)
    another_penaltys = jnp.einsum('ij,i->ij', another_directions, angle_penaltys)
    another_penaltys = jnp.roll(another_penaltys, 2, axis=0)
    return oneside_penaltys + another_penaltys