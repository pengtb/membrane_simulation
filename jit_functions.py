import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

@jit
def DistanceMatrix(positions):
    return jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

@jit
def PairDistanceMatrix(positions1, positions2):
    return jnp.linalg.norm(positions2[:, None, :] - positions1[None, :, :], axis=-1)

@jit
def RelativePositions(positions):
    related_positions = positions[:, None, :] - positions[None, :, :]
    return related_positions

@jit
def PairRelativePositions(positions1, positions2):
    related_positions = positions2[:, None, :] - positions1[None, :, :]
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
def StringForce(relative_positions, neighborhood, distances, k=1, d0=1):
    magnitude =  jnp.abs(k * (distances - d0))
    direction = jnp.nan_to_num(relative_positions / distances, copy=False)
    force = magnitude * direction
    return force * neighborhood

@jit 
def vdWForce(relative_positions, neighborhood, distances, k=1, d0=1):
    # d0 = 2 * r0
    # magnitude = 4*k*(-12*d0**12/(distances **13 + episilon) + 6*d0**6/(distances **7+ episilon))
    # magnitude = 4*k*(-12*d0**12/(distances **13) + 6*d0**6/(distances **7))
    magnitude = 4*k*(-12*d0**12/(distances **14) + 6*d0**6/(distances **8))
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

# @jit
# def AnglePenalty(angle_diffs, edge_vectors, penalty_constant=1):
#     """"use differences of angles to calculate the penalty on each node"""
#     # angle penalty
#     angle_penaltys = angle_diffs * penalty_constant
#     # direction vertical to the edge
#     oneside_directions = jnp.array([edge_vectors[:,1], -edge_vectors[:,0]]).T
#     # normalize the direction
#     oneside_directions = jnp.nan_to_num(oneside_directions / jnp.linalg.norm(oneside_directions, axis=-1, keepdims=True), copy=False)
#     # one side penalty
#     oneside_penaltys = jnp.einsum('ij,i->ij', oneside_directions, angle_penaltys)
#     # another side penalty
#     another_directions = jnp.roll(oneside_directions, -1, axis=0)
#     another_penaltys = jnp.einsum('ij,i->ij', another_directions, angle_penaltys)
#     another_penaltys = jnp.roll(another_penaltys, 2, axis=0)
#     return oneside_penaltys + another_penaltys

@jit
def AnglePenalty(positions, angle_diffs, penalty_constant=1):
    """
    use differences of angles to calculate the penalty on each node
    positions: (N, 2)
    init_angles: (N,)
    """
    # coordinate of each node
    x1, y1 = positions.T
    x2, y2 = jnp.roll(positions, -1, axis=0).T
    x3, y3 = jnp.roll(positions, -2, axis=0).T
    k = penalty_constant
    # vector of each edge
    v1x = x2 - x1
    v1y = y2 - y1
    v2x = x3 - x2
    v2y = y3 - y2
    # force on (x1, y1)
    fx1 = k * (-v1y) * angle_diffs / (v1x ** 2 + v1y ** 2)
    fy1 = k * (v1x) * angle_diffs / (v1x ** 2 + v1y ** 2)
    # force on (x3, y3)
    fx3 = k * (-v2y) * angle_diffs / (v2x ** 2 + v2y ** 2)
    fy3 = k * (v2x) * angle_diffs / (v2x ** 2 + v2y ** 2)
    # force on (x2, y2)
    fx2 = -k * angle_diffs * (x1**2*y2 - x1**2*y3 - 
    2*x1*x2*y2 + 2*x1*x2*y3 + 
    x2**2*y1 - x2**2*y3 - 
    2*x2*x3*y1 + 2*x2*x3*y2 + 
    x3**2*y1 - x3**2*y2 + 
    y1**2*y2 - y1**2*y3 - 
    y1*y2**2 + y1*y3**2 + 
    y2**2*y3 - y2*y3**2) / (v1x ** 2 + v1y ** 2) / (v2x ** 2 + v2y ** 2)
    fy2 = k * angle_diffs * (x1**2*x2 - x1**2*x3 - 
    x1*x2**2 + x1*x3**2 +
    x1*y2**2 - 2*x1*y2*y3 + 
    x1*y3**2 + x2**2*x3 - 
    x2*x3**2 + x2*y1**2 - 
    2*x2*y1*y2 + 2*x2*y2*y3 -
    x2*y3**2 - x3*y1**2 +
    2*x3*y1*y2 - x3*y2**2) / (v1x ** 2 + v1y ** 2) / (v2x ** 2 + v2y ** 2)
    # total force
    f1 = jnp.array([fx1, fy1]).T
    f2 = jnp.array([fx2, fy2]).T
    f3 = jnp.array([fx3, fy3]).T
    f2 = jnp.roll(f2, 1, axis=0)
    f3 = jnp.roll(f3, 2, axis=0)
    f = f1 + f2 + f3
    f = -f # because we want to minimize the energy
    return f

@jit
def PlaceLipidProb(dist, lower=0.375, threshold=0.75, power=3):
    """
    the probability of adding new lipids to the center of two lipids of distance dist
    dist: (N,)
    """
    dist = jnp.maximum(dist - lower, 0)
    dist = jnp.minimum(dist, threshold)
    prob = dist / (threshold - lower)
    return prob**power

@jit
def NooverlappNewLipidPos(edge_vecs, edge_lens, d=0.375):
    """
    the direction vector of placing new lipid if no overlap is allowed
    edge_vecs: (N, 2)
    edge_lens: (N,)
    """
    # rotate angle
    cos_angle = edge_lens / 2 / d # (N,)
    sin_angle = jnp.sqrt(1 - cos_angle**2) # (N,)
    # rotate direction vector by angle
    rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]).transpose(2,0,1) # (N, 2, 2)
    # rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]).transpose(2,1,0) # (N, 2, 2)
    new_edge_vecs = jnp.einsum('ijk,ik->ij', rotation_matrix, edge_vecs) # (N, 2)
    # adjust the length to d
    new_edge_vecs = new_edge_vecs / cos_angle
    return new_edge_vecs

@jit
def ClipVelocity(velocities, max_vel=0.05):
    """
    clip the velocity to a maximum value
    """
    norm = jnp.linalg.norm(velocities, axis=-1, keepdims=True)
    return velocities / (norm + 1e-10) * jnp.minimum(norm, max_vel)

@jit
def ClipMovement(movement, max_move=0.05):
    """
    clip the movement to a maximum value
    """
    norm = jnp.linalg.norm(movement, axis=-1, keepdims=True)
    return movement / (norm + 1e-10) * jnp.minimum(norm, max_move)

@jit
def ClipTimeStep(vel, acc, mov, max_move=1e-4, prev_dt=0.001, min_dt=1e-6):
    """
    Lower timestep to prevent movement on x/y larger than max_move
    velocities: (2)
    acceleration: (2)
    vt + 0.5 * at^2 = max_move
    """
    vx, vy = vel
    ax, ay = acc
    mx, my = mov
    vx_t = vx + ax * prev_dt
    vy_t = vy + ay * prev_dt
    # max velocity of x/y to get min dt
    vx_max = jnp.maximum(jnp.abs(vx), jnp.abs(vx_t))
    vy_max = jnp.maximum(jnp.abs(vy), jnp.abs(vy_t))
    vmax = jnp.maximum(vx_max, vy_max)
    clip_dt = max_move / vmax
    return jnp.maximum(clip_dt, min_dt)

def ClipTimeStepSimple(vel, acc, mov, max_move=1e-4):
    """
    Lower timestep to prevent movement on x/y larger than max_move
    velocities: (2)
    acceleration: (2)
    vt + 0.5 * at^2 = max_move
    """
    vx, vy = vel
    ax, ay = acc
    mx, my = mov
    tx = jnp.sqrt(2*max_move / (jnp.abs(ax) + 1e-10))
    ty = jnp.sqrt(2*max_move / (jnp.abs(ay) + 1e-10))
    return jnp.minimum(tx, ty)

@jit
def SizeVector(vectors):
    """
    vectors: (num_lipids, 2)
    sizes: (num_lipids,)
    """
    return jnp.linalg.norm(vectors, axis=-1)
    