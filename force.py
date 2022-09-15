import numpy as np

# attractive force on a lipid with pos0 and r0
def attractive_force(pos=None, pos0=None, r0=1, k=1, power=2, epislon=1e-6, vec=None, distance=None):
    d0 = 2 * r0
    if vec is None:
        vec = pos - pos0
    # distance between two points
    if distance is None:
        distance = np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)
    else:
        distance = distance[:,:,np.newaxis]
    # magnitude of force
    magnitude =  np.abs(k * (distance - d0) ** power) * np.maximum(0, np.sign(distance - d0))
    # direction of force
    direction = vec / (distance + epislon)
    # force
    force = magnitude * direction
    return force

# repulsive force on a lipid with pos0 and r0
def repulsive_force(pos=None, pos0=None, r0=1, k=1, power=2, epislon=1e-6, vec=None, distance=None):
    d0 = 2 * r0
    if vec is None:
        vec = pos - pos0
    # distance between two points
    if distance is None:
        distance = np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)
    else:
        distance = distance[:,:,np.newaxis]
    # magnitude of force
    magnitude =  np.abs(k * (distance - d0) ** power) * np.maximum(0, np.sign(d0 - distance))
    # direction of force
    direction = - vec / (distance + epislon)
    # force
    force = magnitude * direction 
    return force

# van der Waals force
def vdW_force(pos=None, pos0=None, r0=1, k=1, epsilon=1e-6, vec=None, distance=None):
    d0 = 2 * r0
    if vec is None:
        vec = pos - pos0
    # distance between two points
    if distance is None:
        distance = np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)
    else:
        distance = distance[:,:,np.newaxis]
    # magnitude
    magnitude = -4*k*(-12*d0**12/(distance**13+epsilon)+6*d0**6/(distance**7+epsilon))
    # direction
    direction = vec / (distance+epsilon)
    # force
    force = magnitude * direction
    return force

# collision momentum transfer
# def collision_momentum_transfer(pos, pos0, r0, v, v0, e=0.5):
#     x, y = pos
#     x0, y0 = pos0
#     vx, vy = v
#     vx0, vy0 = v0
#     # distance between two points
#     distance = np.sqrt((x - x0)**2 + (y - y0)**2)
#     # vn **2 + v0n** 2 = (v**2 + v0**2) * (1 - e)
#     if distance <= r0:
#         magnitude = 0

# friction
def friction(velocity, other_force, force=20., threshold=1e-4, epsilon=1e-6):
    v_size = np.linalg.norm(velocity, ord=2, axis=-1)
    if v_size <= threshold:
        other_force_size = np.linalg.norm(other_force, ord=2, axis=-1)
        other_force_direction = other_force / (other_force_size + epsilon)
        return - np.minimum(other_force_size, force) * other_force_direction
    else:
        return - force * velocity / v_size

# friction
def friction_force(velocity, other_force, force=20., threshold=1e-4, epsilon=1e-6):
    v_size = np.linalg.norm(velocity, ord=2, axis=-1)
    force_vec = - force * velocity / (v_size.reshape(-1,1) + epsilon)
    # for those too slow
    other_force_size = np.linalg.norm(other_force, ord=2, axis=-1, keepdims=True)
    other_force_direction = other_force / (other_force_size + epsilon)
    static_force = - np.minimum(other_force_size, force) * other_force_direction
    force_vec[v_size <= threshold, :] = static_force[v_size <= threshold, :]
    return force_vec

# pull one lipid
def pull(uid, N, force=5.):
    if uid == 0:
        return np.array([force, 0.])
    else:
        # balance
        return np.array([-force/(N-1), 0.])

def pull_force(uid, N, force=5.):
    force_vec = np.tile([-force/(N-1), 0.], (N, 1))
    force_vec[uid, 0] = force
    return force_vec

# use three point to determine the center/radius of a circle
def get_circle(p1,p2,p3):
    x, y, z = p1[0]+p1[1]*1j, p2[0]+p2[1]*1j, p3[0]+p3[1]*1j
    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    center =  np.array([-c.real,-c.imag])
    radius = np.abs(c+x)
    return center, radius

# pressure
def pressure(p1, p2, p3, area, area0, center, k=1., epislon=1e-6):
    magnitude = k * (area - area0)
    # direction of force is from p2 to circle center
    circ_center, _ = get_circle(p1, p2, p3)
    direction = (circ_center - p2) * np.sign((circ_center - p2) * (center - p2))
    direction = direction / (np.linalg.norm(direction, ord=2) + epislon)
    # force
    force = magnitude * direction
    return force

# constant velocity
def zero_acceleration(uid, N):
    vec = np.ones((N, 2))
    vec[uid] = np.array([0., 0.])
    return vec

def constant_velocity(uid, N, velocity=1.):
    velocity_vec = np.zeros((N, 2))
    velocity_vec[uid, 0] = velocity
    return velocity_vec