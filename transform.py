# pylint: disable=invalid-name, bad-whitespace, too-many-arguments
"""
Basic graphics related geometry tools to complement numpy
Quaternion, graphics 4x4 matrices, and vector utilities.
@author: franco
"""
# Python built-in modules
import math                 # mainly for trigonometry functions
from numbers import Number  # useful to check type of arg: scalar or vector?

# external module
import numpy as np          # matrices, vectors & quaternions are numpy arrays


TRAN_ADJ_MATRIX = np.array([ 1.0, -1.0, -1.0])
ROT_ADJ_MATRIX = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

# Matrix transform functions between conventional camera coordinate frame and
# OpenGL coordinate frame
def mat_to_gl(Rt):
    M = np.eye(4)
    M[:3,:3] = np.dot(ROT_ADJ_MATRIX,Rt[:3,:3])
    M[:3,3] = Rt[:3,3] * TRAN_ADJ_MATRIX
    return M

def mat_from_gl(Rt):
    M = np.eye(4)
    M[:3,:3] = np.dot(np.linalg.inv(ROT_ADJ_MATRIX),Rt[:3,:3])
    M[:3,3] = Rt[:3,3] * TRAN_ADJ_MATRIX
    return M


# Some useful functions on vectors -------------------------------------------
def vec(*iterable):
    """ shortcut to make numpy vector of any iterable(tuple...) or vector """
    return np.asarray(iterable if len(iterable) > 1 else iterable[0], 'f')


def normalized(vector):
    """ normalized version of any vector, with zero division check """
    norm = math.sqrt(sum(vector*vector))
    return vector / norm if norm > 0. else vector


def lerp(point_a, point_b, fraction):
    """ linear interpolation between two quantities with linear operators """
    return point_a + fraction * (point_b - point_a)


# Typical 4x4 matrix utilities for OpenGL ------------------------------------
def identity():
    """ 4x4 identity matrix """
    return np.identity(4, 'f')


def ortho(left, right, bot, top, near, far):
    """ orthogonal projection matrix for OpenGL """
    dx, dy, dz = right - left, top - bot, far - near
    rx, ry, rz = -(right+left) / dx, -(top+bot) / dy, -(far+near) / dz
    return np.array([[2/dx, 0,    0,     rx],
                     [0,    2/dy, 0,     ry],
                     [0,    0,    -2/dz, rz],
                     [0,    0,    0,     1]], 'f')


def perspective(fovy, aspect, near, far, cx=0, cy=0):
    """ perspective projection matrix, from field of view and aspect ratio """
    _scale = 1.0/math.tan(math.radians(fovy)/2.0)
    sx, sy = _scale / aspect, _scale
    zz = (far + near) / (near - far)
    zw = 2 * far * near/(near - far)
    return np.array([[sx, 0,  cx,  0],
                     [0,  sy, cy,  0],
                     [0,  0, zz, zw],
                     [0,  0, -1,  0]], 'f')


def frustum(xmin, xmax, ymin, ymax, zmin, zmax):
    """ frustum projection matrix for OpenGL, from min and max coordinates"""
    a = (xmax+xmin) / (xmax-xmin)
    b = (ymax+ymin) / (ymax-ymin)
    c = -(zmax+zmin) / (zmax-zmin)
    d = -2*zmax*zmin / (zmax-zmin)
    sx = 2*zmin / (xmax-xmin)
    sy = 2*zmin / (ymax-ymin)
    return np.array([[sx, 0,  a, 0],
                     [0, sy,  b, 0],
                     [0,  0,  c, d],
                     [0,  0, -1, 0]], 'f')


def translate(x=0.0, y=0.0, z=0.0):
    """ matrix to translate from coordinates (x,y,z) or a vector x"""
    matrix = np.identity(4, 'f')
    matrix[:3, 3] = vec(x, y, z) if isinstance(x, Number) else vec(x)
    return matrix


def scale(x, y=None, z=None):
    """scale matrix, with uniform (x alone) or per-dimension (x,y,z) factors"""
    x, y, z = (x, y, z) if isinstance(x, Number) else (x[0], x[1], x[2])
    y, z = (x, x) if y is None or z is None else (y, z)  # uniform scaling
    return np.diag((x, y, z, 1))


def sincos(degrees=0.0, radians=None):
    """ Rotation utility shortcut to compute sine and cosine of an angle. """
    radians = radians if radians else math.radians(degrees)
    return math.sin(radians), math.cos(radians)


def rotate(axis=(1., 0., 0.), angle=0.0, radians=None):
    """ 4x4 rotation matrix around 'axis' with 'angle' degrees or 'radians' """
    x, y, z = normalized(vec(axis))
    s, c = sincos(angle, radians)
    nc = 1 - c
    return np.array([[x*x*nc + c,   x*y*nc - z*s, x*z*nc + y*s, 0],
                     [y*x*nc + z*s, y*y*nc + c,   y*z*nc - x*s, 0],
                     [x*z*nc - y*s, y*z*nc + x*s, z*z*nc + c,   0],
                     [0,            0,            0,            1]], 'f')


def lookat(eye, target, up):
    """ Computes 4x4 view matrix from 3d point 'eye' to 'target',
        'up' 3d vector fixes orientation """
    view = normalized(vec(target)[:3] - vec(eye)[:3])
    up = normalized(vec(up)[:3])
    right = np.cross(view, up)
    up = np.cross(right, view)
    rotation = np.identity(4)
    rotation[:3, :3] = np.vstack([right, up, -view])
    return np.dot(rotation, translate(-eye))


# quaternion functions -------------------------------------------------------
def quaternion(x=vec(0., 0., 0.), y=0.0, z=0.0, w=1.0):
    """ Init quaternion, w=real and, x,y,z or vector x imaginary components """
    x, y, z = (x, y, z) if isinstance(x, Number) else (x[0], x[1], x[2])
    return np.array((w, x, y, z), 'f')


def quaternion_from_axis_angle(axis, degrees=0.0, radians=None):
    """ Compute quaternion from an axis vec and angle around this axis """
    sin, cos = sincos(radians=radians*0.5) if radians else sincos(degrees*0.5)
    return quaternion(normalized(vec(axis))*sin, w=cos)


def quaternion_from_euler(yaw=0.0, pitch=0.0, roll=0.0, radians=None):
    """ Compute quaternion from three euler angles in degrees or radians """
    siy, coy = sincos(yaw * 0.5, radians[0] * 0.5 if radians else None)
    sir, cor = sincos(roll * 0.5, radians[1] * 0.5 if radians else None)
    sip, cop = sincos(pitch * 0.5, radians[2] * 0.5 if radians else None)
    return quaternion(x=coy*sir*cop - siy*cor*sip, y=coy*cor*sip + siy*sir*cop,
                      z=siy*cor*cop - coy*sir*sip, w=coy*cor*cop + siy*sir*sip)


def quaternion_mul(q1, q2):
    """ Compute quaternion which composes rotations of two quaternions """
    return np.dot(np.array([[q1[0], -q1[1], -q1[2], -q1[3]],
                            [q1[1],  q1[0], -q1[3],  q1[2]],
                            [q1[2],  q1[3],  q1[0], -q1[1]],
                            [q1[3], -q1[2],  q1[1],  q1[0]]]), q2)


def quaternion_matrix(q):
    """ Create 4x4 rotation matrix from quaternion q """
    q = normalized(q)  # only unit quaternions are valid rotations.
    nxx, nyy, nzz = -q[1]*q[1], -q[2]*q[2], -q[3]*q[3]
    qwx, qwy, qwz = q[0]*q[1], q[0]*q[2], q[0]*q[3]
    qxy, qxz, qyz = q[1]*q[2], q[1]*q[3], q[2]*q[3]
    return np.array([[2*(nyy + nzz)+1, 2*(qxy - qwz),   2*(qxz + qwy),   0],
                     [2 * (qxy + qwz), 2 * (nxx + nzz) + 1, 2 * (qyz - qwx), 0],
                     [2 * (qxz - qwy), 2 * (qyz + qwx), 2 * (nxx + nyy) + 1, 0],
                     [0, 0, 0, 1]], 'f')


def quaternion_slerp(q0, q1, fraction):
    """ Spherical interpolation of two quaternions by 'fraction' """
    # only unit quaternions are valid rotations.
    q0, q1 = normalized(q0), normalized(q1)
    dot = np.dot(q0, q1)

    # if negative dot product, the quaternions have opposite handedness
    # and slerp won't take the shorter path. Fix by reversing one quaternion.
    q1, dot = (q1, dot) if dot > 0 else (-q1, -dot)

    theta_0 = math.acos(np.clip(dot, -1, 1))  # angle between input vectors
    theta = theta_0 * fraction                # angle between q0 and result
    q2 = normalized(q1 - q0*dot)              # {q0, q2} now orthonormal basis

    return q0*math.cos(theta) + q2*math.sin(theta)


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


# a trackball class based on provided quaternion functions -------------------
class Trackball(object):
    """Virtual trackball for 3D scene viewing. Independent of window system."""

    def __init__(self, yaw=0., roll=0., pitch=0., distance=1., radians=None):
        """ Build a new trackball with specified view, angles in degrees """
        self.rotation = quaternion_from_euler(yaw, roll, pitch, radians)
        self.distance = max(distance, 0.001)
        self.pos2d = vec(0.0, 0.0)
        self.view_matrix = np.eye(4)
        self.model_matrix = np.eye(4)

    def drag(self, old, new, winsize):
        """ Move trackball from old to new 2d normalized window position """
        old, new = ((2*vec(pos) - winsize) / winsize for pos in (old, new))
        self.rotation = quaternion_mul(self._rotate(old, new), self.rotation)
        self._update()

    def zoom(self, delta, size):
        """ Zoom trackball by a factor delta normalized by window size """
        self.distance = max(0.001, self.distance * (1 - 50*delta/size))
        self._update()

    def pan(self, old, new):
        """ Pan in camera's reference by a 2d vector factor of (new - old) """
        self.pos2d += (vec(new) - old) * 0.001 * self.distance
        self._update()

    def pose_matrix(self):
        """ View matrix transformation, including distance to target point """
        return np.dot(translate(*self.pos2d, z=-self.distance), self.rotate())

    def projection_matrix(self, winsize):
        """ Projection matrix with z-clipping range adaptive to distance """
        z_range = vec(0.1, 100) * self.distance  # proportion to dist
        return perspective(35, winsize[0] / winsize[1], *z_range)

    def rotate(self):
        """ Rotational component of trackball position """
        return quaternion_matrix(self.rotation)

    def set_view_matrix(self, Rt):
        self.view_matrix = Rt
        self.update_pose_matrix()

    def set_model_matrix(self, Rt):
        self.model_matrix = Rt
        self.update_pose_matrix()

    def update_pose_matrix(self):
        M = np.dot(self.view_matrix, self.model_matrix)
        self.set_pose_matrix(M)

    def set_pose_matrix(self, M):
        self.pos2d = vec(M[0,3], M[1,3])
        self.distance = -M[2,3]
        self.rotation = quaternion_from_matrix(M[:3,:3])
        self._update()

    def view_from_pose_vec(self, V, invert=False):
        V = np.array(V)
        T = translate(V[0], V[1], V[2])
        R = quaternion_matrix(V[3:])
        if invert:
            self.view_from_pose_matrix(np.linalg.inv(np.dot(T,R)))
        else:
            self.view_from_pose_matrix(np.dot(T,R))

    def view_from_pose_matrix(self, Rt):
        """  Set the model view matrix from object pose. """
        # setup 4*4 model view matrix
        # mat_to_gl(Rt)
        # self.pos2d = vec(M[0,3], M[1,3])
        # self.distance = -M[2,3]
        # self.rotation = quaternion_from_matrix(M[:3,:3])
        self.set_view_matrix(mat_to_gl(Rt))

    # def set_pose_matrix(self, Rt):
    #     self.model_matrix = np.dot(np.linalg.inv(self.view_matrix), Rt)

    def _update(self):
        self.model_matrix = np.dot(np.linalg.inv(self.view_matrix), self.pose_matrix())

    def _project3d(self, position2d, radius=0.8):
        """ Project x,y on sphere OR hyperbolic sheet if away from center """
        p2, r2 = sum(position2d*position2d), radius*radius
        zcoord = math.sqrt(r2 - p2) if 2*p2 < r2 else r2 / (2*math.sqrt(p2))
        return vec(position2d[0], position2d[1], zcoord)

    def _rotate(self, old, new):
        """ Rotation of axis orthogonal to old & new's 3D ball projections """
        old, new = (normalized(self._project3d(pos)) for pos in (old, new))
        phi = 2 * math.acos(np.clip(np.dot(old, new), -1, 1))
        return quaternion_from_axis_angle(np.cross(old, new), radians=phi)
