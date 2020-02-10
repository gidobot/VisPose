from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle

### Helper Functions ###

def my_calibration(sz):
  row,col = sz
  fx = 2555*col/2592
  fy = 2586*row/1936
  K = np.diag([fx,fy,1])
  K[0,2] = 0.5*col
  K[1,2] = 0.5*row
  return K


def rotation_matrix(a):
    """  Creates a 3D rotation matrix for rotation
      around the axis of the vector a. """
    R = np.eye(4)
    R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R

### Camera Class ###

class Camera(object):
  """ Class for representing pin-hole cameras. """

  def __init__(self, calib_file, cam_name='cam0'):
    """ Initialize P = K[R|t] camera model. """
    self.P = None
    self.K = None  # calibration matrix
    self.c = None  # camera center
    self.Rt = None # extrinsics
    self.width = None  # image width
    self.height = None # image height
    self.cam_name = cam_name
    self.load_cal(calib_file, self.cam_name)


  def project(self,X):
    """  Project points in X (4*n array) and normalize coordinates. """
    x = np.dot(self.P,X)
    x = x / x[2]
    return x


  # def factor(self):
  #   """  Factorize the camera matrix into K,R,t as P = K[R|t]. """

  #   # factor first 3*3 part
  #   K,R = linalg.rq(self.P[:,:3])

  #   # make diagonal of K positive
  #   T = np.diag(np.sign(np.diag(K)))
  #   if linalg.det(T) < 0:
  #     T[1,1] *= -1

  #   self.K = np.dot(K,T)
  #   self.R = np.dot(T,R) # T is its own inverse
  #   self.t = np.dot(linalg.inv(self.K),self.P[:,3])

  #   return self.K, self.R, self.t


  def save_cal(self):
    with open('ar_camera.pkl','w') as f:
      pickle.dump(self.K,f)
      pickle.dump(np.dot(linalg.inv(self.K),self.P),f)


  def load_cal(self, filename, cam_name):
    """ Load camera calibration from yaml. """
    self.cam_name = cam_name
    # TODO: add distortion support
    f = file(filename, 'r')
    calib_data = yaml.load(f)
    calib_data = calib_data[cam_name]
    # currently only support pinhole cameras
    # import pdb; pdb.set_trace()
    assert calib_data['camera_model'] == 'pinhole'

    # load intrinsic matrix
    k = calib_data['intrinsics'] # [fu fv pu pv]
    K = np.diag([k[0],k[1],1])
    K[0,2] = k[2]
    K[1,2] = k[3]
    self.K = K

    # image width and height
    [width, height] = calib_data['resolution']

    # extrinsics
    Rt = np.hstack((np.eye(3),np.array([[0],[0],[0]])))
    self.Rt = Rt

    # projection
    self.P = np.dot(K,Rt)

    # center
    self.c = -np.dot(self.Rt[:3,:3].T,self.Rt[:,3])


if __name__ == '__main__':
  ## setup camera
  cam = Camera('test_calib.yaml')

  ## test projection

  # load points
  points = np.loadtxt('house.p3d').T
  points = np.vstack((points,np.ones(points.shape[1])))
  x = cam.project(points)

  # plot projection 
  plt.figure()
  # plt.subplot(1,2,1)
  plt.plot(x[0],x[1],'k.')
  # plt.show()

  # save camera calibration
  cam.save_cal()


  # # create transformation
  # r = 0.05*np.random.rand(3)
  # rot = rotation_matrix(r)

  # # rotate camera and project
  # plt.subplot(1,2,2)
  # for t in range(20):
  #   cam.P = np.dot(cam.P,rot)
  #   x = cam.project(points)
  #   plt.plot(x[0],x[1],'k.')
  # # plt.show()

  # # factorize camera matrix P
  # K = np.array([[1000,0,500],[0,1000,300],[0,0,1]])
  # tmp = rotation_matrix([0,0,1])[:3,:3]
  # Rt = np.hstack((tmp,np.array([[50],[40],[30]])))
  # cam = Camera(np.dot(K,Rt))
  # print K,Rt
  # print cam.factor()

  # # create and save camera calibration
  # K = my_calibration((480,640))
  # tmp = rotation_matrix([0,0,1])[:3,:3]
  # Rt = np.hstack((tmp,np.array([[0],[0],[20]])))
  # cam = Camera(np.dot(K,Rt))
  # cam.save_cal()