import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import pygame, pygame.image, pygame.font
from pygame.locals import *

import math, random
import numpy as np
from numpy import linalg
import pickle
import objloader
import os.path
import logging

import pyassimp
from pyassimp.postprocess import *
from pyassimp.helper import *

logger = logging.getLogger("pyassimp")
gllogger = logging.getLogger("OpenGL")
gllogger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
# OpenGL.ERROR_ON_COPY = True
# OpenGL.FULL_LOGGING = True

## constants
# INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
#                            [-1.0,-1.0,-1.0,-1.0],
#                            [-1.0,-1.0,-1.0,-1.0],
#                            [ 1.0, 1.0, 1.0, 1.0]])

TRAN_ADJ_MATRIX = np.array([ 1.0, -1.0, -1.0])

ROT_ADJ_PRE_MATRIX = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
ROT_ADJ_POST_MATRIX = np.array([[1,0,0],[0,0,-1],[0,1,0]])

ROTATION_180_X = numpy.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=numpy.float32)

DEFAULT_CLIP_PLANE_NEAR = 0.001
DEFAULT_CLIP_PLANE_FAR = 1000.0

def setup(width, height):
  """ Setup window and pygame environment. """
  # glutInit()
  pygame.init()
  pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF)
  pygame.display.set_caption('OpenGL AR demo')
  pygame.key.set_repeat(100, 10)
  # glEnable(GL_BLEND)
  # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def set_projection_from_camera(K, width, height):
  """  Set view from a camera calibration matrix. """

  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()

  fx = K[0,0]
  fy = K[1,1]
  fovy = 2*np.arctan(0.5*height/fy)*180/np.pi
  aspect = (width*fy)/(height*fx)

  # define the near and far clipping planes
  near = 0.001
  far = 1000.0

  # set perspective
  gluPerspective(fovy,aspect,near,far)
  glViewport(0,0,width,height)


def set_modelview_from_camera(Rt):
  """  Set the model view matrix from camera pose. """

  # setup 4*4 model view matrix
  M = np.eye(4)
  M[:3,:3] = np.dot(np.dot(ROT_ADJ_PRE_MATRIX,Rt[:3,:3]),ROT_ADJ_POST_MATRIX)
  M[:3,3] = Rt[:,3] * TRAN_ADJ_MATRIX
  M = M.T

  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glLoadMatrixf(M)


def load_bg_image(imname):
  # load background image (should be .bmp) to OpenGL texture
  bg_image = pygame.image.load(imname).convert()
  bg_data = pygame.image.tostring(bg_image,"RGBX",1)
  return bg_data

def make_bg_texture(bg_data):
  # bind the texture
  bg_texture = glGenTextures(1)
  glEnable(GL_TEXTURE_2D)
  glBindTexture(GL_TEXTURE_2D,bg_texture)
  glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
  return bg_texture


def draw_background(bg_texture):
  """  Draw background image using a quad. """
  glDisable(GL_LIGHTING)
  glDisable(GL_LIGHT0)
  glDisable(GL_DEPTH_TEST)

  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

  # bind the texture
  glEnable(GL_TEXTURE_2D)
  glBindTexture(GL_TEXTURE_2D, bg_texture)

  # create quad to fill the whole window
  glBegin(GL_QUADS)
  glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
  glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
  glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
  glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
  glEnd()

def draw_teapot(size):
  """ Draw a red teapot at the origin. """
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glEnable(GL_DEPTH_TEST)
  glClear(GL_DEPTH_BUFFER_BIT)

  # draw red teapot
  glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
  glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
  glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
  glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
  glutSolidTeapot(size)


def load_model(obj_name):
  obj = objloader.OBJ(obj_name+'.obj',swapyz=True)
  return obj

def draw_model(obj):
  """  Loads a model from an .obj file using objloader.py.
    Assumes there is a .mtl material file with the same name. """
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glEnable(GL_DEPTH_TEST)
  glClear(GL_DEPTH_BUFFER_BIT)

  # set model color
  glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
  glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.75,1.0,0.0])
  glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)

  # load from a file
  # obj = objloader.OBJ(obj_name+'.obj',swapyz=True)
  glCallList(obj.gl_list)

if __name__ == '__main__':
  width,height = 640,480

  setup(width, height)

  obj_name = 'mug'
  img_name = '000482-color.png'

  # load camera data
  with open('ar_camera.pkl','r') as f:
    K = pickle.load(f)
    # Rt = pickle.load(f)

  # Rt = np.hstack((np.eye(3),np.array([[0],[0],[-1]])))
  Rt =  np.array([[0.9285,   -0.3686,    0.0458,    0.0957],
                  [-0.1438,   -0.4703,   -0.8707,    0.0042],
                  [0.3424,    0.8018,   -0.4897,    0.6644]])
  # Rt[:3,:3] = Rt[:3,:3].T
  # Rt[:,3] = -Rt[:,3]
  # draw_teapot(1)
  # load_and_draw_model(obj_name)
  bg_data = load_bg_image(img_name)
  bg_texture = make_bg_texture(bg_data)
  obj = load_model(obj_name)
  print("Loaded model")

  while True:
    draw_background(bg_texture)
    set_projection_from_camera(K, width, height)
    event = pygame.event.poll()
    if event.type == QUIT:
      break
    elif event.type == KEYDOWN:
      if event.key == pygame.K_LEFT:
        Rt[0,3] -= 0.01
      elif event.key == pygame.K_RIGHT:
        Rt[0,3] += 0.01
      elif event.key == pygame.K_UP:
        Rt[1,3] -= 0.01
      elif event.key == pygame.K_DOWN:
        Rt[1,3] += 0.01
      elif event.key == pygame.K_w:
        Rt[2,3] += 0.01
      elif event.key == pygame.K_s:
        Rt[2,3] -= 0.01
    set_modelview_from_camera(Rt)
    draw_model(obj)
    pygame.display.flip()
