#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import OpenGL.GLU as GLU              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL

import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D resource loader
import pyassimp.errors              # Assimp error management + exceptions
from PIL import Image # load images for textures
import yaml

from transform import *
from cuda_utils import project_points_fisheye

# import logging
# logger = logging.getLogger("pyassimp")
# handler = logging.StreamHandler()
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)


# ------------  simple color fragment shader demonstrated in Practical 1 ------
COLOR_VERT = """#version 330 core
uniform mat4 modelviewprojection;
uniform vec3 color;
layout(location = 0) in vec3 position;
out vec3 fragColor;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    fragColor = color;
}"""

FISHEYE_COLOR_VERT = """#version 330 core
uniform mat4 modelviewprojection;
uniform mat4 modelview;
uniform vec3 color;
uniform float fov;
layout(location = 0) in vec3 position;
out vec3 fragColor;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    vec4 tmp_point = modelview * vec4(position, 1);

    float f = 1/(fov/2);
    float r = length(tmp_point.xyz);
    float theta = acos(tmp_point.z/r);
    float R = f*theta;
    gl_Position.xy = R * tmp_point.xy / length(tmp_point.xy);

    fragColor = color;
}"""


COLOR_FRAG = """#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main() {
    outColor = vec4(fragColor, 1);
}"""

# -------------- Example texture plane class ----------------------------------
TEXTURE_VERT = """#version 330 core
uniform mat4 modelviewprojection;
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texposition;
out vec2 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    fragTexCoord = texposition;
}"""

FISHEYE_TEXTURE_VERT = """#version 330 core
uniform mat4 modelviewprojection;
uniform mat4 modelview;
uniform float fov;
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texposition;
out vec2 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    vec4 tmp_point = modelview * vec4(position, 1);
    float r = length(tmp_point.xyz);

    //if (r==0){
    //  gl_Position.x = 0;
    //  gl_Position.y = 0;
    //}
    //else
    //  gl_Position.xy = tmp_point.xy / (2*r);

    float f = 1/(fov/2);
    float theta;
    if (r == 0)
      theta = 0;
    else
      theta = acos(tmp_point.z/r);
    float R = f*theta;
    gl_Position.xy = R * tmp_point.xy / length(tmp_point.xy);
    gl_Position.xy = gl_Position.xy / 2.0;
    
    fragTexCoord = texposition;
}"""

TEXTURE_FRAG = """#version 330 core
uniform sampler2D diffuseMap;
uniform float alpha;
in vec2 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
    outColor.a = alpha;
}"""

# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        try:
            # imports image as a numpy array in exactly right format
            tex = np.array(Image.open(file))
            format = format[0 if len(tex.shape) == 2 else tex.shape[2] - 1]
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, format, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            # message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            # print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except IOError as e:
          print(os.strerror(e.errno)) 
        # except FileNotFoundError:
        #     print("ERROR: unable to load texture file %s" % file)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

class Background:
    def __init__(self, file):
      self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

      # quad to fill whole window
      #vertices = np.array(((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)), np.float32)
      vertices = np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
      texCoords = np.array(((0, 1), (1, 1), (1, 0), (0, 0)), np.float32)
      faces = np.array(((0, 1, 2), (0, 2, 3)), np.uint32)
      self.vertex_array = VertexArray([vertices, texCoords], faces)

      # background image as texture
      self.wrap_mode = GL.GL_CLAMP_TO_EDGE
      self.filter_mode = (GL.GL_NEAREST, GL.GL_NEAREST)

      # setup texture and upload it to GPU
      self.texture = Texture(file, self.wrap_mode, *self.filter_mode)

    def set(self, file):
      self.texture = Texture(file, self.wrap_mode, *self.filter_mode)

    def draw(self):
      """  Draw background image using a quad. """
      GL.glUseProgram(self.shader.glid)
      GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

      # projection geometry
      loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
      GL.glUniformMatrix4fv(loc, 1, True, np.eye(4))

      # texture access setups
      loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
      GL.glActiveTexture(GL.GL_TEXTURE0)
      GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
      GL.glUniform1i(loc, 0)
      self.vertex_array.execute(GL.GL_TRIANGLES)

      # clear depth for background
      GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

      # leave clean state for easier debugging
      GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
      GL.glUseProgram(0)


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers += [GL.glGenBuffers(1)]
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)




# ------------  Scene object classes ------------------------------------------

class Node(object):
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        model = np.dot(model, self.transform) # what to insert here for hierarchical update?
        for child in self.children:
            child.draw(projection, view, model, **param)


class BoundingBox:
    def __init__(self, coords, width=0.01, color=[1,0,0]):
      self.shader = Shader(COLOR_VERT, COLOR_FRAG)

      self.color = np.array(color)

      minx, maxx, miny, maxy = coords # normalized coordinates as [minx, maxx, miny, maxy]

      vertices = np.array(((minx, maxy+width, 0), (minx-width, maxy+width, 0), (minx-width, maxy, 0),
                           (minx-width, miny, 0), (minx-width, miny-width, 0), (minx, miny-width, 0),
                           (maxx, miny-width, 0), (maxx+width, miny-width, 0), (maxx+width, miny, 0),
                           (maxx+width, maxy, 0), (maxx+width, maxy+width, 0), (maxx, maxy+width, 0)), np.float32)
      # vertices = np.clip(vertices, -1, 1)
      faces = np.array(((0, 1, 4), (0, 4, 5),
                        (8, 3, 4), (8, 4, 7),
                        (10, 11, 6), (10, 6, 7),
                        (10, 1, 2), (10, 2, 9)), np.uint32)
      self.vertex_array = VertexArray([vertices], faces)

      # background image as texture
      self.wrap_mode = GL.GL_CLAMP_TO_EDGE
      self.filter_mode = (GL.GL_NEAREST, GL.GL_NEAREST)

    def draw(self):
      """  Draw background image using a quad. """
      GL.glUseProgram(self.shader.glid)

      # projection geometry
      loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
      GL.glUniformMatrix4fv(loc, 1, True, np.eye(4))

      loc = GL.glGetUniformLocation(self.shader.glid, 'color')
      GL.glUniform3fv(loc, 1, self.color)

      self.vertex_array.execute(GL.GL_TRIANGLES)

      # leave clean state for easier debugging
      GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
      GL.glUseProgram(0)


class Marker:
    def __init__(self, coords, color=[1,0,0]):
      self.shader = Shader(COLOR_VERT, COLOR_FRAG)

      self.color = np.array(color)

      minx, maxx, miny, maxy = coords # normalized coordinates as [minx, maxx, miny, maxy]
      vertices = np.array(((minx, maxy, 0),
                           (minx, miny, 0),
                           (maxx, miny, 0),
                           (maxx, maxy, 0)), np.float32)
      # vertices = np.clip(vertices, -1, 1)
      faces = np.array(((0, 1, 2), (3, 0, 2)), np.uint32)
      self.vertex_array = VertexArray([vertices], faces)

      # background image as texture
      self.wrap_mode = GL.GL_CLAMP_TO_EDGE
      self.filter_mode = (GL.GL_NEAREST, GL.GL_NEAREST)

    def draw(self):
      """  Draw background image using a quad. """
      GL.glUseProgram(self.shader.glid)

      # projection geometry
      loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
      GL.glUniformMatrix4fv(loc, 1, True, np.eye(4))

      loc = GL.glGetUniformLocation(self.shader.glid, 'color')
      GL.glUniform3fv(loc, 1, self.color)

      self.vertex_array.execute(GL.GL_TRIANGLES)

      # leave clean state for easier debugging
      GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
      GL.glUseProgram(0)


class TexturedPlane:
    """ Simple first textured object """

    def __init__(self, file):
        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # triangle and face buffers
        vertices = 100 * np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
        faces = np.array(((0, 1, 2), (0, 2, 3)), np.uint32)
        self.vertex_array = VertexArray([vertices, vertices[:,:2]], faces)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = file

        # setup texture and upload it to GPU
        self.texture = Texture(file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, np.dot(np.dot(projection,view), model))

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.execute(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


class TexturedMesh:
    """ Textured object class """
    def __init__(self, obj_name, mesh, texture, attributes, indices):
        self.obj_name = obj_name # name associated with meshes of object instance
        self.mesh = mesh
        # feel free to move this up in the viewer as per other practicals
        #self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        self.model_matrix = np.eye(4)
        # self.model_matrix[2,3] = -1

        # triangle and face buffers
        self.vertex_array = VertexArray(attributes, indices)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = texture

        # object transparancy
        self.alpha = 1.0

        # setup texture and upload it to GPU
        self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

    def draw(self, shader, projection, view, model=None, is_fisheye=False, fish_fov=np.pi, win=None, **_kwargs):
        if model is None:
          model = self.model_matrix
        else:
          self.model_matrix = model

        # some interactive elements
        if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        GL.glUseProgram(shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, np.dot(np.dot(projection,view),model))

        if is_fisheye:
          loc = GL.glGetUniformLocation(shader.glid, 'modelview')
          GL.glUniformMatrix4fv(loc, 1, True, mat_from_gl(np.dot(view,model)))
          loc = GL.glGetUniformLocation(shader.glid, 'fov')
          GL.glUniform1f(loc, fish_fov)

        # texture access setups
        #GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

        # object transparancy
        loc = GL.glGetUniformLocation(shader.glid, 'alpha')
        GL.glUniform1f(loc, self.alpha)

        #GL.glColor4f(1.0, 1.0, 1.0, 0.5)
        loc = GL.glGetUniformLocation(shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.execute(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_BLEND)


# mesh to refactor all previous classes
class ColorMesh(object):

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader, **param):

        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(GL.GL_TRIANGLES)


class SimpleTriangle(ColorMesh):
    """Hello triangle object"""

    def __init__(self):

        # triangle position buffer
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        super().__init__([position, color])


class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        super(Cylinder, self).__init__()
        self.add(*load('cylinder.obj'))  # just load the cylinder from file


# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


# -------------- 3D textured mesh loader ---------------------------------------
def load_textured(file, obj_name):
    """ load resources using pyassimp, return list of TexturedMeshes """
    # try:
    option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
    scene = pyassimp.load(file, option)
    # except pyassimp.errors.AssimpError:
    # except Exception as e:
    #     print('pyassimp unable to load', file)
    #     print('ERROR: ', e)
    #     # print('Is mesh triangulated?')
    #     return []  # error reading => return empty list

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    path = os.path.join('.', '') if path == '' else path
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = tname[0]
            else:
                print('Failed to find texture:', tname)

    # prepare textured mesh
    meshes = []
    for mesh in scene.meshes:
        texture = scene.materials[mesh.materialindex].texture

        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)

        # create the textured mesh object from texture, attributes, and indices
        # TODO: currently only support single mesh objects in pose logic
        meshes.append(TexturedMesh(obj_name, mesh, texture, [mesh.vertices, tex_uv], mesh.faces))

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super(GLFWTrackball, self).__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


# Note: camera intrinsic center should be at center pixel coordinate
class Camera:
  def __init__(self, params, clipplanenear=0.001, clipplanefar=10.0, is_fisheye=False, name='cam0'):
    self.name = name
    self.clipplanenear = clipplanenear
    self.clipplanefar = clipplanefar

    self.is_fisheye = is_fisheye

    # params either passed as path to yaml file or as dictionary of values
    self.K = None
    self.persp_width = None
    self.persp_height = None
    self.fish_width = None
    self.fish_height = None
    self.fish_fov = 180 # assumes circular image with width fully filled
    self.fish_f = None
    self.fish_cx = None
    self.fish_cy = None
    self.fish_dist_coeff = None
    if isinstance(params, str):
      self.load_cal(params, self.name)
    else:
      self.K = params.K
      self.persp_width = params.width
      self.persp_height = params.height

    # convenient pre-calcs for for OpenGL
    fx = self.K[0,0]
    fy = self.K[1,1]
    cx = self.K[0,2]
    cy = self.K[1,2]
    self.fovy = 2*np.arctan(0.5*self.persp_height/fy)*180/np.pi
    self.aspect = (self.persp_width*fy)/(self.persp_height*fx)
    self.horizontalfov = self.fovy * np.pi / 180
    self.Cx = 2*(cx/self.persp_width)-1
    self.Cy = 2*(cy/self.persp_height) - 1

  def __str__(self):
      return self.name

  def width(self):
    if self.is_fisheye:
      return self.fish_width
    else:
      return self.persp_width

  def height(self):
    if self.is_fisheye:
      return self.fish_height
    else:
      return self.persp_height

  def load_cal(self, filename, cam_name):
    """ Load camera calibration from yaml. """

    # TODO: add distortion support
    f = file(filename, 'r')
    calib_data = yaml.load(f)
    if self.is_fisheye:
      fisheye_data = calib_data['fisheye']
      k = fisheye_data['intrinsics']
      self.fish_f = (k[0]+k[1])/2.
      [self.fish_width, self.fish_height] = fisheye_data['resolution']
      self.fish_fov = self.fish_width/self.fish_f
      self.fish_cx, self.fish_cy = k[2], k[3]
      self.fish_dist_coeff = fisheye_data['distortion_coeffs']

    calib_data = calib_data[cam_name]

    # currently only support pinhole cameras
    assert calib_data['camera_model'] == 'pinhole'

    # load intrinsic matrix
    k = calib_data['intrinsics'] # [fu fv pu pv]
    K = np.diag([k[0],k[1],1])
    K[0,2] = k[2]
    K[1,2] = k[3]
    self.K = K

    # image width and height
    [self.persp_width, self.persp_height] = calib_data['resolution']


class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, camera, background, max_dim=1000.0):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.init()             # initialize window system glfw
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        cam_width = camera.width()
        cam_height = camera.height()
        dim = max(cam_width, cam_height)
        if dim > max_dim:
          factor = max_dim / dim
        else:
          factor = 1.0
        self.win = glfw.create_window(int(cam_width*factor), int(cam_height*factor), 'Viewer', None, None)
        # glfw.set_window_aspect_ratio(self.win, cam_width, cam_height)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_window_size_callback(self.win, self.on_win_resize)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)         # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)          # backface culling enabled (TP2)

        # compile and initialize shader programs once globally
        # if camera.is_fisheye:
        #   self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
        #   self.texture_shader = Shader(FISHEYE_TEXTURE_VERT, TEXTURE_FRAG)
        # else:
        #   self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
        #   self.texture_shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
        self.texture_shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # initially empty list of object to draw
        self.drawables = []
        self.bboxes = []
        self.active_drawable = 0

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # initialize background
        self.background = Background(background)

        # initialize camera
        self.projection_matrix = None
        self.camera = camera
        self.set_camera_projection()

        # frame marker
        self.markers = []

    def __del__(self):
        glfw.terminate()        # destroy all glfw windows and GL contexts

    def set_camera_projection(self):
        znear = self.camera.clipplanenear
        zfar = self.camera.clipplanefar
        aspect = self.camera.aspect
        fovy = self.camera.fovy
        cx = self.camera.Cx
        cy = self.camera.Cy
        self.projection_matrix = perspective(fovy, aspect, znear, zfar, cx, cy)

    def set_view_from_pose_matrix(self, Rt):
      self.trackball.view_from_pose_matrix(Rt)

    def set_view_from_pose_vec(self, V, invert=False):
      self.trackball.view_from_pose_vec(V, invert)

    def set_pose_matrix(self, Rt):
      self.trackball.set_pose_matrix(mat_to_gl(Rt))
      self.drawables[self.active_drawable].model_matrix = self.trackball.model_matrix

    def get_pose_matrix(self):
      view = self.trackball.view_matrix
      model = self.trackball.model_matrix
      return mat_from_gl(np.dot(view,model))

    def get_pose_vec(self):
      Rt = self.get_pose_matrix()
      pose = np.zeros(7)
      pose[0:3] = Rt[0:3,3]
      pose[3:] = quaternion_from_matrix(Rt[0:3,0:3])
      return pose

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            if self.background is not None:
              self.background.draw()

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix
            model = self.trackball.model_matrix
            projection = self.projection_matrix #self.trackball.projection_matrix(winsize)
            
            # draw our scene objects
            for idx, drawable in enumerate(self.drawables):
                if idx == self.active_drawable:
                  drawable.draw(self.texture_shader, projection, view, model=model,
                    is_fisheye=self.camera.is_fisheye, fish_fov=self.camera.fish_fov, win=self.win)
                else:
                  drawable.draw(self.texture_shader, projection, view,
                    is_fisheye=self.camera.is_fisheye, fish_fov=self.camera.fish_fov, win=self.win)

            for bbox in self.bboxes:
              bbox.draw()

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def render(self):
        """ Main render single iteration for this OpenGL window """
        # clear draw buffer and depth buffer (<-TP2)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.background is not None:
          self.background.draw()

        winsize = glfw.get_window_size(self.win)
        view = self.trackball.view_matrix
        model = self.trackball.model_matrix
        projection = self.projection_matrix #self.trackball.projection_matrix(winsize)
        
        # draw our scene objects
        for idx, drawable in enumerate(self.drawables):
            if idx == self.active_drawable:
              drawable.draw(self.texture_shader, projection, view, model=model,
                is_fisheye=self.camera.is_fisheye, fish_fov=self.camera.fish_fov, win=self.win)
            else:
              drawable.draw(self.texture_shader, projection, view,
                is_fisheye=self.camera.is_fisheye, fish_fov=self.camera.fish_fov, win=self.win)

        for bbox in self.bboxes:
          bbox.draw()

        for marker in self.markers:
          marker.draw()

        # flush render commands, and swap draw buffers
        glfw.swap_buffers(self.win)

        # Poll for and process events
        glfw.poll_events()

    def add_bbox(self, bbox):
      minx = bbox[0]; maxx = bbox[0]+bbox[2]
      miny = bbox[1]; maxy = bbox[1]+bbox[3]

      nminx = 2*(minx-self.camera.width()/2)/float(self.camera.width())
      nmaxx = 2*(maxx-self.camera.width()/2)/float(self.camera.width())
      nmaxy = -2*(miny-self.camera.height()/2)/float(self.camera.height())
      nminy = -2*(maxy-self.camera.height()/2)/float(self.camera.height())

      self.bboxes.append(BoundingBox([nminx, nmaxx, nminy, nmaxy]))

    def add_marker(self, center=[0.5, 0.5], width=0.1, color=[1,0,0]):
      minx = center[0] - width/2.0
      miny = center[1] - width/2.0
      maxx = center[0] + width/2.0
      maxy = center[1] + width/2.0
      self.markers.append(Marker([minx, maxx, miny, maxy], color=color))

    def clear_markers(self):
      self.markers = []

    def clear_bboxes(self):
      self.bboxes = []

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def remove(self, obj_name):
      for i, drawable in enumerate(self.drawables):
        if drawable.obj_name == obj_name:
          del self.drawables[i]
          self.active_drawable = 0

    def rename(self, obj_name, new_name):
      for i, drawable in enumerate(self.drawables):
        if drawable.obj_name == obj_name:
          drawable.obj_name = new_name

    def set_active(self, obj_name):
      for i, drawable in enumerate(self.drawables):
        if drawable.obj_name == obj_name:
          self.active_drawable = i
          self.trackball.set_model_matrix(drawable.model_matrix)

    def set_alpha(self, obj_name, alpha):
      for i, drawable in enumerate(self.drawables):
        if drawable.obj_name == obj_name:
          drawable.alpha = alpha

    def get_alpha(self, obj_name):
      for i, drawable in enumerate(self.drawables):
        if drawable.obj_name == obj_name:
          return drawable.alpha
      return 1.0

    def in_frustum(self, M, pvec):
        # visible_points = []
        visible = False
        gl4d = np.dot(M, pvec)
        for i in range(gl4d.shape[1]):
          Pclip = gl4d[:,i]
          visible = visible or (abs(Pclip[0]) < Pclip[3] and
                 abs(Pclip[1]) < Pclip[3] and
                 0 < Pclip[2] and
                 Pclip[2] < Pclip[3])
        #   if visible:
        #     visible_points.append(pvec[:,i])
        # return np.array(visible_points).T
        return visible

    def project_fisheye(self, p3d):
      p = p3d
      r = np.sqrt(p[0]**2+p[1]**2+p[2]**2)
      k = self.camera.fish_dist_coeff
      theta = np.arccos(p[2]/r)
      d_theta = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8);
      # d_theta = theta + k[0]*theta + k[1]*theta**3 + k[2]*theta**5 + k[3]*theta**7
      R = self.camera.fish_f*d_theta
      rp = np.sqrt(p[0]**2+p[1]**2)
      # dx = d_theta*p[0]/rp
      # dy = d_theta*p[1]/rp
      p2d = np.zeros((2,))
      p2d[0] = R*p[0]/rp + self.camera.fish_cx
      p2d[1] = R*p[1]/rp + self.camera.fish_cy
      # w = self.camera.fish_width
      # fov = self.camera.fish_fov
      # p2d[0] = (((w/fov)*np.arctan(rp/p[2])) / np.sqrt((p[1]/p[0])**2+1)) + dx + self.camera.fish_cx
      # p2d[1] = (((w/fov)*np.arctan(rp/p[2])) / np.sqrt((p[0]/p[1])**2+1)) + dy + self.camera.fish_cy
      return p2d

    def bounding_box_fisheye(self):
      # assumes ideal fisheye model R = f*theta
      if self.drawables:
        drawable = self.drawables[self.active_drawable]
        vertices = drawable.mesh.vertices.T
        x3d = np.ones((4,vertices.shape[1]))
        x3d[0:3,:] = vertices
        view = self.trackball.view_matrix
        model = self.trackball.model_matrix
        glModelViewProj = np.dot(self.projection_matrix, np.dot(view, model))
        modelView = mat_from_gl(np.dot(view, model))
        modelView = modelView[0:3,0:4]
        p3d = np.dot(modelView, x3d)
        # x2d = np.zeros((2,p3d.shape[1]))
        # for i in range(p3d.shape[1]):
        #   x2d[:,i] = self.project_fisheye(p3d[:,i])
        x2d = project_points_fisheye(p3d, self.camera.fish_dist_coeff,
          [self.camera.fish_cx,self.camera.fish_cy], self.camera.fish_f)

          # p = p3d[:,i]
          # r = np.sqrt(p[0]**2+p[1]**2+p[2]**2)
          # theta = np.arccos(p[2]/r)
          # R = self.camera.fish_f*theta
          # xr = np.sqrt(p[0]**2+p[1]**2)
          # x2d[0,i] = R*p[0]/xr + self.camera.fish_cx
          # x2d[1,i] = R*p[1]/xr + self.camera.fish_cy
        # x2d = x2d + np.array([self.camera.fish_width/2, self.camera.fish_height/2])[:,np.newaxis]
        minx = min(x2d[0,:]); maxx = max(x2d[0,:])
        miny = min(x2d[1,:]); maxy = max(x2d[1,:])
        if (maxx < 0 or minx > self.camera.fish_width or maxy < 0 or miny > self.camera.fish_height):
        # if not self.in_frustum(glModelViewProj, x3d):
          maxx = minx = maxy = miny = 0
        else:
          minx = max(0, minx); maxx = min(self.camera.fish_width, maxx)
          miny = max(0, miny); maxy = min(self.camera.fish_height, maxy)
        return [minx, miny, maxx-minx, maxy-miny]
      else:
        return None

    def bounding_box(self):
      if self.drawables:
        drawable = self.drawables[self.active_drawable]
        vertices = drawable.mesh.vertices.T
        x3d = np.ones((4,vertices.shape[1]))
        x3d[0:3,:] = vertices
        view = self.trackball.view_matrix
        model = self.trackball.model_matrix
        glModelViewProj = np.dot(self.projection_matrix, np.dot(view, model))
        modelView = mat_from_gl(np.dot(view, model))
        modelView = modelView[0:3,0:4]
        P = np.dot(self.camera.K, modelView)
        x2d = np.dot(P,x3d)
        x2d /= x2d[2,:]
        minx = min(x2d[0,:]); maxx = max(x2d[0,:])
        miny = min(x2d[1,:]); maxy = max(x2d[1,:])
        if (maxx < 0 or minx > self.camera.persp_width or maxy < 0 or miny > self.camera.persp_height
          or maxx - minx > 10000 or maxy-miny > 10000 or modelView[2,3] < 0):
        # if not self.in_frustum(glModelViewProj, x3d):
          maxx = minx = maxy = miny = 0
        else:
          minx = max(0, minx); maxx = min(self.camera.persp_width, maxx)
          miny = max(0, miny); maxy = min(self.camera.persp_height, maxy)
        return [minx, miny, maxx-minx, maxy-miny]
      else:
        return None

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

    def on_win_resize(self, _win, action, _mods):
      glfw.swap_buffers(self.win)

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    # camera = Camera('test_calib_ycb.yaml')
    # viewer = Viewer(camera, background='000482-color.png')
    camera = Camera('fisheye_calib.yaml', is_fisheye=True)
    viewer = Viewer(camera, background='fisheye.png')

    # place instances of our basic objects
    # viewer.add(*[mesh for file in sys.argv[1:] for mesh in load(file)])
    viewer.add(*[mesh for file in sys.argv[1:] for mesh in load_textured(file, 'test')])
    # viewer.add(*[TexturedPlane("texture_map.png")])
    if len(sys.argv) < 2:
        print('Usage:\n\t%s [3dfile]*\n\n3dfile\t\t the filename of a model in'
              ' format supported by pyassimp.' % (sys.argv[0],))

    # Rt =  np.array([[0.9285,   -0.3686,    0.0458,    0.0957],
    #               [-0.1438,   -0.4703,   -0.8707,    0.0042],
    #               [0.3424,    0.8018,   -0.4897,    0.6644]])

    Rt =  np.array([[1., 0,    0,    0],
                  [0,   1.,   0,    0.5],
                  [0,    0,   1.,    0.5]])

    viewer.set_view_from_pose_matrix(Rt)

    # viewer.bounding_box()
    bbox = viewer.bounding_box_fisheye()
    viewer.add_bbox(bbox)

    # cylinder = Cylinder()

    # # ---- let's make our shapes ---------------------------------------
    # base_shape = Node(transform=scale(x=1,y=.1,z=1))
    # base_shape.add(cylinder)
    # arm_shape = Node(transform=np.dot(translate(0,3,0),scale(.1,1,.1)))
    # arm_shape.add(cylinder)
    # forearm_shape = Node(transform=np.dot(translate(0,1,0),scale(.1,1,.1)))
    # forearm_shape.add(cylinder)

    # # ---- construct our robot arm hierarchy ---------------------------
    # theta = 45.0        # base horizontal rotation angle
    # phi1 = 45.0         # arm angle
    # phi2 = 20.0         # forearm angle

    # transform_forearm = Node(transform=rotate((0,0,1), phi2))
    # transform_forearm.add(forearm_shape)

    # transform_arm = Node(transform=rotate((0,0,1), phi1))
    # transform_arm.add(arm_shape, transform_forearm)

    # transform_base = Node(transform=rotate((0,1,0), theta))
    # transform_base.add(base_shape, transform_arm)

    # viewer.add(transform_base)
    viewer.run()


if __name__ == '__main__':
    main()                     # main function keeps variables locally scoped
