import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

project_fisheye_mod = SourceModule("""
  __global__ void project_fisheye(float *p3d, float *p2d, float *k, float *c, float f, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
      float x = p3d[idx];
      float y = p3d[n+idx];
      float z = p3d[2*n+idx];
      float r = sqrtf(x*x+y*y+z*z);
      float theta = acos(z/r);
      float theta_2 = theta*theta;
      float theta_4 = theta_2*theta_2;
      float theta_6 = theta_4*theta_2;
      float theta_8 = theta_6*theta_2;
      float theta_d = theta * ((1 + k[0] * theta_2 + k[1] * theta_4 + k[2] * theta_6 + k[3] * theta_8));
      float rp = sqrtf(x*x+y*y);
      float R = f*theta_d;
      p2d[idx] = (R*x/rp) + c[0];
      p2d[n+idx] = (R*y/rp) + c[1];
    }
  }
  """)

persp_fisheye_mod = SourceModule("""
  #include <stdio.h>
  __global__ void fish2persp(float *src, float *dst, float *vp,
        float *c, float fs, float fd, int *src_size, int *dst_size, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
      int ux = idx % dst_size[0];
      int uy = idx / dst_size[0];

      float rx = ux - (dst_size[0]/2);
      float ry = uy - (dst_size[1]/2);

      // perspective to polar
      rx = rx/fd;
      ry = ry/fd;
      float r_2 = rx*rx+ry*ry;
      float costheta0 = cos(vp[0]);
      float sintheta0 = sin(vp[0]);
      float cos_c = 1.0/sqrtf(1.0+r_2);
      float theta, phi;
      if (r_2 == 0){
        theta = 0;
        phi = 0;
      }
      else{
        theta = asin((sintheta0 + ry*costheta0)*cos_c);
        phi = vp[1] + atan2(rx, (costheta0 - ry*sintheta0));
      }

      // polar to cartesian
      float r = cos(theta);
      float x = r*sin(phi);
      float z = r*cos(phi);
      float y = sin(theta);

      // cartesian to fisheye
      r = sqrtf(x*x+y*y);
      theta = atan2(r,z);
      float R = fs*theta;
      int sx, sy;
      if (r == 0){
          sx = c[0]-1;
          sy = c[1]-1;
      }
      else{
          sx = (R*x/r) + c[0]-1;
          sy = (R*y/r) + c[1]-1;
      }
      if (sx<0 || sy<0 || sx>src_size[0]-1 || sy>src_size[1]-1){
        int didx = 3*(ux+dst_size[0]*uy);
        dst[didx] = 0;
        dst[didx+1] = 0;
        dst[didx+2] = 0;
      }
      else{
        // offset=n_3+N_3*(n_2+N_2*n_1)
        int didx = 3*(ux+dst_size[0]*uy);
        int sidx = 3*(sx+src_size[0]*sy);
        dst[didx] = src[sidx];
        dst[didx+1] = src[sidx+1];
        dst[didx+2] = src[sidx+2];
        //printf("%d\\n", sidx);
      }
    }
  }
  """)

test_mod = SourceModule("""
  __global__ void test(float *in, float *out, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        int row3d = 3*idx;
        out[row3d] = in[row3d]*idx;
        out[row3d+1] = in[row3d+1]*idx;
        out[row3d+2] = in[row3d+2]*idx;
    }
  }
  """)

def fish2persp_numpy(src, dst, vp, c, fs, fd, src_size, dst_size):
  src_flat = src.flatten()
  dst_flat = dst.flatten()
  for idx in range(dst_size[0]*dst_size[1]):
    ux = idx % dst_size[0]
    uy = idx / dst_size[0]

    rx = ux - (dst_size[0]/2)
    ry = uy - (dst_size[1]/2)

    # perspective to polar
    rx = rx/fd
    ry = ry/fd
    r_2 = rx*rx+ry*ry
    costheta0 = np.cos(vp[0])
    sintheta0 = np.sin(vp[0])
    cos_c = 1.0/np.sqrt(1.0+r_2)
    if r_2 == 0:
      theta = 0
      phi = 0
    else:
      theta = np.arcsin((sintheta0 + ry*costheta0)*cos_c)
      phi = vp[1] + np.arctan2(rx, (costheta0 - ry*sintheta0))

    # polar to cartesian
    r = np.cos(theta)
    x = r*np.sin(phi)
    z = r*np.cos(phi)
    y = np.sin(theta)

    # cartesian to fisheye
    r = np.sqrt(x*x+y*y)
    theta = np.arctan2(r,z)
    R = fs*theta
    if r == 0:
        sx = int(c[0]-1)
        sy = int(c[1]-1)
    else:
        sx = int((R*x/r) + c[0]-1)
        sy = int((R*y/r) + c[1]-1)
    didx = int(3*(ux+dst_size[0]*uy))
    sidx = int(3*(sx+src_size[0]*sy))
    if sx<0 or sy<0 or sx>src_size[0]-1 or sy>src_size[1]-1:
      dst_flat[didx] = 0
      dst_flat[didx+1] = 0
      dst_flat[didx+2] = 0
    else:
      # offset=n_3+N_3*(n_2+N_2*n_1)
      dst_flat[didx] = src_flat[sidx]
      dst_flat[didx+1] = src_flat[sidx+1]
      dst_flat[didx+2] = src_flat[sidx+2]
      #printf("%d\\n", sidx);
    # import pdb; pdb.set_trace()
  dst = dst_flat.reshape((dst_size[1],dst_size[0],3))
  return dst

def test(p3d):
    p3d = p3d.astype(np.float32).transpose()
    thread_num = p3d.shape[0]
    max_block_size = 512
    if thread_num <= max_block_size:
        block_size = thread_num
        grid_size = 1
        p3d_tmp = p3d
    else:
        block_size = max_block_size
        div = int(thread_num / block_size)
        rem = int(thread_num % block_size)
        grid_size = div + int(rem>0)
        p3d_tmp = np.concatenate((p3d, np.zeros((rem,3))), axis=0)
    out = np.zeros((p3d_tmp.shape[0],3), dtype=np.float32)
    func = test_mod.get_function("test")
    func(cuda.In(p3d_tmp), cuda.Out(out), np.int32(thread_num),
        block=(block_size,1,1),
        grid=(grid_size,1))

    return out[:,:thread_num].transpose()

def project_points_fisheye(p3d, k, c, f):
    # p3d = np.copy(p3d, order='C')
    k = np.array(k, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    p3d = p3d.astype(np.float32)
    thread_num = p3d.shape[1]
    max_block_size = 512
    if thread_num <= max_block_size:
        block_size = thread_num
        grid_size = 1
    else:
        block_size = max_block_size
        div = int(thread_num / block_size)
        rem = int(thread_num % block_size)
        grid_size = div + int(rem>0)
        # p3d_tmp = np.concatenate((p3d, np.zeros((3,block_size-rem))), axis=1)
    p2d = np.zeros((2,thread_num), dtype=np.float32)

    func = project_fisheye_mod.get_function("project_fisheye")
    func(cuda.In(p3d), cuda.Out(p2d), cuda.In(k), cuda.In(c), np.float32(f),
        np.int32(thread_num),
        block=(block_size,1,1),
        grid=(grid_size,1))

    # return p2d[:,:thread_num].transpose()
    return p2d[:,:thread_num]

def fish2persp_cuda(img, vp, fd, fs, c, src_size, dst_size):
  img = img.astype(np.float32)
  vp = np.array(vp, dtype=np.float32)
  c = np.array(c, dtype=np.float32)
  src_size = np.array(src_size, dtype=np.int32)
  dst_size = np.array(dst_size, dtype=np.int32)
  fd = np.float32(fd); fs = np.float32(fs)

  thread_num = dst_size[0]*dst_size[1]
  max_block_size = 512
  if thread_num <= max_block_size:
    block_size = thread_num
    grid_size = 1
  else:
    block_size = max_block_size
    div = int(thread_num / block_size)
    rem = int(thread_num % block_size)
    grid_size = div + int(rem>0)
  pimg = np.zeros((dst_size[1],dst_size[0],3), dtype=np.float32)

  func = persp_fisheye_mod.get_function("fish2persp")
  func(cuda.In(img), cuda.Out(pimg), cuda.In(vp), cuda.In(c), fs, fd,
    cuda.In(src_size), cuda.In(dst_size),
    np.int32(thread_num),
    block=(block_size,1,1),
    grid=(grid_size,1))

  # pimg = fish2persp_numpy(img, pimg, vp, c, fs, fd, src_size, dst_size)

  return pimg


def project_fisheye_numpy(p3d, k, c, f):
  p = p3d
  r = np.sqrt(p[0]**2+p[1]**2+p[2]**2)
  theta = np.arccos(p[2]/r)
  d_theta = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8);
  R = f*d_theta
  rp = np.sqrt(p[0]**2+p[1]**2)
  p2d = np.zeros((2,))
  p2d[0] = R*p[0]/rp + c[0]
  p2d[1] = R*p[1]/rp + c[1]
  return p2d


if __name__ == "__main__":
    p3d = np.random.randn(3,513)
    x2d = project_points_fisheye(p3d, [0,0,0,0], [1224,1024], 750.1)
    print(x2d)
    x2d = np.zeros((2,p3d.shape[1]))
    for i in range(p3d.shape[1]):
      x2d[:,i] = project_fisheye_numpy(p3d[:,i], [0,0,0,0], [1224,1024], 750.1)
    print(x2d)

    # p3d = np.ones((3,5))
    # out = test(p3d)
    # print(out)

    # # create local numpy array
    # a = np.random.randn(4,4)
    # a = a.astype(np.float32)
    
    # # allocate memory on gpu
    # a_gpu = cuda.mem_alloc(a.nbytes)
    # cuda.memcpy_htod(a_gpu, a)

    # # cuda program to double array values
    # mod = SourceModule("""
    #   __global__ void doublify(float *a)
    #   {
    #     int idx = threadIdx.x + threadIdx.y*4;
    #     a[idx] *= 2;
    #   }
    #   """)

    # # call cuda program
    # func = mod.get_function("doublify")
    # # func(a_gpu, block=(4,4,1))
    # grid = (1,1)
    # block=(4,4,1)
    # func.prepare("P")
    # func.prepared_call(grid, block, a_gpu)
    # # this call overwrites a with result, so no need for
    # # allocation or fetch
    # # func(cuda.InOut(a), block=(4,4,1))

    # # fetch data from gpu
    # a_doubled = np.empty_like(a)
    # cuda.memcpy_dtoh(a_doubled, a_gpu)
    # print a_doubled
    # print a

    # # using gpu arrays
    # a_gpu = gpuarray.to_gpu(np.random.randn(4,4).astype(np.float32))
    # a_doubled = (2*a_gpu).get()
    # print a_doubled
    # print a_gpu