#!/usr/bin/env python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
import skimage
from skimage.io import imread


def persp2polar(x, y, f, theta0=0., phi0=0.):
	rx = x/f
	ry = y/f

	r_2 = rx**2+ry**2
	# source: http://speleotrove.com/pangazer/gnomonic_projection.html
	costheta0 = np.cos(theta0)
	sintheta0 = np.sin(theta0)
	cos_c = 1./np.sqrt(1.+r_2)

	if r_2 == 0:
		theta = 0.
		phi = 0.
	else:
		theta = np.arcsin((sintheta0 + ry*costheta0)*cos_c)
		phi = phi0 + np.arctan2(rx, (costheta0 - ry*sintheta0))
	return theta, phi

def polar2persp(theta, phi, f, theta0=0., phi0=0.):
	costheta = np.cos(theta)
	costheta0 = np.cos(theta0)
	cosphi = np.cos(phi-phi0)
	sinphi = np.sin(phi-phi0)
	sintheta = np.sin(theta)
	sintheta0 = np.sin(theta0)

	x = costheta*sinphi/(sintheta0*sintheta+costheta0*costheta*cosphi)
	y = (costheta0*sintheta-sintheta0*costheta*cosphi)/(sintheta0*sintheta+costheta0*costheta*cosphi)
	return x*f, y*f

def polar2cartesian(theta, phi):
	r = np.cos(theta)
	x = r*np.sin(phi)
	z = r*np.cos(phi)
	y = np.sin(theta)

	return x, y, z

def fish2persp(ux, uy, vp=[0.,0.], f_u=500, f_d=774., c=[1224., 1024.], src_size=[2448, 2048], dst_size=[500.,500.]):
	theta_adj = vp[0]
	phi_adj = vp[1]

	u_width = dst_size[0]
	u_height = dst_size[1]
	d_width = src_size[0]
	d_height = src_size[1]

	rx = ux - u_width/2.
	ry = uy - u_height/2.
	theta, phi = persp2polar(rx, ry, f_u, theta0=theta_adj, phi0=phi_adj)
	x, y, z = polar2cartesian(theta, phi)

	r = np.sqrt(x**2+y**2)
	theta = np.arctan2(r,z)
	R = f_d*theta

	if r == 0:
		sx, sy = c[0], c[1]
	else:
		sx, sy = c[0]+R*x/r, c[1]+R*y/r
	if sx < 0 or sy < 0 or sx > d_width-1 or sy > d_height-1:
		return (None, None)
	return (int(round(sx)), int(round(sy)))

def fish2polar(x, y, f):
	# project to sphere to get cartesian coordinates
	R = np.sqrt(x**2+y**2)
	theta = R/f
	z = R/np.tan(theta)
	r = np.sqrt(x**2+y**2+z**2)
	# get polar angles from cartesian coordinates
	phi = np.arctan2(x,z)
	theta = np.arcsin(y/r)
	return theta, phi

def polar2fish(theta, phi, f):
	x, y, z = polar2cartesian(theta, phi)
	# z = 1.0
	r = np.sqrt(x**2+y**2+z**2)
	if r == 0:
		theta = 0
	else:
		theta = np.arccos(z/r)
	rp = np.sqrt(x**2+y**2)
	if rp == 0:
		px = c[0]
		py = c[1]
	else:
		R = f*theta
		px = R*x/rp
		py = R*y/rp
	return px, py

def roi2persp(roi, f_u=500., f_d=774., c=[1224,1024], src_size=[2448.,2048.]):
	d_width = src_size[0]
	d_height = src_size[1]

	[dx, dy, dw, dh] = roi

	# import pdb; pdb.set_trace()

	# # project to sphere to get cartesian coordinates
	dxmid, dymid = dx+dw/2-c[0], dy+dh/2-c[1]
	dxmin, dxmax = dx-c[0], dx+dw-c[0]
	dymin, dymax = dy-c[1], dy+dh-c[1]

	theta_adj, phi_adj = fish2polar(dxmid, dymid, f_d)

	theta_1, phi_1 = fish2polar(dxmin, dymin, f_d)
	theta_2, phi_2 = fish2polar(dxmin, dymax, f_d)
	theta_3, phi_3 = fish2polar(dxmax, dymin, f_d)
	theta_4, phi_4 = fish2polar(dxmax, dymax, f_d)
	rx_1, ry_1 = polar2persp(theta_1, phi_1, f_u, theta0=theta_adj, phi0=phi_adj)
	rx_2, ry_2 = polar2persp(theta_2, phi_2, f_u, theta0=theta_adj, phi0=phi_adj)
	rx_3, ry_3 = polar2persp(theta_3, phi_3, f_u, theta0=theta_adj, phi0=phi_adj)
	rx_4, ry_4 = polar2persp(theta_4, phi_4, f_u, theta0=theta_adj, phi0=phi_adj)

	rxmin, rymin = np.min([rx_1, rx_2, rx_3, rx_4]), np.min([ry_1, ry_2, ry_3, ry_4])
	rxmax, rymax = np.max([rx_1, rx_2, rx_3, rx_4]), np.max([ry_1, ry_2, ry_3, ry_4])

	width, height = rxmax-rxmin, rymax-rymin

	return [int(width), int(height), theta_adj, phi_adj]

if __name__ == "__main__":
	import cuda_utils as cu
	source = skimage.img_as_float(imread("0.png"))
	source = source[:,:,::-1]
	scale = 1000./np.min(source.shape[:2])
	source_scaled = cv2.resize(source,None,fx=scale,fy=scale)
	while True:
		fromCenter = False
		showCrosshair = False
		r = cv2.selectROI("Object Selector", source_scaled, fromCenter, showCrosshair)
		r = np.array(r)/scale

		[w, h, dt, dp] = roi2persp(r, f_u=250)

		# [w,h,dp,dt] = [500,500,0,0*np.pi/180.]
		USE_CUDA = True
		if USE_CUDA:
			out = cu.fish2persp_cuda(source, [dt,dp], fd=250, fs=774, c=[1224,1024], src_size=[2448, 2048], dst_size=[w,h])
		else:
			out = np.zeros((h,w,3))
			for x in range(w):
				for y in range(h):
					sp = fish2persp(x, y, f_u=250., vp=[dt,dp], dst_size=[w,h])
					if None in sp:
						out[y,x,:] = [0,0,0]
					else:
						out[y,x,:] = source[sp[1],sp[0],:]

		cv2.imshow("Perspective", out)
		cv2.waitKey(0)
