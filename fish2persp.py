#!/usr/bin/env python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
import skimage
from skimage.io import imread

def dist(x,y):
    return np.sqrt(x*x+y*y)

def correct_fisheye(dx,dy,src_size=[2448.,2048.],dest_size=[2448.,2048.],factor=3.0):
    """ returns a tuple of source coordinates (sx,sy)
        (note: values can be out of range)"""
    # convert dx,dy to relative coordinates
    rx, ry = dx-(dest_size[0]/2), dy-(dest_size[1]/2)
    # calc theta
    r = dist(rx,ry)/(dist(src_size[0],src_size[1])/factor)
    if 0==r:
        theta = 1.0
    else:
        theta = np.arctan(r)/r
    # back to absolute coordinates
    sx, sy = (src_size[0]/2)+theta*rx, (src_size[1]/2)+theta*ry
    # done
    return (int(np.round(sx)),int(np.round(sy)))

def persp2latlon(x, y, f, lat0=0., lon0=0.):
	rx = x/f
	ry = y/f

	r_2 = rx**2+ry**2
	# r = np.sqrt(r_2)
	# c = np.arctan(r)
	# sin_c = np.sin(c)
	# cos_c = np.cos(c)
	# cosphi1 = np.cos(phi1)
	# sinphi1 = np.sin(phi1)
	coslat0 = np.cos(lat0)
	sinlat0 = np.sin(lat0)
	cos_c = 1./np.sqrt(1.+r_2)

	if r_2 == 0:
		lat = 0.
		lon = 0.
	else:
		# lat = np.arcsin(ry*sin_c/r)
		# lon = np.arctan2(rx*sin_c, r*cos_c)
		lat = np.arcsin((sinlat0 + ry*coslat0)*cos_c)
		lon = lon0 + np.arctan2(rx, (coslat0 - ry*sinlat0))
	return lat, lon

def latlon2persp(lat, lon, f, lat0=0., lon0=0.):
	coslat = np.cos(lat)
	coslat0 = np.cos(lat0)
	coslon = np.cos(lon-lon0)
	sinlon = np.sin(lon-lon0)
	sinlat = np.sin(lat)
	sinlat0 = np.sin(lat0)

	x = coslat*sinlon/(sinlat0*sinlat+coslat0*coslat*coslon)
	y = (coslat0*sinlat-sinlat0*coslat*coslon)/(sinlat0*sinlat+coslat0*coslat*coslon)
	return x*f, y*f

def latlon2cartesian(lat, lon):
	# r = np.tan(np.pi-lat)
	# x = -r*np.cos(lon)
	# y = r*np.sin(lon)
	# z = 1.
	phi = lon
	theta = lat
	r = np.cos(theta)
	x = r*np.sin(phi)
	z = r*np.cos(phi)
	y = np.sin(theta)

	return x, y, z

def fish2persp_gnom(ux, uy, vp=[0.,0.], f_u=500, f_d=774., c=[1224., 1024.], src_size=[2448, 2048], dst_size=[500.,500.]):
	phi_adj = vp[0]
	theta_adj = vp[1]

	d_fov = src_size[0]/f_d
	# u_fov = 90.*np.pi/180.
	# u_width = 500.
	# u_height = 500.
	# f_u = np.absolute( u_width/( 2*np.tan(u_fov/2) ) )
	u_width = dst_size[0]
	u_height = dst_size[1]
	d_width = src_size[0]
	d_height = src_size[1]

	rx = ux - u_width/2.
	ry = uy - u_height/2.

	r = np.sqrt(rx**2+ry**2)
	rp = np.arctan2(r/f_u)*f_u
	x = rp*rx/r
	y = rp*ry/r

	lat, lon = persp2latlon(rx, ry, f_u)
	x, y, z = latlon2cartesian(lat, lon)
	import pdb; pdb.set_trace()

	# phi = np.arcsin(rx*np.sin(rho)/r)
	# theta = np.arctan2(ry*np.sin(rho), r*np.cos(rho))

	# x = np.cos(theta)*np.sin(phi)
	# z = np.cos(theta)*np.cos(phi)
	# y = np.sin(theta)

	r = np.sqrt(x**2+y**2+z**2)
	theta = np.arccos(z/r)
	rp = np.sqrt(x**2+y**2)
	R = f_d*theta

	sx, sy = c[0]+R*x/rp, c[1]+R*y/rp
	if sx < 0 or sy < 0 or sx > d_width-1 or sy > d_height-1:
		return (None, None)
	return (int(round(sx)), int(round(sy)))

def fish2persp_tmp(ux, uy, vp=[0.,0.], f_u=500, f_d=774., c=[1224., 1024.], src_size=[2448, 2048], dst_size=[500.,500.]):
	phi_adj = vp[0]
	theta_adj = vp[1]

	# u_fov = 90.*np.pi/180.
	# u_width = 500.
	# u_height = 500.
	# f_u = np.absolute( u_width/( 2*np.tan(u_fov/2) ) )
	u_width = dst_size[0]
	u_height = dst_size[1]
	d_width = src_size[0]
	d_height = src_size[1]

	rx = ux - u_width/2.
	ry = uy - u_height/2.
	lat, lon = persp2latlon(rx, ry, f_u, lat0=theta_adj, lon0=phi_adj)
	x, y, z = latlon2cartesian(lat, lon)

	# phi = np.arcsin(rx*np.sin(rho)/r)
	# theta = np.arctan2(ry*np.sin(rho), r*np.cos(rho))

	# x = np.cos(theta)*np.sin(phi)
	# z = np.cos(theta)*np.cos(phi)
	# y = np.sin(theta)

	r = np.sqrt(x**2+y**2)
	theta = np.arctan2(r,z)
	R = f_d*theta
	# import pdb; pdb.set_trace()

	if r == 0:
		sx, sy = c[0], c[1]
	else:
		sx, sy = c[0]+R*x/r, c[1]+R*y/r
	if sx < 0 or sy < 0 or sx > d_width-1 or sy > d_height-1:
		return (None, None)
	return (int(round(sx)), int(round(sy)))

def fish2persp(ux, uy, vp=[0.,0.], f_u=500, f_d=774., c=[1224., 1024.], src_size=[2448, 2048], dst_size=[500.,500.]):
	phi_adj = vp[0]
	theta_adj = vp[1]

	# u_fov = 90.*np.pi/180.
	# u_width = 500.
	# u_height = 500.
	# f_u = np.absolute( u_width/( 2*np.tan(u_fov/2) ) )
	u_width = dst_size[0]
	u_height = dst_size[1]
	d_width = src_size[0]
	d_height = src_size[1]

	rx, ry = ux-(u_width/2), uy-(u_height/2)
	R_u = np.sqrt(rx*rx+ry*ry)
	theta = np.arctan(R_u/f_u)

	R_d = f_d*theta

	# Adjust for point of view using polar angle offsets
	if R_u == 0:
		dx, dy = 0., 0.
	else:
		dx, dy = R_d*rx/R_u, R_d*ry/R_u
	dx += phi_adj*f_d
	dy += theta_adj*f_d

	sx, sy = c[0]+dx-1, c[1]+dy-1
	if sx < 0 or sy < 0 or sx > d_width-1 or sy > d_height-1:
		return (None, None)
	return (int(round(sx)), int(round(sy)))

def fish2persp2(ux, uy, vp=[0.,0.], k=[0.,0.,0.,0.], f_u=500., f_d=774., c=[1224.,1024.], dst_size=[500,500], src_size=[2448,2048]):
	ux = float(ux); uy = float(uy)
	f_u = float(f_u)
	f_d = float(f_d)
	dst_size = [float(x) for x in dst_size]
	src_size = [float(x) for x in src_size]
	c = [float(x) for x in c]
	vp = [float(x) for x in vp]

	phi_adj = vp[0]
	theta_adj = vp[1]

	d_width = src_size[0]
	d_height = src_size[1]
	d_fov = d_width/f_d

	u_width = dst_size[0]
	u_height = dst_size[1]

	# get spherical coordinates from perspective
	rx, ry = ux-(u_width/2.), uy-(u_height/2.)
	phi_u = np.arctan(rx/f_u)
	theta_u = np.arctan(ry/f_u)

	# adjust spherical angles with looking direction
	phi = phi_u + phi_adj
	theta = theta_u + theta_adj

	# adjust reference axis for spherical coordinates
	phi = (np.pi/2.) - phi
	theta = (np.pi/2.) - theta

	# spherical coordinates to 3d vector on unit sphere
	# y and z axis are swapped from standard spherical coordinate system
	x = np.sin(theta)*np.cos(phi)
	y = np.cos(theta)
	z = np.sin(theta)*np.sin(phi)

	# project to distorted fisheye image
	# note change in definition of theta
	theta = np.arccos(z)
	theta_d = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)


	R_d = f_d*theta_d
	r = np.sqrt(x*x+y*y)

	# Adjust for point of view using polar angle offsets
	if r == 0:
		dx, dy = 0., 0.
	else:
		dx, dy = R_d*x/r, R_d*y/r

	sx, sy = dx+c[0]-1, dy+c[1]-1
	if sx < 0 or sy < 0 or sx > d_width-1 or sy > d_height-1:
		return (None, None)
	return (int(round(sx)), int(round(sy)))

def fish2persp3(ux, uy, vp=[0.,0.], k=[0.,0.,0.,0.], f_u=500., f_d=774., c=[1224.,1024.], dst_size=[500,500], src_size=[2448,2048]):
	phi_adj = vp[0]
	theta_adj = vp[1]

	d_width = src_size[0]
	d_height = src_size[1]

	# get spherical coordinates from perspective
	phi_u, theta_u = persp2polar(ux, uy, f_u)
	phi, theta = phi_u+phi_adj, theta_u+theta_adj

	dx, dy = polar2fish(phi, theta, f_d)
	sx, sy = dx + d_width/2., dy + d_height/2.
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
	return phi, theta

def polar2cartesian(phi, theta):
	z = np.sin(theta)*np.cos(phi)
	x = np.sin(theta)*np.sin(phi)
	y = np.cos(theta)
	return x, y, z

def polar2fish(phi, theta, f):
	x, y, z = polar2cartesian(phi, theta)
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

def persp2polar(x, y, f):
	r = np.sqrt(x**2+y**2)
	theta = np.arctan(r/f)
	phi = np.arctan2(y,x)
	# phi = np.arctan(x/f)
	# theta = np.pi/2 - np.arctan(y/f)
	return phi, theta

def polar2perspnorm(phi, theta, f):
	r = f*np.tan(theta)
	x = r*np.cos(phi)
	y = r*np.sin(phi)	
	# x = f*np.tan(phi)
	# theta = np.pi/2 - theta
	# y = f*np.tan(theta)
	return x, y

def roi2persp4(roi, f_u=500., f_d=774., c=[1224,1024], src_size=[2448.,2048.]):
	d_width = src_size[0]
	d_height = src_size[1]

	[dx, dy, dw, dh] = roi

	# import pdb; pdb.set_trace()

	# # project to sphere to get cartesian coordinates
	dxmid, dymid = dx+dw/2-c[0], dy+dh/2-c[1]
	dxmin, dxmax = dx-c[0], dx+dw-c[0]
	dymin, dymax = dy-c[1], dy+dh-c[1]

	phi_adj, theta_adj = fish2polar(dxmid, dymid, f_d)

	phi_1, theta_1 = fish2polar(dxmin, dymin, f_d)
	phi_2, theta_2 = fish2polar(dxmin, dymax, f_d)
	phi_3, theta_3 = fish2polar(dxmax, dymin, f_d)
	phi_4, theta_4 = fish2polar(dxmax, dymax, f_d)
	rx_1, ry_1 = latlon2persp(theta_1, phi_1, f_u, lat0=theta_adj, lon0=phi_adj)
	rx_2, ry_2 = latlon2persp(theta_2, phi_2, f_u, lat0=theta_adj, lon0=phi_adj)
	rx_3, ry_3 = latlon2persp(theta_3, phi_3, f_u, lat0=theta_adj, lon0=phi_adj)
	rx_4, ry_4 = latlon2persp(theta_4, phi_4, f_u, lat0=theta_adj, lon0=phi_adj)

	rxmin, rymin = np.min([rx_1, rx_2, rx_3, rx_4]), np.min([ry_1, ry_2, ry_3, ry_4])
	rxmax, rymax = np.max([rx_1, rx_2, rx_3, rx_4]), np.max([ry_1, ry_2, ry_3, ry_4])

	width, height = rxmax-rxmin, rymax-rymin

	return [int(width), int(height), phi_adj, theta_adj]

def roi2persp3(roi, f_u=500., f_d=774., src_size=[2448.,2048.]):
	d_width = src_size[0]
	d_height = src_size[1]

	[dx, dy, dw, dh] = roi

	phi_adj, theta_adj = fish2polar(dx+dw/2, dy+dh/2, f_d, src_size)
	import pdb; pdb.set_trace()

	# # project to sphere to get cartesian coordinates
	dxmid, dymid = dx+dw/2, dy+dh/2
	dxmin, dxmax = dx, dx+dw
	dymin, dymax = dy, dy+dh

	phi_1, theta_1 = fish2polar(dxmin, dymin, f_d, src_size)
	phi_2, theta_2 = fish2polar(dxmin, dymax, f_d, src_size)
	phi_3, theta_3 = fish2polar(dxmax, dymin, f_d, src_size)
	phi_4, theta_4 = fish2polar(dxmax, dymax, f_d, src_size)
	rx_1, ry_1 = polar2perspnorm(phi_1-phi_adj, theta_1-theta_adj, f_u)
	rx_2, ry_2 = polar2perspnorm(phi_2-phi_adj, theta_2-theta_adj, f_u)
	rx_3, ry_3 = polar2perspnorm(phi_3-phi_adj, theta_3-theta_adj, f_u)
	rx_4, ry_4 = polar2perspnorm(phi_4-phi_adj, theta_4-theta_adj, f_u)

	rxmin, rymin = np.min([rx_1, rx_2, rx_3, rx_4]), np.min([ry_1, ry_2, ry_3, ry_4])
	rxmax, rymax = np.max([rx_1, rx_2, rx_3, rx_4]), np.max([ry_1, ry_2, ry_3, ry_4])

	width, height = rxmax-rxmin, rymax-rymin

	return [int(width), int(height), phi_adj, theta_adj]


def roi2persp2(roi, f_u=500., f_d=774., src_size=[2448.,2048.]):
	d_width = src_size[0]
	d_height = src_size[1]

	[dx, dy, dw, dh] = roi

	phi_adj, theta_adj = fish2polar(dx+dw/2-d_width/2, dy+dh/2-d_height/2)

	# # project to sphere to get cartesian coordinates
	dxmid, dymid = dx+dw/2, dy+dh/2
	# R_d_mid = np.sqrt(dxmid**2+dymid**2)
	# theta_mid = R_d_mid/f_d
	# dzmid = R_d_mid/np.tan(theta_mid)
	# drmid = np.sqrt(dxmid**2+dymid**2+dzmid**2)

	# # get polar angles from cartesian coordinates
	# phi_adj = (np.pi/2.) - np.arctan2(dzmid,dxmid)
	# theta_adj = (np.pi/2.) - np.arccos(dymid/drmid)
	# dxmin, dxmax = dx-d_width/2, dx+dw-d_width/2
	# dymin, dymax = dy-d_height/2, dy+dh-d_height/2
	dxmin, dxmax = -dw/2, dw/2
	dymin, dymax = -dh/2, dh/2

	phi_1, theta_1 = fish2polar(dxmin, dymin)
	phi_2, theta_2 = fish2polar(dxmin, dymax)
	phi_3, theta_3 = fish2polar(dxmax, dymin)
	phi_4, theta_4 = fish2polar(dxmax, dymax)
	rx_1, ry_1 = polar2persp(phi_1, theta_1, 500.)
	rx_2, ry_2 = polar2persp(phi_2, theta_2, 500.)
	rx_3, ry_3 = polar2persp(phi_3, theta_3, 500.)
	rx_4, ry_4 = polar2persp(phi_4, theta_4, 500.)

	rxmin, rymin = np.min([rx_1, rx_2, rx_3, rx_4]), np.min([ry_1, ry_2, ry_3, ry_4])
	rxmax, rymax = np.max([rx_1, rx_2, rx_3, rx_4]), np.max([ry_1, ry_2, ry_3, ry_4])

	width, height = rxmax-rxmin, rymax-rymin

	return [int(width), int(height), phi_adj*180./np.pi, theta_adj*180./np.pi]


def roi2persp(roi, f_u=500., f_d=774., c=[1224.,1024.]):
	[dx, dy, dw, dh] = roi
	dxmid, dymid = dx+dw/2-c[0], dy+dh/2-c[1]

	phi_adj = dxmid/f_d
	theta_adj = dymid/f_d

	dxmin, dxmax = roi[0]-c[0], roi[0]+roi[2]-c[0]
	dymin, dymax = roi[1]-c[1], roi[1]+roi[3]-c[1]
	dxmin, dxmax = dxmin-dxmid, dxmax-dxmid
	dymin, dymax = dymin-dymid, dymax-dymid

	R_d_min = np.sqrt(dxmin**2+dymin**2)
	theta_min = R_d_min/f_d
	R_d_max = np.sqrt(dxmax**2+dymax**2)
	theta_max = R_d_max/f_d

	R_u_min = f_u*np.tan(theta_min)
	R_u_max = f_u*np.tan(theta_max)
	rxmin, rymin = R_u_min*dxmin/R_d_min, R_u_min*dymin/R_d_min
	rxmax, rymax = R_u_max*dxmax/R_d_max, R_u_max*dymax/R_d_max
	width, height = rxmax-rxmin, rymax-rymin
	
	# rxmid, rymid = rxmin+width/2, rymin+height/2

	# R_u_mid = np.sqrt(rxmid**2+rymid**2)
	# theta_mid = np.arctan(R_u_mid/f_u)
	# R_d_mid = f_d*theta_mid
	# dxmid, dymid = R_d_mid*rxmid/R_u_mid, R_d_mid*rymid/R_u_mid

	# phi_adj = dxmid*d_fov/d_width
	# theta_adj = dymid*d_fov/d_height

	return [int(width), int(height), phi_adj, theta_adj]

if __name__ == "__main__":
	# source = mpimg.imread("0.png")
	# source = cv2.imread("0.png")
	source = skimage.img_as_float(imread("0.png"))
	source = source[:,:,::-1]
	scale = 1000./np.min(source.shape[:2])
	source_scaled = cv2.resize(source,None,fx=scale,fy=scale)
	# persp2latlon(30, 0, 250.)
	# import pdb; pdb.set_trace()
	while True:
		fromCenter = False
		showCrosshair = False
		r = cv2.selectROI("Object Selector", source_scaled, fromCenter, showCrosshair)
		r = np.array(r)/scale

		[w, h, dp, dt] = roi2persp4(r, f_u=250)

		# import pdb; pdb.set_trace()

		# [w,h,dp,dt] = [500,500,0,40*np.pi/180.]


		out = np.zeros((h,w,3))
		for x in range(w):
			for y in range(h):
				# sp = fish2persp3(x, y, f_u=500., vp=[dp,dt], dst_size=[w,h])
				sp = fish2persp_tmp(x, y, f_u=250., vp=[dp,dt], dst_size=[w,h])
				# sp = fish2persp2(x, y, f_u=250., vp=[dp,dt], dst_size=[w,h])
				# sp = fish2persp(x, y, f_u=250., f_d=750., vp=[dp,dt], dst_size=[w,h])
				if None in sp:
					out[y,x,:] = [0,0,0]
				else:
					out[y,x,:] = source[sp[1],sp[0],:]

		cv2.imshow("Perspective", out)
		cv2.waitKey(0)
		# plot = plt.imshow(out)
		# plt.show()
		# plt.waitforbuttonpress()


	    # FOV = 3.141592654*u_fovFactor
	    # width = u_width
	    # height = u_height

	    # # Polar angles
	    # theta = 2.0 * 3.14159265 * (pfish['x'] / width - 0.5)
	    # phi = 3.14159265 * (pfish['y'] / height - 0.5)
	    # r = height * phi / FOV 

	    # # Vector in 3D space
	    # psph['x'] = cos(phi) * sin(theta)
	    # psph['y'] = cos(phi) * cos(theta)
	    # psph['z'] = sin(phi)
	    
	    # # Calculate fisheye angle and radius
	    # theta = atan(psph['z'], psph['x'])
	    # phi = atan(sqrt(psph['x']*psph['x'] + psph['z']*psph['z']), psph['y'])
	    # r = height * phi / FOV 

	    # # Pixel in fisheye space
	    # ret = {}
	    # ret['x'] = 0.5 * width + r * cos(theta)
	    # ret['y'] = 0.5 * height + r * sin(theta)
	    
	    # return ret