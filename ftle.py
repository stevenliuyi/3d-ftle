import sys
import time
import numpy as np
from scipy import ndimage
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
#from mayavi import mlab

# -----------------------------------------------------------------------------
# arguments: t_start, direction, scale, delta, inttime, method

# -----------------------------------------------------------------------------
# setup data grid
def setup_grid(nx1, ny1, nz1, nx2, ny2, nz2):
    xx = np.load(data_folder + 'x.npy', mmap_mode='r')
    yy = np.load(data_folder + 'y.npy', mmap_mode='r')
    zz = np.load(data_folder + 'z.npy', mmap_mode='r')

    x = xx[nx1:(nx2+1), 0, 0]
    y = yy[0, ny1:(ny2+1), 0]
    z = zz[0, 0, nz1:(nz2+1)]

    return (x, y, z)

# -----------------------------------------------------------------------------
# setup trajectory grid
def setup_traj_grid(ox, oy, oz):
    x = np.linspace(x_data[0], x_data[-1], ox)
    z = np.linspace(z_data[0], z_data[-1], oz)
    y = y_data

    return (x, y, z)

# -----------------------------------------------------------------------------
# read u, v, w data at given time
def read_data(t):
    ufile = np.load(data_folder + 'u_' + str(t) + '.npy', mmap_mode='r')
    vfile = np.load(data_folder + 'v_' + str(t) + '.npy', mmap_mode='r')
    wfile = np.load(data_folder + 'w_' + str(t) + '.npy', mmap_mode='r')

    uu = ufile[nx1:(nx2+1), ny1:(ny2+1), nz1:(nz2+1)]
    vv = vfile[nx1:(nx2+1), ny1:(ny2+1), nz1:(nz2+1)]
    ww = wfile[nx1:(nx2+1), ny1:(ny2+1), nz1:(nz2+1)]
    return (uu, vv, ww)

# -----------------------------------------------------------------------------
# linear interpolation
def interp(x1, x2, y1, y2, xm):
    ym = y2 - (y2-y1)/(x2-x1) * (x2-xm)
    return ym

# -----------------------------------------------------------------------------
# interpolate velocity in space at all points
def interpn_vel(velt, trajx, trajy, trajz):
    points = np.transpose(np.array((trajx, trajy, trajz)), axes=[1,2,3,0])
    vx_in = interpn((x_data, y_data, z_data), velt[:,:,:,0], points, \
            method='linear', bounds_error=False)
    vy_in = interpn((x_data, y_data, z_data), velt[:,:,:,1], points, \
            method='linear', bounds_error=False)
    vz_in = interpn((x_data, y_data, z_data), velt[:,:,:,2], points, \
            method='linear', bounds_error=False)
    vx_out = interpn((x_data, y_data, z_data), velt[:,:,:,0], points, \
            method='nearest', bounds_error=False, fill_value=None)
    vy_out = interpn((x_data, y_data, z_data), velt[:,:,:,1], points, \
            method='nearest', bounds_error=False, fill_value=None)
    vz_out = interpn((x_data, y_data, z_data), velt[:,:,:,2], points, \
            method='nearest', bounds_error=False, fill_value=None)
    vx_out_pts = np.isnan(vx_in)
    vy_out_pts = np.isnan(vy_in)
    vz_out_pts = np.isnan(vz_in)

    vx = vx_out * vx_out_pts + np.nan_to_num(vx_in) * np.invert(vx_out_pts)
    vy = vy_out * vy_out_pts + np.nan_to_num(vy_in) * np.invert(vy_out_pts)
    vz = vz_out * vz_out_pts + np.nan_to_num(vz_in) * np.invert(vz_out_pts)

    return vx, vy, vz

# -----------------------------------------------------------------------------
# interpolate velocity in space
def interp_vel(velt, xij, yij, zij):
    xp = max(min(np.searchsorted(x_data, xij)-1, nx-2), 0)
    yp = max(min(np.searchsorted(y_data, yij)-1, ny-2), 0)
    zp = max(min(np.searchsorted(z_data, zij)-1, nz-2), 0)

    xm = interp(x_data[xp], x_data[xp+1], xp, xp+1, xij)
    ym = interp(y_data[yp], y_data[yp+1], yp, yp+1, yij)
    zm = interp(z_data[zp], z_data[zp+1], zp, zp+1, zij)

    um = ndimage.map_coordinates(velt[:,:,:,0], [[xm],[ym],[zm]], \
            order=1, mode='nearest')
    vm = ndimage.map_coordinates(velt[:,:,:,1], [[xm],[ym],[zm]], \
            order=1, mode='nearest')
    wm = ndimage.map_coordinates(velt[:,:,:,2], [[xm],[ym],[zm]], \
            order=1, mode='nearest')

    return (um[0],vm[0],wm[0])

# -----------------------------------------------------------------------------
# interpolate velocity in time
def set_vel(tt):
    velt = np.zeros((nx,ny,nz,3))
    velt[:,:,:,0] = interp(delt*t_jump, \
            delt*(t_jump+direction), u1, u, tt*delta*direction)
    velt[:,:,:,1] = interp(delt*t_jump, \
            delt*(t_jump+direction), v1, v, tt*delta*direction)
    velt[:,:,:,2] = interp(delt*t_jump, \
            delt*(t_jump+direction), w1, w, tt*delta*direction)
    #print delt*t_jump, tt*delta*direction, delt*(t_jump+direction)
    # for i in range(0,nx):
    #     for j in range(0,ny):
    #         for k in range(0,nz):
    #             velt[i,j,k,0] = interp(delt*t_jump, \
    #                     delt*(t_jump+direction), u1[i,j,k], u[i,j,k], tt*delta*direction)
    #             velt[i,j,k,1] = interp(delt*t_jump, \
    #                     delt*(t_jump+direction), v1[i,j,k], v[i,j,k], tt*delta*direction)
    #             velt[i,j,k,2] = interp(delt*t_jump, \
    #                     delt*(t_jump+direction), w1[i,j,k], w[i,j,k], tt*delta*direction)
    return velt

# -----------------------------------------------------------------------------
# update trajectory
def update_traj(method):
    global traj_x, traj_y, traj_z

    if (method == 1):
        (vx, vy, vz) = interpn_vel(velt, traj_x, traj_y, traj_z)
        traj_x += vx*delta*direction
        traj_y += vy*delta*direction
        traj_z += vz*delta*direction
    elif (method == 2):
        (vx, vy, vz) = interpn_vel(velt, traj_x, traj_y, traj_z)
        k1x = vx*delta*direction
        k1y = vy*delta*direction
        k1z = vz*delta*direction
        (vx1, vy1, vz1) = interpn_vel((velt+ velt2)/2, traj_x+k1x/2, \
                traj_y+k1y/2, traj_z+k1z/2)
        k2x = vx1*delta*direction
        k2y = vy1*delta*direction
        k2z = vz1*delta*direction
        traj_x += k2x
        traj_y += k2y
        traj_z += k2z
    elif (method == 3):
        (vx, vy, vz) = interpn_vel(velt, traj_x, traj_y, traj_z)
        k1x = vx*delta*direction
        k1y = vy*delta*direction
        k1z = vz*delta*direction
        (vx1, vy1, vz1) = interpn_vel((velt+ velt2)/2, traj_x+k1x/2, \
                traj_y+k1y/2, traj_z+k1z/2)
        k2x = vx1*delta*direction
        k2y = vy1*delta*direction
        k2z = vz1*delta*direction
        (vx2, vy2, vz2) = interpn_vel((velt+ velt2)/2, traj_x+k2x/2, \
                traj_y+k2y/2, traj_z+k2z/2)
        k3x = vx2*delta*direction
        k3y = vy2*delta*direction
        k3z = vz2*delta*direction
        (vx3, vy3, vz3) = interpn_vel(velt2, traj_x+k3x, \
                traj_y+k3y, traj_z+k3z)
        k4x = vx3*delta*direction
        k4y = vy3*delta*direction
        k4z = vz3*delta*direction
        traj_x += k1x/6 + k2x/3 + k3x/3 + k4x/6
        traj_y += k1y/6 + k2y/3 + k3y/3 + k4y/6
        traj_z += k1z/6 + k2z/3 + k3z/3 + k4z/6

    # for i in range(0,ox):
    #     for j in range(0,oy):
    #         for k in range(0,oz):
    #             (vx, vy, vz) = interp_vel(velt, \
    #                     traj_x[i,j,k], traj_y[i,j,k], traj_z[i,j,k])
    #             if (method == 1):
    #                 traj_x[i,j,k] += vx*delta*direction
    #                 traj_y[i,j,k] += vy*delta*direction
    #                 traj_z[i,j,k] += vz*delta*direction
    #             elif (method == 2):
    #                 k1x = vx*delta*direction
    #                 k1y = vy*delta*direction
    #                 k1z = vz*delta*direction
    #                 (vx1, vy1, vz1) = interp_vel(velt,  traj_x[i,j,k]+k1x/2, \
    #                         traj_y[i,j,k]+k1y/2, traj_z[i,j,k]+k1z/2)
    #                 (vx2, vy2, vz2) = interp_vel(velt2, traj_x[i,j,k]+k1x/2, \
    #                         traj_y[i,j,k]+k1y/2, traj_z[i,j,k]+k1z/2)
    #                 k2x = (vx1+vx2)/2*delta*direction
    #                 k2y = (vy1+vy2)/2*delta*direction
    #                 k2z = (vz1+vz2)/2*delta*direction
    #                 traj_x[i,j,k] += k2x
    #                 traj_y[i,j,k] += k2y
    #                 traj_z[i,j,k] += k2z
    #             elif (method == 3):
    #                 k1x = vx*delta*direction
    #                 k1y = vy*delta*direction
    #                 k1z = vz*delta*direction
    #                 (vx1, vy1, vz1) = interp_vel(velt,  traj_x[i,j,k]+k1x/2, \
    #                         traj_y[i,j,k]+k1y/2, traj_z[i,j,k]+k1z/2)
    #                 (vx2, vy2, vz2) = interp_vel(velt2, traj_x[i,j,k]+k1x/2, \
    #                         traj_y[i,j,k]+k1y/2, traj_z[i,j,k]+k1z/2)
    #                 k2x = (vx1+vx2)/2*delta*direction
    #                 k2y = (vy1+vy2)/2*delta*direction
    #                 k2z = (vz1+vz2)/2*delta*direction
    #                 (vx3, vy3, vz3) = interp_vel(velt,  traj_x[i,j,k]+k2x/2, \
    #                         traj_y[i,j,k]+k2y/2, traj_z[i,j,k]+k2z/2)
    #                 (vx4, vy4, vz4) = interp_vel(velt2, traj_x[i,j,k]+k2x/2, \
    #                         traj_y[i,j,k]+k2y/2, traj_z[i,j,k]+k2z/2)
    #                 k3x = (vx3+vx4)/2*delta*direction
    #                 k3y = (vy3+vy4)/2*delta*direction
    #                 k3z = (vz3+vz4)/2*delta*direction
    #                 (vx5, vy5, vz5) = interp_vel(velt2, traj_x[i,j,k]+k3x, \
    #                         traj_y[i,j,k]+k3y, traj_z[i,j,k]+k3z)
    #                 k4x = vx5*delta*direction
    #                 k4y = vy5*delta*direction
    #                 k4z = vz5*delta*direction
    #                 traj_x[i,j,k] += k1x/6 + k2x/3 + k3x/3 + k4x/6
    #                 traj_y[i,j,k] += k1y/6 + k2y/3 + k3y/3 + k4y/6
    #                 traj_z[i,j,k] += k1z/6 + k2z/3 + k3z/3 + k4z/6
    return

# -----------------------------------------------------------------------------
# calculate FTLE field
def calc_ftle(num):
    global ftle
    for i in range(0,ox):
        for j in range(0,oy):
            for k in range(0,oz):
                # central differencing except end points
                if (i==0):
                    xa = np.array([traj_x[i  ,j,k], traj_y[i  ,j,k], traj_z[i  ,j,k]])
                    xc = np.array([traj_x[i+1,j,k], traj_y[i+1,j,k], traj_z[i+1,j,k]])
                    xoa = x[i  ]
                    xoc = x[i+1]
                elif (i==ox-1):
                    xa = np.array([traj_x[i-1,j,k], traj_y[i-1,j,k], traj_z[i-1,j,k]])
                    xc = np.array([traj_x[i  ,j,k], traj_y[i  ,j,k], traj_z[i  ,j,k]])
                    xoa = x[i-1]
                    xoc = x[i  ]
                else:
                    xa = np.array([traj_x[i-1,j,k], traj_y[i-1,j,k], traj_z[i-1,j,k]])
                    xc = np.array([traj_x[i+1,j,k], traj_y[i+1,j,k], traj_z[i+1,j,k]])
                    xoa = x[i-1]
                    xoc = x[i+1]

                if (j==0):
                    ya = np.array([traj_x[i,j  ,k], traj_y[i,j  ,k], traj_z[i,j  ,k]])
                    yc = np.array([traj_x[i,j+1,k], traj_y[i,j+1,k], traj_z[i,j+1,k]])
                    yoa = y[j  ]
                    yoc = y[j+1]
                elif (j==oy-1):
                    ya = np.array([traj_x[i,j-1,k], traj_y[i,j-1,k], traj_z[i,j-1,k]])
                    yc = np.array([traj_x[i,j  ,k], traj_y[i,j  ,k], traj_z[i,j  ,k]])
                    yoa = y[j-1]
                    yoc = y[j  ]
                else:
                    ya = np.array([traj_x[i,j-1,k], traj_y[i,j-1,k], traj_z[i,j-1,k]])
                    yc = np.array([traj_x[i,j+1,k], traj_y[i,j+1,k], traj_z[i,j+1,k]])
                    yoa = y[j-1]
                    yoc = y[j+1]
    
                if (k==0):
                    za = np.array([traj_x[i,j,k  ], traj_y[i,j,k  ], traj_z[i,j,k  ]])
                    zc = np.array([traj_x[i,j,k+1], traj_y[i,j,k+1], traj_z[i,j,k+1]])
                    zoa = z[k  ]
                    zoc = z[k+1]
                elif (k==oz-1):
                    za = np.array([traj_x[i,j,k-1], traj_y[i,j,k-1], traj_z[i,j,k-1]])
                    zc = np.array([traj_x[i,j,k  ], traj_y[i,j,k  ], traj_z[i,j,k  ]])
                    zoa = z[k-1]
                    zoc = z[k  ]
                else:
                    za = np.array([traj_x[i,j,k-1], traj_y[i,j,k-1], traj_z[i,j,k-1]])
                    zc = np.array([traj_x[i,j,k+1], traj_y[i,j,k+1], traj_z[i,j,k+1]])
                    zoa = z[k-1]
                    zoc = z[k+1]
                lambdas = eigs(xc, xa, yc, ya, zc, za, \
                        xoc, xoa, yoc, yoa, zoc, zoa)

                if (lambdas=='nan'):
                    ftle[i,j,k] = float('nan')
                else:
                    ftle[i,j,k] = .5*np.log(max(lambdas))/(num*delta)
    return

# -----------------------------------------------------------------------------
# calculate eigenvalues of [dx/dx0]^T[dx/dx0]
def eigs(xd1, xd2, yd1, yd2, zd1, zd2, x01, x02, y01, y02, z01, z02):
    ftlemat = np.zeros((3,3))
    ftlemat[0,0] = (xd1[0] - xd2[0]) / (x01 - x02)
    ftlemat[1,0] = (xd1[1] - xd2[1]) / (x01 - x02)
    ftlemat[2,0] = (xd1[2] - xd2[2]) / (x01 - x02)
    ftlemat[0,1] = (yd1[0] - yd2[0]) / (y01 - y02)
    ftlemat[1,1] = (yd1[1] - yd2[1]) / (y01 - y02)
    ftlemat[2,1] = (yd1[2] - yd2[2]) / (y01 - y02)
    ftlemat[0,2] = (zd1[0] - zd2[0]) / (z01 - z02)
    ftlemat[1,2] = (zd1[1] - zd2[1]) / (z01 - z02)
    ftlemat[2,2] = (zd1[2] - zd2[2]) / (z01 - z02)

    if (True in np.isnan(ftlemat)): return 'nan'
    ftlemat = np.dot(ftlemat.transpose(), ftlemat)
    w, v = np.linalg.eig(ftlemat)

    return w

# -----------------------------------------------------------------------------
# identify LCSs
def get_lcs(percent):
    global lcs
    ftle_max = np.nanmax(ftle)
    for i in range(0,ox):
        for j in range(0,oy):
            if (ftle[i][j]>percent*ftle_max):
                lcs[i][j] = 1.
            else:
                lcs[i][j] = 0.
    return 

# -----------------------------------------------------------------------------
# write FTLE data to files
def write_ftle():
    np.save('ftle.npy', ftle)
    return

# -----------------------------------------------------------------------------
# write trajectory data to files
def write_traj():
    np.save('trajx.npy', traj_x)
    np.save('trajy.npy', traj_y)
    np.save('trajz.npy', traj_z)
    return

# -----------------------------------------------------------------------------
# grid info
# 2001 * 400 * 513
# cut y
# (nx1, nx2) = (0, 2000)
# (ny1, ny2) = (198, 202)
# (nz1, nz2) = (0, 512)

# cut z
(nx1, nx2) = (0, 2000)
(ny1, ny2) = (0, 399)
(nz1, nz2) = (254, 258)

nx = nx2 - nx1 + 1
ny = ny2 - ny1 + 1
nz = nz2 - nz1 + 1

# scale = (float(sys.argv[3]), float(sys.argv[3]), float(sys.argv[3]))
scale = (1, 1, 1)

# integration info
# inttime = float(sys.argv[5])
# delta = float(sys.argv[4])
# direction = int(sys.argv[2])
# method = int(sys.argv[6])

inttime = 0.48
delt = 0.06     # time difference between adjacent data files
delta = 0.003    # interation time step
direction = -1
method = 1

# data info
data_folder = '/Volumes/TOSHIBA EXT/Boundary/'

# -----------------------------------------------------------------------------
# setup data grid
(x_data,y_data,z_data) = setup_grid(nx1, ny1, nz1, nx2, ny2, nz2)

# setup trajectory grid
ox = int(nx * scale[0])
oy = int(ny * scale[1])
oz = int(nz * scale[2])
(x, y, z) = setup_traj_grid(ox, oy, oz)

#t_start = int(sys.argv[1])
t_start = 861

t_jump = 0
(u1,v1,w1) = read_data(t_start+t_jump)

(u,v,w) = read_data(t_start+t_jump+direction)

# initialize trajectory data
traj_x, traj_y, traj_z = np.meshgrid(x, y, z, indexing='ij')

# initialize FTLE field
ftle = np.zeros((ox,oy,oz))

#show_traj(5,5,'r')
start_time = time.time()

dir_str = 'pos' if direction==1 else 'neg'

# start FTLE integration
for t in range(0, int(inttime/delta)):
    print 'time: ' + str(t_start) + ' (' + dir_str + \
          '), integration time: ' + str(t+1) + \
          '/' + str(int(inttime/delta))
    
    if (t == 0):
        if (abs(t*delta) > abs(delt*(t_jump+direction))):
            t_jump += direction
            (u1,v1,w1) = (u,v,w)
            (u,v,w) = read_data(t_start+t_jump+direction)
        # interpolate velocity in time
        velt = set_vel(t)
    else:
        velt = velt2

    if (abs((t+1)*delta) > abs(delt*(t_jump+direction))):
        t_jump += direction
        (u1,v1,w1) = (u,v,w)
        (u,v,w) = read_data(t_start+t_jump+direction)
    velt2 = set_vel(t+1)

    # use updated velocity to compute trajectory 
    update_traj(method)

    print 'calculation time: ' + str(time.time()-start_time)

# cacluate FTLE field
calc_ftle(t)

write_traj()
write_ftle()
print dir_str + ' FTLE field at t=' + str(t_start) + \
      ' has been calculated'
