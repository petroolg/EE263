import numpy as np
from matplotlib import pyplot as plt

n_pixels = 30

N = 1225

y = np.loadtxt('HW3/y')
lines_d = np.loadtxt('HW3/lines_d')
lines_theta = np.loadtxt('HW3/lines_theta')

def line_pixel_length(d, theta, n):
    # for angle in [np.pi/4,3*np.pi/4],
    # flip along diagonal (transpose) and call recursively
    if np.pi / 4 < theta < 3 / 4 * np.pi:
        return line_pixel_length(d, np.pi / 2 - theta, n).T

        # for angle in [3*np.pi/4,np.pi],
        # redefine line to go in opposite direction
    if theta > np.pi / 2:
        d = -d
        theta = theta - np.pi

        # for angle in [-np.pi/4,0],
        # flip along x-axis (up/down) and call recursively
    if theta < 0:
        return np.flipud(line_pixel_length(-d, -theta, n))

    if theta > np.pi / 2 or theta < 0:
        print('invalid angle')
        return

    L = np.zeros((n, n))

    ct = np.cos(theta)
    st = np.sin(theta)

    x0 = n / 2 - d * st
    y0 = n / 2 + d * ct

    y = y0 - x0 * st / ct
    jy = int(np.ceil(y))
    dy = np.remainder(y + n, 1)

    for jx in range(n):
        dynext = dy + st / ct
        if dynext < 1:
            if 1 <= jy <= n:
                L[n - jy, jx] = 1 / ct
            dy = dynext
        else:
            if 1 <= jy <= n:
                L[n - jy, jx] = (1 - dy) / st
            if 1 <= jy + 1 <= n:
                L[n - (jy + 1), jx] = (dynext - 1) / st
            dy = dynext - 1
            jy = jy + 1

    return L


#################
# Solution to HW3
#################

L = []

for n in range(N):
    line = line_pixel_length(lines_d[n], lines_theta[n], n_pixels)
    L.append(line.ravel())


x = np.linalg.pinv(L).dot(y)

plt.imshow(x.reshape((n_pixels, n_pixels)))
plt.colorbar()
plt.show()
