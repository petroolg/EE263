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
        return line_pixel_length(d, np.pi / 2 - theta, n)

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
    jy = np.ceil(y)
    dy = np.remainder(y + n, 1)

    for jx in range(n):
        dynext = dy + st / ct
        if dynext < 1:
            if 1 <= jy <= n:
                L[int(n - jy), int(jx)] = 1 / ct
            dy = dynext
        else:
            if 1 <= jy <= n:
                L[int(n - jy), int(jx)] = (1 - dy) / st
            if 1 <= jy + 1 <= n:
                L[int(n - (jy + 1)), int(jx)] = (dynext - 1) / st
            dy = dynext - 1
            jy = jy + 1

    return L.reshape((-1,1), order='F').T

L = []

for n in range(N):
    L.append(line_pixel_length(lines_d[n], lines_theta[n], n_pixels))




L = np.array(L).reshape((N, n_pixels**2))
x = np.linalg.pinv(L).dot(y)

plt.imshow(x.reshape((n_pixels,n_pixels), order='F'))
plt.colorbar()
plt.show()
