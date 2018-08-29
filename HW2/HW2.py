# Homework for Stanford Course
import numpy as np
from matplotlib import pyplot as plt


# This defines 20 discrete wavelengths
wavelength = np.linspace(380, 760, 20) #(nm)

# Define the vectors of cone responsiveness
# Cone sensitivies for wavelengths in the visible range.  This data is=20
#  based on a graph on page 93 in Wandell's "Foundations of Vision"=20
L_cone_log = np.array([-.9,-1.1,-1,-.8,-.7,-.6,-.5,-.4,-.3,-.25,-.3,-.4,-.8,-1.2,-1.6,-1.9,-2.4,-2.8,-3.6,-4])
M_cone_log = np.array([-.8,-.85,-.85,-.6,-.5,-.4,-.3,-.2,-.1,-.15,-.3,-.7,-1.2,-1.7,-2.2,-2.9,-3.5,-4.2,-4.8,-5.5])
S_cone_log = np.array([-.4,-.3,-.2,-.1,-.3,-.6,-1.2,-1.8,-2.5,-3.2,-4.1,-5,-6,-7,-8,-9,-10,-10.5,-11,-11.5])

# Raise,everything to the 10 power to get the actual responsitivies. =20
# These vectors contain the coefficients l_i, m_i, and s_i described in =
the=20
# problem statement from shortest to longest wavelength.=20
L_coefficients = 10**(L_cone_log)
M_coefficients = 10**(M_cone_log)
S_coefficients = 10**(S_cone_log)

# Plot the cone responsiveness.=20
plt.figure()
plt.plot(wavelength,L_cone_log,'-*')
plt.plot(wavelength,M_cone_log,'--x')
plt.plot(wavelength,S_cone_log,'-o')
plt.xlabel('Light Wavelength (nm)')
plt.ylabel('Log Relative Sensitivity')
plt.legend(['L cones', 'M cones', 'S cones'])
plt.title('Approximate Human Cone Sensitivities')



# Spectral Power Distribution of phosphors from shortest to longest wavelength
B_phosphor = np.array([30,35,45,75,90,100,90,75,45,35,30,26,23,21,20,19,18,17,16,15])
G_phosphor = np.array([21,23,26,30,35,45,75,90,100,90,75,45,35,30,26,23,21,20, 19,18])
R_phosphor = np.array([15,16,17,18,19,21,23,26,30,35,45,75,90,100,90,75,45,35,30,26])

#Plot the phosphors
plt.figure()
plt.plot(wavelength, B_phosphor,'-*')
plt.plot(wavelength, G_phosphor,'--x')
plt.plot(wavelength, R_phosphor,'-o')
plt.xlabel('Light wavelength (nm)')
plt.xlabel('Spectral Power Distribution of Phosphors')
plt.legend(['B phosphor', 'G phosphor', 'R phosphor'])


# This is the spectral power distribution of the test light from the=20
# shortest wavelength to the longest.=20
test_light = np.array([ 58.2792,42.3496,51.5512,33.3951,43.2907,22.5950,57.9807,
               76.0365,52.9823,64.0526,20.9069,37.9818,78.3329,68.0846,
               46.1095,56.7829,79.4211,5.9183,60.2869,5.0269])

#Plot the test light
plt.figure()
plt.plot(wavelength, test_light, '-*')
plt.grid()
plt.xlabel('Light wavelength (nm)')
plt.xlabel('Spectral Power Distribution of Test Light')
plt.title('Test light')

# Define approximate spectrums for sunlight and a tungsten bulb. =20
# The powers are in order of increasing wavelength
tungsten = np.array([20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210])
sunlight= np.array([40,70,100,160,240,220,180,160,180,180,160,140,140,140,140,130,120,116,110,110])

#Plot,the specturms
plt.figure()
plt.plot(wavelength,tungsten,'-*')
plt.plot(wavelength,sunlight,'-o')
plt.xlabel('Light wavelength (nm)')
plt.xlabel('Spectral Power Distributions')
plt.legend(['Tungsten','Sunlight'])



# 3.2 c)

X = np.vstack((L_coefficients, M_coefficients, S_coefficients))
colors = np.vstack((R_phosphor, G_phosphor, B_phosphor))

a = np.linalg.inv(X.dot(colors.T)).dot(X).dot(test_light[np.newaxis].T)
pow_dens = colors.T.dot(a)

plt.figure()
plt.plot(wavelength, test_light, '-*')
plt.plot(wavelength,pow_dens, '-o')
plt.title('Spectrum of test light vs simulated')

print('3.2.c colors of two dens. spectra are the same')
print(X.dot(test_light))
print((X.dot(pow_dens)).T[0])


# 3.2 d)

np.random.seed(1)
r1 = np.random.random((20,1))

ar2 = np.linalg.inv(X
                    .dot(np.diag(tungsten))
                    .dot(colors.T))\
    .dot(X).dot(np.diag(tungsten))\
    .dot(r1)
r2 = colors.T.dot(ar2)

# control the results
print('3.2.b colors of two dens. spectra are the same under tungsten bulb')
print(X.dot(np.diag(tungsten)).dot(r1))
print(X.dot(np.diag(tungsten)).dot(r2))

print('3.2.b colors of two dens. spectra are the different under sun')
print(X.dot(np.diag(sunlight)).dot(r1))
print(X.dot(np.diag(sunlight)).dot(r2))

plt.figure()
plt.plot(wavelength, np.diag(tungsten).dot(r1), '-*')
plt.plot(wavelength, np.diag(tungsten).dot(r2), '-o')
plt.plot(wavelength, np.diag(sunlight).dot(r1), '-*')
plt.plot(wavelength, np.diag(sunlight).dot(r2), '-o')
plt.title('Spectrum of test light vs simulated')
plt.legend(['tungsten o1','tungsten o2','sunlight o1','sunlight o2'])


plt.show()