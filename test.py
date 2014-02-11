import numpy as np
import raytracer as rt

"""Planoconvex test and data collection suite"""

r = Ray([0.0,0.0,0.0],[0.0,0.0,1.0])
s = SphericalRefraction(100.0, .02, 1.0, 1.5168, 49.74937)
p = SphericalRefraction(105.0, 0.0, 1.5168, 1.0, 49.74937)
o = OutputPlane(200.0)
Beams, RMS, DiffractionLimits = [], [], []
x = 100.0 #Total number of rays in beam

n = .5+np.sqrt(0.25 + x/np.pi) #Number of radii increments in beam

#Create and propagate beams of different radii
for t in np.linspace(0.1,5.0,10):
	Beams.append(CollimatedBeam(r, t, n))
for elem in Beams:
	elem.trace2(s,p,o)
	RMS.append(elem.rms)

def optimiselens(self, c1, c2, z):
	lens1 = SphericalRefraction(100.0, c1, 1.0, 1.5168, 49.74937)
	lens2 = SphericalRefraction(105.0, c2, 1.5168, 1.0, 49.74937)
	outputplane = OutputPlane(z)

	