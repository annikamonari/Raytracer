import numpy as np
import raytracer as rt
import random

"""Planoconvex test and data collection suite

r = rt.Ray([0.0,0.0,0.0],[0.0,0.0,1.0])
s = rt.SphericalRefraction(100.0, .02, 1.0, 1.5168, 49.74937)
p = rt.SphericalRefraction(105.0, 0.0, 1.5168, 1.0, 49.74937)
o = rt.OutputPlane(200.0)
Beams, RMS, DiffractionLimits = [], [], []

#Create and propagate beams of different radii
for t in np.linspace(0.1,5.0,10):
	Beams.append(rt.CollimatedBeam(r, t, 100))
for elem in Beams:
	elem.trace2(s,p,o)
	RMS.append(elem.rms)
"""

"""
Lens optimisation test and data collection suite

def optimiselens(beam, n):
	Find the best two lens curvatures to produce the minimum rms value, with
	n = number of curvature increments between -1 and 1, and beam = input beam
	rms = []
	outputplane = rt.OutputPlane(250)
	for i in np.linspace(-1.0, 1.0, n):
		for x in np.linspace(-1.0, 1.0, n):
			if i == 0:
				ar = 30
			else:
				ar = abs(1/i)
			lens1 = rt.SphericalRefraction(100.0, i, 1.0, 1.5168, ar)
			lens2 = rt.SphericalRefraction(105.0, x, 1.5168, 1.0, ar)
			beam.trace2(lens1, lens2, outputplane)
			rms.append([i, x, beam.rms(outputplane)])
	m = MinList(rms, [0, 0, 100]) 
	return "lens 1 curvature = %s, lens 2 curvature = %s, rms = %s" %(m[0], m[1], m[1])


def MinList(ls, m):
	In a list of lists, return the list with the smallest index 2 element
 	if ls == []:
 		return m
 	if ls[0][2] < m[2]:
 		return MinList(ls[1:], ls[0])
 	else:
 		return MinList(ls[1:], m)

r = rt.Ray([0.0,0.0,0.0],[0.0,0.0,1.0])
b = rt.CollimatedBeam(r, 5, 100)	

"""

"""Rainbow Test Suite"""
r = rt.Ray([10,10,0.0],[0.0,0.0,1.0], 380*10**(-9))
t = rt.Ray([0,-1,0.0],[0.2,0.1,2.0])
b = rt.CollimatedBeam(r, 1, 20)	
outputplane = rt.OutputPlane(2)
semi1 = rt.DispersiveRefraction('water', 50, 0.05, 20, 1.0)
semi2 = rt.SphericalReflection(90, -0.05, 20)
b.trace2dispersed(semi1, semi2, outputplane)
a = b._Beam[len(b._Beam)-1]
print a._wavelength



"""
b.create()
for Ray in b._Beam:
	Ray._wavelength = random.randint(380, 750)*10**(-9)
b.trace2(semi1, semi2, outputplane)
"""
















