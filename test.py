import numpy as np
import raytracer as rt
import random

"""Planoconvex test and data collection suite - beam path plots and spot plots at the
output plane are saved as .png figures in the working directory

r = rt.Ray([0.0, 0.0, 0.0],[0.0, 0.0, 1.0])
#s = rt.SphericalRefraction(100.0, .02, 1.0, 1.5168, 29.74944)
#p = rt.SphericalRefraction(105.0, 0.0, 1.5168, 1.0, 29.74944)
e = rt.SphericalRefraction(100.0, 0.0, 1.0, 1.5168, 29.74944)
l = rt.SphericalRefraction(105.0, -0.02, 1.5168, 1, 29.74944)
o = rt.OutputPlane(201.748)
Beams, RMS, DiffractionLimits = [], [], []

# Create and propagate beams of different radii
for t in np.linspace(0.1,5.0,10):
	Beams.append(rt.CollimatedBeam(r, t, 150))
for elem in Beams:
	elem.trace2(e,l,o)
	RMS.append(elem.rms(o))
	DiffractionLimits.append(elem.diffractionlimit(e,o))
print RMS
print DiffractionLimits

#s = rt.SphericalRefraction(100.0, .02, 1.0, 1.5168, 29.74944)
#p = rt.SphericalRefraction(105.0, 0.0, 1.5168, 1.0, 29.74944)





#o = rt.OutputPlane(210.0)
#r = rt.Ray([-0.1,-0.1, 0.0],[0.0,0.0,1.0])
#t = rt.Ray([0.1,0.1,0.0],[0.0,0.0,1.0])
#r.parallaxtracemultiple(t, s, p, o)

"""

"""Lens optimisation test and data collection suite"""

def optimiselens(beam, n):
	"""Find the best two lens curvatures to produce the minimum rms value, with
	n = number of curvature increments between -1 and 1, and beam = input beam"""
	rms = []
	outputplane = rt.OutputPlane(200)
	for i in np.linspace(-1.0, 1.0, n):
		for x in np.linspace(-1.0, 1.0, n):
			lens1 = rt.SphericalRefraction(100.0, i, 1.0, 1.5168, 29.74944)
			lens2 = rt.SphericalRefraction(105.0, x, 1.5168, 1.0, 29.74944)
			beam.trace2(lens1, lens2, outputplane)
			rms.append([i, x, beam.rms(outputplane)])
	m = MinList(rms, [0, 0, 100]) 
	print "lens 1 curvature = %s, lens 2 curvature = %s, rms = %s" %(m[0], m[1], m[1])


def MinList(ls, m):
	"""Finding the lowest rms values, returned with the two corresponding curvatures
	In a list of lists, return the list with the smallest index 2 element"""
 	if ls == []:
 		return m
 	if ls[0][2] < m[2]:
 		return MinList(ls[1:], ls[0])
 	else:
 		return MinList(ls[1:], m)


r = rt.Ray([0.0,0.0,0.0],[0.0,0.0,1.0])
b = rt.CollimatedBeam(r, 5, 50)
optimiselens(b, 10)




"""Rainbow Test Suite

r = rt.Ray([5,5,0.0],[0.0,0.0,1.0], 580*10**(-9))
t = rt.Ray([0,-1,0.0],[0.2,0.1,2.0])


b = rt.CollimatedBeam(r, 1, 50)	
outputplane = rt.OutputPlane(0)
semi1 = rt.DispersiveRefraction('water', 50, 0.05, 20, 1.0)
semi2 = rt.SphericalReflection(90, -0.05, 20)
#b.trace2dispersed(semi1, semi2, outputplane)
Beams = []
for t in np.linspace(0.1,5.0,10):
	for n in np.linspace(10, 200, 4):
		Beams.append(rt.CollimatedBeam(r, t, n))
for elem in Beams:
	elem.trace2dispersed(semi1, semi2, outputplane)

"""














