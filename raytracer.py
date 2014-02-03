import numpy as np
import matplotlib as plt

"""

My module creating a 3D 
optical ray tracer

"""

class Ray:

	def __init__(self, p = [0.0, 0.0, 0.0], k = [0.0, 0.0, 0.0]):
		self.points = [np.array(p)]
		self.directions = [np.array(k)/np.sqrt(sum(n**2 for n in k))]
		self.checklength()
	
	def checklength(self):
		if any(len(i) != 3 for i in self.points + self.directions):
			raise Exception("Ray point or direction parameter size")

	def points(self):
		return self.points

	def directions(self):
		return self.directions

	def p(self):
		return self.points[len(self.points)-1]

	def k(self):
		return self.directions[len(self.directions)-1]

	def append(self, p = [0.0, 0.0, 0.0], k = [0.0, 0.0, 0.0]):
		self.points.append(np.array(p))
		self.directions.append(np.array(k))
		self.checklength()

	def vertices(self):
		for item in self.points: print item

class OpticalElement:
  	def propagate(self, ray):
  		"""propagate a ray through the optical element"""
  		raise NotImplementedError()

class SphericalRefraction:

	def __init__(self, z0 = 0.0, c = 0.0, n1 = 1.0, n2 = 1.0, ar = 0.0):
		self.z0 = np.array([0.0, 0.0, z0])
		self.c = c
		self.n1 = n1
		self.n2 = n2
		self.ar = ar
		self.r= self.radius()
		self.s = self.surface()
		self.centre = self.centre()
		
		# ar = aperture radius
		# c = curvature

	def radius(self):
		if self.c == 0:
			return float("inf")
		else:
			return abs(1/self.c)

	def surface(self):
		if self.c < 0:
			return "concave"
		elif self.c > 0:
			return "convex"
		elif self.c == 0:
			return "plane"

	def centre(self):
		if self.s == "convex":
			return np.array(0, 0, self.z0 + self.r)
		elif self.s == "concave":
			return np.array(0, 0, self.z0 - self.r)
		elif self.s == "plane":
			return "plane surfaces have no centre"

	def intercept(self, ray):
		r = ray.p() - self.centre
		r_mag = np.sqrt(sum(n**2 for n in r))
		rdotk = np.dot(r, ray.k())
		l1 = -rdotk + np.sqrt(rdotk**2 - r_mag**2 + self.r**2)
		l2 = -rdotk - np.sqrt(rdotk**2 - r_mag**2 + self.r**2)
		lplane = (self.z0 - ray.p()[2]) / ray.k()[2]

		if (rdotk**2 - r_mag**2 + self.r**2) < 0:
			return "No valid intercept"
		elif (rdotk**2 - r_mag**2 + self.r**2) == 0:
			return ray.p() + -rdotk*ray.k()
		elif self.s == "convex": 
			return ray.p() + min(l1, l2)*ray.k()
		elif self.s == "concave":
			return ray.p() + max(l1, l2)*ray.k()
		elif self.s == "plane":
			return ray.p() + lplane*ray.k()

	def unitsurfacenormal(self, ray):
		Q = self.intercept(ray)
		surface_normal = Q - self.centre()
		return surface_normal/np.sqrt(sum(n**2 for n in surface_normal))

	def refract(self, ray, n = [0.0, 0.0, 0.0]):
		if n is None:
			n_unit = self.unitsurfacenormal(ray)
		else: 
			n_unit = n/np.sqrt(sum(i**2 for i in np.array(n)))
			# in case the person doesn't input n as a unit vector
		k1 = ray.k() 
		index = self.n1/self.n2
		ndotk1 = np.dot(n_unit, k1)
		if 1/index <= np.sin(np.arccos(ndotk1):
			return index*k1 - (index*ndotk1 - np.sqrt(1- index**2(1-ndotk1**2)))*n_unit
		else:
			return "None"

	def propagate_ray(self, ray):
		k2 = self.refract(ray)
		p = self.intercept(ray) + k2
		ray.append(p, k)
		if self.intercept(ray) == "No valid intercept" or self.refract(ray) == "None":
			return "Terminated"
		else:
			return "Final Point: " + ray.p().to_s + "and" + "Direction: " + ray.k().to_s

	







		


