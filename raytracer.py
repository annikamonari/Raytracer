import numpy as np
import matplotlib as plt

"""

My module creating a 3D 
optical ray tracer

"""

class Ray:

	def __init__(self, p = [0.0, 0.0, 0.0], k = [0.0, 0.0, 0.0]):
		self.__points = [np.array(p)]
		self.__directions = [np.array(k)]
		self.checklength()
	
	def checklength(self):
		if any(len(i) != 3 for i in self.__points + self.__directions):
			raise Exception("Ray point or direction parameter size")

	def points(self):
		return self.__points

	def directions(self):
		return self.__directions

	def p(self):
		return self.__points[len(self.__points)-1]

	def k(self):
		return self.__directions[len(self.__directions)-1]

	def kunit(self):
		return self.k()/sum(self.k())


	def append(self, p, k):
		self.__points.append(np.array(p))
		self.__directions.append(np.array(k))
		self.checklength()

	def vertices(self):
		for item in self.__points: print item

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
		rdotkunit = np.dot(r, ray.kunit())
		l1 = -rdotkunit + np.sqrt(rdotkunit**2 - r_mag**2 + self.r**2)
		l2 = -rdotkunit - np.sqrt(rdotkunit**2 - r_mag**2 + self.r**2)
		lplane = (self.z0 - ray.p()[2]) / ray.kunit()[2]

		if (rdotkunit**2 - r_mag**2 + self.r**2) < 0:
			return None
		elif (rdotkunit**2 - r_mag**2 + self.r**2) == 0:
			return ray.p() + -rdotkunit*ray.kunit()
		elif self.s == "convex": 
			return ray.p() + min(l1, l2)*ray.kunit()
		elif self.s == "concave":
			return ray.p() + max(l1, l2)*ray.kunit()
		elif self.s == "plane":
			return ray.p() + lplane*ray.kunit()

	def unitsurfacenormal(self, ray):
		Q = self.intercept(ray)
		surface_normal = Q - self.centre()
		return surface_normal/np.sqrt(sum(n**2 for n in surface_normal))

	def refract(self, ray):
		unit_n = self.unitsurfacenormal(ray)
		unit_k = ray.kunit()
		theta_1 = np.arccos(np.dot(unit_k, unit_n))
		theta_2 = np.arcsin((self.n1*np.sin(theta_1)/self.n2))
		






		


