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
		self.__z0 = np.array(0.0, 0.0, z0)
		self.__c = c
		self.__n1 = n1
		self.__n2 = n2
		self.__ar = ar
		self.__r= self.radius()
		self.__s = self.surface()
		self.__centre = self.centre()
		
		# ar = aperture radius
		# c = curvature

	def radius(self):
		if self.__c == 0:
			return float("inf")
		else:
			return abs(1/self.__c)

	def surface(self):
		if self.__c < 0:
			return "concave"
		elif self.__c > 0:
			return "convex"
		else:
			return "plane"

	def centre(self):
		if self.__s == "convex":
			return np.array(0, 0, self.__z0 + self.__r)
		elif self.__s == "concave":
			return np.array(0, 0, self.__z0 - self.__r)
		else:
			return "plane surfaces have no centre"

	def intercept(self, ray):
		r = ray.p() - self.__centre
		r_mag = np.sqrt(sum(n**2 for n in r))
		rdotkunit = np.dot(r, ray.kunit())
		l1 = -rdotkunit + np.sqrt(rdotkunit**2 - r_mag**2 + self.__r**2)
		l2 = -rdotkunit - np.sqrt(rdotkunit**2 - r_mag**2 + self.__r**2)
		lplane = (self.__z0 - ray.p()[2]) / ray.kunit()[2]

		if (rdotkunit**2 - r_mag**2 + self.__r**2) < 0:
			return None
		elif (rdotkunit**2 - r_mag**2 + self.__r**2) == 0:
			return ray.p() + -rdotkunit*ray.kunit()
		elif self.__s == "convex": 
			return ray.p() + min(l1, l2)*ray.kunit()
		elif self.__s == "concave":
			return ray.p() + max(l1, l2)*ray.kunit()
		else:
			return ray.p() + lplane*ray.kunit()






		


