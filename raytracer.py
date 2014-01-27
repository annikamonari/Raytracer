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
    	"propagate a ray through the optical element"
    	raise NotImplementedError()

class SphericalRefraction(OpticalElement):

	def __init__(self, z0 = 0.0, c = 0.0, n1 = 1.0, n2 = 1.0, ar = 0.0):
		self.__z0 = np.array(0.0, 0.0, z0)
		self.__c = c
		self.__n1 = n1
		self.__n2 = n2
		self.__ar = ar
		self.__Rvector = self.radialvector()
		
		# ar = aperture radius
		# c = curvature

	def radialvector(self):
		return np.array(0.0, 0.0, -(1/self.__c))
		#Rmag = abs(Rvector)
		#return Rvector

	def surface(self):
		if self.__c < 0:
			return "concave surface"
		elif self.__c > 0:
			return "convex surface"
		else:
			return "plane surface"

	def intercept(self, ray):
		


		


