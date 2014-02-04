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

	def append(self, p = None , k = None):
		if p is not None:
			self.points.append(np.array(p))
		if k is not None:
			self.directions.append(np.array(k))		
		self.checklength()

	def vertices(self):
		for item in self.points: print item

class OpticalElement:

	def intercept(self, ray):
  		"""calculate the ray's intersection with the optical element"""
  		raise NotImplementedError()

  	def propagate_ray(self, ray):
  		"""propagate a ray through the optical element"""
  		raise NotImplementedError()


class SphericalRefraction(OpticalElement):

	def __init__(self, z0 = 0.0, c = 0.0, n1 = 1.0, n2 = 1.0, ar = 0.0):
		self.z0 = z0
		self.c = c
		self.n1 = n1
		self.n2 = n2
		self.ar = ar
		self.R= self.radius()
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
			return np.array([0, 0, self.z0 + self.R])
		elif self.s == "concave":
			return np.array([0, 0, self.z0 - self.R])
		elif self.s == "plane":
			return None

	def intercept(self, ray):
		ar_z = np.sqrt(self.R**2 - self.ar**2) 
		#ar_z = distance from aperture radius z intercept to centre of sphere
		r = ray.p() - self.centre
		r_mag = np.sqrt(sum(n**2 for n in r))
		rdotk = np.dot(r, ray.k())

		if (rdotk**2 - r_mag**2 + self.R**2) < 0:
			return None
		else:
			l1 = -rdotk + np.sqrt(rdotk**2 - r_mag**2 + self.R**2)
			l2 = -rdotk - np.sqrt(rdotk**2 - r_mag**2 + self.R**2)
			lplane = (self.z0 - ray.p()[2]) / ray.k()[2]

		if self.s == "convex":
			if (rdotk**2 - r_mag**2 + self.R**2) == 0:
				if self.centre[2] - ar_z >= (ray.p() + -rdotk*ray.k())[2]:
					return ray.p() + -rdotk*ray.k()
				else:
					return None
			else:
				if self.centre[2] - ar_z >= (ray.p() + min(l1, l2)*ray.k())[2]:
					return ray.p() + min(l1, l2)*ray.k()
				else:
					return None
		elif self.s == "concave":
			if (rdotk**2 - r_mag**2 + self.R**2) == 0:
				if self.centre[2] + ar_z <= (ray.p() + -rdotk*ray.k())[2]:
					return ray.p() + -rdotk*ray.k()
				else:
					return None
			else:
				if self.centre[2] + ar_z <= (ray.p() + max(l1, l2)*ray.k())[2]:
					return ray.p() + max(l1, l2)*ray.k()
				else:
					return None
		elif self.s == "plane":
			if np.sqrt((ray.p() + lplane*ray.k())[0]**2 + ray.p() + lplane*ray.k()[1]**2) <= self.ar:
				return ray.p() + lplane*ray.k()
			else:
				return None

	def unitsurfacenormal(self, ray):
		Q = self.intercept(ray)
		surface_normal = Q - self.centre
		return surface_normal/np.sqrt(sum(n**2 for n in surface_normal))

	def refract(self, ray):
		n_unit = self.unitsurfacenormal(ray)
		k1 = ray.k() 
		index = self.n1/self.n2
		ndotk1 = np.dot(n_unit, k1)
		if 1/index <= np.sin(np.arccos(ndotk1)):
			return index*k1 - (index*ndotk1 - np.sqrt(1- index**2(1-ndotk1**2)))*n_unit
		else:
			return None

	def propagate_ray(self, ray):
		if self.intercept(ray) is None or self.refract(ray) is None:
			return "Terminated"
		else:
			k2 = self.refract(ray)
			p = self.intercept(ray) + k2
			ray.append(p, k2)
			return "Final Point: %s" %(ray.p()) + "and Final Direction: %s" %(ray.k())

class OutputPlane(OpticalElement):

	def __init__(self, z0 = 0.0):
		self.z0 = z0

	def intercept(self, ray):
		l = (self.z0 - ray.p()[2]) / ray.k()[2]
		return ray.p() + lplane*ray.k()

	def propagate_ray(self, ray):
		p = self.intercept(ray)
		ray.append(p, None)
		return "Final Point: %s" %(ray.p()) + "and Final Direction: %s" %(ray.k())




	







		


