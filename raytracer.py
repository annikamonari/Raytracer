import numpy as np
import matplotlib.pyplot as plt

"""

My module creating a 3D 
optical ray tracer

"""

def objectchecker(sphericalrefractionlist = None, outputplane = None, ray = None):
	for i in sphericalrefractionlist:
		if not isinstance(i, SphericalRefraction):
			raise Exception("Surface input isn't SphericalRefraction surface")
	if not isinstance(outputplane, OutputPlane):
		raise Exception("Output plane input is not an Output plane")
	if not isinstance(ray, Ray):
		raise Exception("Ray input is not a ray object")

class Ray:

	def __init__(self, p = None, k = None):
		if p == None:
			p = [0,0,0]
		if k == None:
			k = [0,0,0]
		self.__points = [np.array(p)]
		self.__directions = [np.array(k)/np.sqrt(sum(n**2 for n in k))]
		self.checklength()

	def __repr__(self):
		return "[x, y, z] = %s + %st" % (self.p(), self.k())
	
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

	def append(self, p = None , k = None):
		if p is not None:
			self.__points.append(np.array(p))
		if k is not None:
			self.__directions.append(np.array(k))		
		self.checklength()

	def vertices(self):
		for item in self.__points: print item

	# propagates and plots the ray through a sphericalrefracting surface and output plane
	def trace(self, sphericalrefraction, outputplane):
		objectchecker([sphericalrefraction], outputplane)
		sphericalrefraction.propagate_ray(self)
		outputplane.propagate_ray(self)
		self.plot()

	# estimates the paralaxial focus using 'trace' method for two rays and 1 surface
	def parallaxtrace(self, ray, sphericalrefraction, outputplane):
		objectchecker([sphericalrefraction], outputplane, ray)
		sphericalrefraction.propagate_ray(self)
		sphericalrefraction.propagate_ray(ray)
		outputplane.propagate_ray(self)
		outputplane.propagate_ray(ray)
		self.plotparallax(ray)

	# estimates the paralaxial focus using 'trace' method for two rays and 2 surfaces
	def parallaxtrace2(self, sphericalrefraction1, sphericalrefraction2, outputplane, ray):
		objectchecker([sphericalrefraction1, sphericalrefraction2], outputplane, ray)
		sphericalrefraction1.propagate_ray(self)
		sphericalrefraction1.propagate_ray(ray)
		sphericalrefraction2.propagate_ray(self)
		sphericalrefraction2.propagate_ray(ray)
		outputplane.propagate_ray(self)
		outputplane.propagate_ray(ray)
		self.plotparallax(ray)

	def plot(self):
		z, y = [], []
		for i in self.___points:
			z.append(i[2]), y.append(i[1])
		print z, y
		plt.plot(z, y, color = "Blue")
		plt.title('Beam Path')
		plt.xlabel('z')
		plt.ylabel('x')
		plt.show()

	def plotparallax(self, ray):
		objectchecker(None, None, ray)
		z, y, p, q = [], [], [], []
		for i in self.__points:
			z.append(i[2]), y.append(i[1])
		for i in ray.__points:
			p.append(i[2]), q.append(i[1])
		plt.plot(z, y, color = "Red")
		plt.plot(p, q, color = "Blue")
		plt.title('Beam Path')
		plt.xlabel('z')
		plt.ylabel('y')
		plt.show()

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
		self.R = self.radius()
		self.s = self.surface()
		self.centre = self.centre()
		
		# ar = aperture radius
		# c = curvature

	def __repr__(self):
		return "Spherical refracting lens at z=%s of c=%s, n1/n2=%s, ar =%s" %(self.z0, self.c, self.n1/self.n2, self.ar)

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
			return 0

	def intercept(self, ray):
		objectchecker(None, None, ray)
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
			if np.sqrt((ray.p() + lplane*ray.k())[0]**2 + (ray.p() + lplane*ray.k())[1]**2) <= self.ar:
				return ray.p() + lplane*ray.k()
			else:
				return None

	def unitsurfacenormal(self, ray):
		objectchecker(None, None, ray)		
		if self.s == "plane":
			return np.array([0,0,-1])
		else:
			Q = self.intercept(ray)
			surface_normal = Q - self.centre
			return surface_normal/np.sqrt(sum(n**2 for n in surface_normal))

	def refract(self, ray):
		objectchecker(None, None, ray)
		n_unit = -self.unitsurfacenormal(ray)
		k1 = ray.k() 
		ref = self.n1/self.n2
		ndotk1 = np.dot(n_unit, k1)
		if np.sin(np.arccos(ndotk1)) > (1/ref):
			return None
		else:	
			return ref*k1 - (ref*ndotk1 - np.sqrt(1- (ref**2)*(1-ndotk1**2)))*n_unit

	def propagate_ray(self, ray):
		objectchecker(None, None, ray)
		if self.intercept(ray) is None or self.refract(ray) is None:
			return "Terminated"
		else:
			p = self.intercept(ray)
			k2 = self.refract(ray)
			ray.append(p, k2)
			print "Final Point: %s" %(ray.p()) + " and Final Direction: %s" %(ray.k())

class OutputPlane(OpticalElement):

	def __init__(self, z = 0.0):
		self.z_output = z

	def __repr__(self):
		return "Output plane at z=%s" %(self.z_output)

	def intercept(self, ray):
		objectchecker(None, None, ray)
		l = (self.z_output - ray.p()[2]) / ray.k()[2]
		return ray.p() + l*ray.k()

	def propagate_ray(self, ray):
		objectchecker(None, None, ray)
		p = self.intercept(ray)
		ray.append(p, None)
		return "Final Point: %s" %(ray.p()) + "and Final Direction: %s" %(ray.k())

class CollimatedBeam:

	def __init__(self, Ray, r = 10, n = 6):
		self.n = n
		self.r = r
		self.Ray = Ray
		self.x, self.y = self.Ray.p()[0], self.Ray.p()[1]
		self.Beam, self.Beam__points = [self.Ray], [self.Ray.__points]
		self.raynumber = self.raynumber()
		objectchecker(None, None, ray)

	def __repr__(self):
		return "Collimated Beam of radius = %s, density = %s, centred at (%s,%s)" % (self.r,self.d,self.x,self.y)

	def create(self):
		for r in np.linspace(0, self.r, self.n, endpoint = True):
			for i in np.linspace(0, 2*np.pi, (2*np.pi*r*(self.n-1))/self.r, endpoint = True):
				self.Beam.append(Ray([self.x+r*np.cos(i), self.y + r*np.sin(i), 0], self.Ray.k()))
		return self.Beam

	def raynumber(self):
		return len(self.Beam)

	def trace(self, sphericalrefraction, outputplane):
		objectchecker([sphericalrefraction], outputplane)
		self.create()
		for i in self.Beam:
			sphericalrefraction.propagate_ray(i)
			outputplane.propagate_ray(i)
			self.Beam__points.append(i.__points)
		self.plot()
		self.spotplot(outputplane.z_output)
		print 'Diffraction Limit = %s' %(((outputplane.z_output - sphericalrefraction.z0)*680*10**(-9))/(2*self.r))

	def trace2(self, sphericalrefraction1, sphericalrefraction2, outputplane):
		self.create()
		for i in self.Beam:
			sphericalrefraction1.propagate_ray(i)
			sphericalrefraction2.propagate_ray(i)
			outputplane.propagate_ray(i)
			self.Beam__points.append(i.__points)
		self.plot()
		self.spotplot(outputplane.z_output)
		print 'Diffraction Limit = %s' %(((outputplane.z_output - sphericalrefraction1.z0)*680*10**(-9))/(2*self.r))

	def plot(self):
		z_coords, y_coords = [], []
		for list in self.Beam__points:
			z, y = [], []
			for i in list:
				z.append(i[2]), y.append(i[1])
			z_coords.append(z), y_coords.append(y)		
		for z,y in zip(z_coords, y_coords):
				plt.plot(z, y, color = "Blue")
		plt.title('Beam Path for beam of %s rays, beam radius: %s' %(self.raynumber, self.r))
		plt.xlabel('z')
		plt.ylabel('y')
		plt.show()

	def spotplot(self, z = 0):
		coords, distances = [], []
		for list in self.Beam__points:
			for i in list:
				if i[2] == z:
					coords.append((i[0],i[1]))
		for i in coords:
			distances.append(np.sqrt(sum(n**2 for n in i)))
		print 'Rms deviation = %s' %(np.sqrt(sum((n**2 for n in distances))/len(coords)))
		plt.scatter(*zip(*coords), color = "Red")
		plt.title('Spot Diagram at z = %s for beam of %s rays, beam radius: %s' %(z, self.raynumber, self.r))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()



	







		


