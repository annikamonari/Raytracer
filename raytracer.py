import numpy as np
import matplotlib.pyplot as plt

"""

My module creating a 3D 
optical ray tracer


** 
To make sure all operations work correctly, 
input all numbers as floats (e.g. 10.0 rather than 10) 
**


"""

def objectchecker(sphericalrefractionlist = None, outputplane = None, ray = None, sphericalreflectionlist = None):
	"""Check if an object passed as a parameter is the correct object"""
	if sphericalrefractionlist is not None:
		for i in sphericalrefractionlist:
			if not isinstance(i, SphericalRefraction):
				raise Exception("Surface input isn't SphericalRefraction surface")
	if outputplane is not None:
		if not isinstance(outputplane, OutputPlane):
			raise Exception("Output plane input is not an Output plane")
	if ray is not None:
		if not isinstance(ray, Ray):
			raise Exception("Ray input is not a ray object")
	if sphericalreflectionlist is not None:
		for i in sphericalreflectionlist:
			if not isinstance(i, SphericalReflection):
				raise Exception("Surface input isn't SphericalReflection surface")

class Ray:
	"""Create a ray with an initial point and direction"""

	def __init__(self, p = None, k = None):
		if p == None:
			p = [0,0,0]
		if k == None:
			k = [0,0,0]
		self._points = [np.array(p)]
		self._directions = [np.array(k)/np.sqrt(sum(n**2 for n in k))]
		self.checklength()

	def __repr__(self):
		return "[x, y, z] = %s + %st" % (self.p(), self.k())
	
	def checklength(self):
		if any(len(i) != 3 for i in self._points + self._directions):
			raise Exception("Ray point or direction parameter size")

	def points(self):
		return self._points

	def directions(self):
		return self._directions

	def p(self):
		return self._points[len(self._points)-1]

	def k(self):
		return self._directions[len(self._directions)-1]

	def append(self, p = None , k = None):
		if p is not None:
			self._points.append(np.array(p))
		if k is not None:
			self._directions.append(np.array(k))		
		self.checklength()

	def vertices(self):
		return list(item for item in self._points)

	def trace(self, sphericalrefraction, outputplane):
		"""propagates and plots the ray through a sphericalrefracting 
		surface and output plane"""
		#objectchecker([sphericalrefraction], outputplane)
		sphericalrefraction.propagate_ray(self)
		outputplane.propagate_ray(self)
		self.plot()

	def parallaxtrace(self, ray, sphericalrefraction, outputplane, sphericalreflection = None):
		"""estimates the parallaxial focus through 1 spherical refraction lens 
		and an optional sphericalreflecting lens using the trace() method for two rays"""
		objectchecker([sphericalrefraction], outputplane, ray, [sphericalreflection])
		sphericalrefraction.propagate_ray(self)
		sphericalrefraction.propagate_ray(ray)
		if sphericalreflection is not None:
			sphericalreflection.propagate_ray(self)
			sphericalreflection.propagate_ray(ray)
		outputplane.propagate_ray(self)
		outputplane.propagate_ray(ray)
		self.plotparallax(ray)

	def parallaxtrace2(self, ray, sphericalrefraction1, sphericalrefraction2, outputplane, sphericalreflection = None):
		"""estimates the parallaxial focus through two sphericalrefraction lenses and an optional
		sphericalreflecting lens using the trace() method for two rays"""
		objectchecker([sphericalrefraction1, sphericalrefraction2], outputplane, ray, [sphericalreflection])
		sphericalrefraction1.propagate_ray(self)
		sphericalrefraction1.propagate_ray(ray)
		sphericalrefraction2.propagate_ray(self)
		sphericalrefraction2.propagate_ray(ray)
		if sphericalreflection is not None:
			sphericalreflection.propagate_ray(self)
			sphericalreflection.propagate_ray(ray)
		outputplane.propagate_ray(self)
		outputplane.propagate_ray(ray)
		self.plotparallax(ray)

	def plot(self):
		"""plot the ray's path (x vs. z)"""
		z, y = [], []
		for i in self._points:
			z.append(i[2]), y.append(i[1])
		print z, y
		plt.plot(z, y, color = "Blue")
		plt.title('Beam Path')
		plt.xlabel('z')
		plt.ylabel('x')
		plt.show()

	def plotparallax(self, ray):
		"""plot two rays' paths (x vs. z)"""
		objectchecker(None, None, ray)
		z, y, p, q = [], [], [], []
		for i in self._points:
			z.append(i[2]), y.append(i[1])
		for i in ray._points:
			p.append(i[2]), q.append(i[1])
		plt.plot(z, y, color = "Red")
		plt.plot(p, q, color = "Blue")
		plt.title('Beam Path')
		plt.xlabel('z')
		plt.ylabel('y')
		plt.show()

class OpticalElement:
	"""Any type of lens, outputplane or reflecting surface that propagates a ray or beam"""

	def intercept(self, ray):
  		"""calculate the ray's intersection with the optical element"""
  		raise NotImplementedError()

  	def propagate_ray(self, ray):
  		"""propagate a ray through the optical element"""
  		raise NotImplementedError()

class SphericalRefraction(OpticalElement):
	"""A spherical or plane shaped lens that refracts the ray or beam"""

	def __init__(self, z0 = 0.0, c = 0.0, n1 = 1.0, n2 = 1.0, ar = 0.0):
		"""ar is the aperture radius, c is the curvature"""
		self._z0 = z0
		self._c = c
		self._n1 = n1
		self._n2 = n2
		self._ar = ar
		self._r = self.radius()
		self.s = self.surface()
		self.centre = self.centre()

	def __repr__(self):
		return "Spherical refracting lens at z=%s of c=%s, n1/n2=%s, ar =%s" %(self._z0, self._c, self._n1/self._n2, self._ar)

	def radius(self):
		if self._c == 0:
			return float("inf")
		else:
			return abs(1/self._c)

	def surface(self):
		"""Defines the surface type as a string as an attribute accessible from the interface"""
		if self._c < 0:
			return "concave"
		elif self._c > 0:
			return "convex"
		elif self._c == 0:
			return "plane"

	def centre(self):
		"""Calculates the sphere's centre z coordinate based on surface type"""
		if self.s == "convex":
			return np.array([0, 0, self._z0 + self._r])
		elif self.s == "concave":
			return np.array([0, 0, self._z0 - self._r])
		elif self.s == "plane":
			return 0

	def intercept(self, ray):
		objectchecker(None, None, ray)
		ar_z = np.sqrt(self._r**2 - self._ar**2) 
		#ar_z = distance from aperture radius z intercept to centre of sphere
		r = ray.p() - self.centre
		r_mag = np.sqrt(sum(n**2 for n in r))
		rdotk = np.dot(r, ray.k())
		discriminant = rdotk**2 - r_mag**2 + self._r**2

		if discriminant < 0:
			return None
		else:
			l1 = -rdotk + np.sqrt(rdotk**2 - r_mag**2 + self._r**2)
			l2 = -rdotk - np.sqrt(rdotk**2 - r_mag**2 + self._r**2)
			lplane = (self._z0 - ray.p()[2]) / ray.k()[2]

		if self.s == "convex":
			if discriminant == 0:
				"""if ray passes through aperture"""
				if self.centre[2] - ar_z >= (ray.p() + -rdotk*ray.k())[2]:
					return ray.p() + -rdotk*ray.k()
				else:
					return None
			else:
				"""Since convex, return the shortst length from ray point
				if ray passes through the aperture"""
				if self.centre[2] - ar_z >= (ray.p() + min(l1, l2)*ray.k())[2]:
					return ray.p() + min(l1, l2)*ray.k()
				else:
					return None
		elif self.s == "concave":
			if discriminant == 0:
				"""if ray passes through aperture"""
				if self.centre[2] + ar_z <= (ray.p() + -rdotk*ray.k())[2]:
					return ray.p() + -rdotk*ray.k()
				else:
					return None
			else:
				"""Since concave, return the longest length from ray point if ray passes
				through the aperture"""
				if self.centre[2] + ar_z <= (ray.p() + max(l1, l2)*ray.k())[2]:
					return ray.p() + max(l1, l2)*ray.k()
				else:
					return None
		elif self.s == "plane":
			"""if it passes through the aperture"""
			if np.sqrt((ray.p() + lplane*ray.k())[0]**2 + (ray.p() + lplane*ray.k())[1]**2) <= self._ar:
				return ray.p() + lplane*ray.k()
			else:
				return None


	def unitsurfacenormal(self, ray):
		"""Find the surface normal based on ray's intersection with a non-plane surface"""
		objectchecker(None, None, ray)		
		if self.s == "plane":
			return np.array([0,0,-1])
		else:
			Q = self.intercept(ray)
			surface_normal = Q - self.centre
			return surface_normal/np.sqrt(sum(n**2 for n in surface_normal))

	def refract(self, ray):
		"""Refracts the ray using snell's law calculating the new direction vector
		using the ray's direction vector upon intersection"""
		objectchecker(None, None, ray)
		n_unit = -self.unitsurfacenormal(ray)
		k1 = ray.k() 
		ref = self._n1/self._n2
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
			return "Final Point: %s" %(ray.p()) + " and Final Direction: %s" %(ray.k())

class OutputPlane(OpticalElement):
	"""A plane at the end of the ray or beam's path to record its final point after propagation
	through the other optical elements"""

	def __init__(self, z = 0.0):
		self._zoutput = z

	def __repr__(self):
		return "Output plane at z=%s" %(self._zoutput)

	def intercept(self, ray):
		objectchecker(None, None, ray)
		l = (self._zoutput - ray.p()[2]) / ray.k()[2]
		return ray.p() + l*ray.k()

	def propagate_ray(self, ray):
		objectchecker(None, None, ray)
		p = self.intercept(ray)
		ray.append(p, None)
		return "Final Point: %s" %(ray.p()) + "and Final Direction: %s" %(ray.k())

class SphericalReflection(SphericalRefraction):
	"""inherits from SphericalRefraction because the intercept method is long and will be analogous"""

	def __init__(self, z0 = 0.0, c = 0.0, ar = 0.0):
		self._z0 = z0
		self._c = c
		self._ar = ar 
		self._r = self.radius()
		self.s = self.surface()
		self.centre = self.centre()

	def reflect(self, ray):
		n_unit = self.unitsurfacenormal(ray)
		k1 = ray.k()
		ndotk1 = np.dot(n_unit, k1)
		return k1 - 2*ndotk1*n_unit 

	def propagate_ray(self, ray):
		objectchecker(None, None, ray)
		if self.intercept(ray) is None:
			return "Terminated"
		else:
			p = self.intercept(ray)
			k2 = self.reflect(ray)
			ray.append(p, k2)
			return "Final Point: %s" %(ray.p()) + " and Final Direction: %s" %(ray.k())

class CollimatedBeam:
	"""Create a collimated beam concentrically around an input ray with a point
	at x,y (whose direction is used for all rays in beam), a beam radius (r) 
	and the number rays in the beam (n)"""

	def __init__(self, ray, r, n):
		self._n = n
		self._r = r
		self._ray = ray
		self._x, self._y = self._ray.p()[0], self._ray.p()[1]
		self._Beam, self._Beampoints = [self._ray], [self._ray._points]
		self.rms, self.diflim = [], []
		objectchecker(None, None, ray)

	def __repr__(self):
		return "Collimated Beam of radius = %s, centred at (%s,%s)" % (self._r,self._x,self._y)

	def create(self):
		"""Fills in the self._Beam list by creating a variable number of
		new ray instances"""
		radii_num = .5+np.sqrt(0.25 + self._n/np.pi)
		for r in np.linspace(0, self._r, radii_num, endpoint = True):
			for i in np.linspace(0, 2*np.pi, (2*np.pi*r*(radii_num-1))/self._r, endpoint = True):
				self._Beam.append(Ray([self._x+r*np.cos(i), self._y + r*np.sin(i), 0], self._ray.k()))
		return self._Beam

	def raynumber(self):
		"""the number of rays in the beam, also given by np.pi*self._n*(self._n-1)"""
		return len(self._Beam)

	def trace(self, sphericalrefraction, outputplane):
		"""propagates and plots the beam through a sphericalrefracting 
		surface and output plane"""
		objectchecker([sphericalrefraction], outputplane)
		#self.create()
		for i in self._Beam:
			sphericalrefraction.propagate_ray(i)
			outputplane.propagate_ray(i)
			self._Beampoints.append(i.vertices())
		self.plot()
		self.spotplot(outputplane._zoutput)
		print 'Diffraction Limit = %s' %(((outputplane._zoutput - sphericalrefraction._z0)*680*10**(-9))/(2*self._r))

	def trace2(self, sphericalrefraction1, sphericalrefraction2, outputplane, sphericalreflection = None):
		"""propagates and plots the beam through two sphericalrefraction lenses 
		using the trace() method for two rays"""
		objectchecker([sphericalrefraction1, sphericalrefraction2], outputplane, None, [sphericalreflection])
		self.create()
		for i in self._Beam:
			sphericalrefraction1.propagate_ray(i)
			sphericalrefraction2.propagate_ray(i)
			outputplane.propagate_ray(i)
			self._Beampoints.append(i.vertices())
		self.plot()
		self.spotplot(outputplane._zoutput)
		diflim=((outputplane._zoutput - sphericalrefraction1._z0)*680*10**(-9))/(2*self._r)
		self.diflim.append('for r=%s, Diffraction limit=%s' %(self._r,diflim))
		print 'Diffraction Limit = %s' %(diflim)

	def plot(self):
		"""plot the beam's path (y vs. z)"""
		z_coords, y_coords = [], []
		for list in self._Beampoints:
			z, y = [], []
			for i in list:
				z.append(i[2]), y.append(i[1])
			z_coords.append(z), y_coords.append(y)		
		for z,y in zip(z_coords, y_coords):
				plt.plot(z, y, color = "Blue")
		plt.title('Beam Path for beam of %s rays, beam radius: %s' %(self.raynumber(), self._r))
		plt.xlabel('z')
		plt.ylabel('y')
		plt.xlim(0, 300)
		#plt.show()
		plt.savefig('%s_path.png' %(self._r))

	def spotplot(self, z = 0):
		"""plot the x-y distribution of rays in the beam at various z coordinates 
		along the beam's path"""
		coords, distances = [], []
		for list in self._Beampoints:
			for i in list:
				if i[2] == z:
					coords.append((i[0],i[1]))
		for i in coords:
			distances.append(np.sqrt(sum(n**2 for n in i)))
		rms = np.sqrt(sum((n**2 for n in distances))/len(coords))
		self.rms.append('for r=%s, rms=%s' %(self._r, rms))
		print 'Rms deviation = %s' %(rms)
		plt.scatter(*zip(*coords), color = "Red")
		plt.title('Spot Diagram at z = %s for beam of %s rays, beam radius: %s' %(z, self.raynumber(), self._r))
		plt.xlabel('x')
		plt.ylabel('y')
		#plt.show()
		plt.savefig('%s_spot.png' %(self._r))





	





		


