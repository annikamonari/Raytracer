import numpy as np
import matplotlib.pyplot as plt
import random

"""

My module creating a 3D 
optical ray tracer


** 
To make sure all operations work correctly, 
input all numbers as floats (e.g. 10.0 rather than 10) 
**


"""

class Ray:
	"""Create a ray with an initial point and direction"""

	def __init__(self, p = None, k = None, wavelength = None):
		if p == None:
			p = [0,0,0]
		if k == None:
			k = [0,0,0]
		if wavelength == None:
			self._wavelength = 680*10**(-9)
		else:
			self._wavelength = wavelength
		self._points = [np.array(p)]
		self._directions = [np.array(k)/np.sqrt(sum(n**2 for n in k))]
		self.checklength()

	def __repr__(self):
		return "[x, y, z] = %s + %st" % (self.p(), self.k())
	
	def checklength(self):
		if any(len(i) != 3 for i in self._points+self._directions):
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

	def trace(self, opticalelement, outputplane):
		"""propagates and plots the ray through two optical elements 
		including the output plane"""
		opticalelement.propagate_ray(self)
		outputplane.propagate_ray(self)
		self.plot()

	def parallaxtrace(self, ray, opticalelement, outputplane):
		"""estimates the parallaxial focus through two optical elements
		including the output plane using the trace() method for two rays"""
		opticalelement.propagate_ray(self)
		opticalelement.propagate_ray(ray)
		outputplane.propagate_ray(self)
		outputplane.propagate_ray(ray)
		self.plotparallax(ray)

	def parallaxtracemultiple(self, ray, opticalelement1, opticalelement2, outputplane, opticalelement3 = None, opticalelement4 = None):
		"""estimates the parallaxial focus through multiple optical elements (including
		the output plane) using the trace() method for two rays"""
		opticalelement1.propagate_ray(self)
		opticalelement1.propagate_ray(ray)
		opticalelement2.propagate_ray(self)
		opticalelement2.propagate_ray(ray)
		if opticalelement3 is not None:
			opticalelement3.propagate_ray(self)
			opticalelement3.propagate_ray(ray)
		elif opticalelement4 is not None:
			opticalelement4.propagate_ray(self)
			opticalelement4.propagate_ray(ray)
		outputplane.propagate_ray(self)
		outputplane.propagate_ray(ray)
		self.plotparallax(ray)

	def Raycolour(self):
		"""Creates a string of the wavelength's corresponding visible light colour"""
		if 380 <= self._wavelength/(10**(-9)) < 450:
			return "Violet"
		elif 450 <= self._wavelength/(10**(-9)) < 495:
			return "Blue"
		elif 495 <= self._wavelength/(10**(-9)) < 570:
			return "Green"
		elif 570 <= self._wavelength/(10**(-9)) < 590:
			return "Yellow"
		elif 590 <= self._wavelength/(10**(-9)) < 620:
			return "Orange"
		elif 620 <= self._wavelength/(10**(-9)) <= 750:
			return "Red"

	def plot(self):
		"""plot the ray's path (x vs. z)"""
		z, y = [], []
		for i in self._points:
			z.append(i[2]), y.append(i[1])
		plt.plot(z, y, color = self.Raycolour())
		plt.title('Beam Path')
		plt.xlabel('z')
		plt.ylabel('x')
	

	def plotparallax(self, ray):
		"""plot two rays' paths (x vs. z) in their wavelength corresponding colour"""
		z, y, p, q = [], [], [], []
		for i in self._points:
			z.append(i[2]), y.append(i[1])
		for i in ray._points:
			p.append(i[2]), q.append(i[1])
		plt.plot(z, y, color = self.Raycolour())
		plt.plot(p, q, color = ray.Raycolour())
		plt.title('Beam Path')
		plt.xlabel('z')
		plt.ylabel('y')
		plt.savefig('r1_%s.%s_r2_%s.%s.png' %(self._points[0][0], self._points[0][1], ray._points[0][0], ray._points[0][1]))
		plt.show()

class OpticalElement:
	"""Any type of lens (dispersive or not), outputplane or reflecting 
	surface that propagates a ray or beam"""

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
			return np.array([0, 0, self._z0+self._r])
		elif self.s == "concave":
			return np.array([0, 0, self._z0-self._r])
		elif self.s == "plane":
			return 0

	def intercept(self, ray):
		r = ray.p()-self.centre
		r_mag = np.sqrt(sum(n**2 for n in r))
		rdotk = np.dot(r, ray.k())
		discriminant = rdotk**2-r_mag**2+self._r**2
		"""Discriminant in calculating l for convex/concave lenses"""
		if discriminant < 0:
			return None
		elif discriminant == 0:
			return ray.p() + -rdotk*ray.k()
		else:
			"""Calculates different lengths to intercept and intercepts for each curvature type"""
			l1, l2 = -rdotk+np.sqrt(rdotk**2-r_mag**2+self._r**2), -rdotk-np.sqrt(rdotk**2-r_mag**2+self._r**2)
			interceptconvex, interceptconcave = ray.p() + min(l1, l2)*ray.k(), ray.p() + max(l1, l2)*ray.k()
			lplane = (self._z0-ray.p()[2])/ray.k()[2]	
			interceptplane = ray.p() + lplane*ray.k()			
			if self.s == "convex":
				if np.sqrt(interceptconvex[0]**2+interceptconvex[1]**2) <= self._ar:
					return interceptconvex
				else:
					return None
			elif self.s == "concave":
				if np.sqrt(interceptconcave[0]**2+interceptconcave[1]**2) <= self._ar:
					return interceptconcave
			elif self.s == "plane":
				if np.sqrt(interceptplane[0]**2+interceptplane[1]**2) <= self._ar:
					return interceptplane
				else:
					return None

	def unitsurfacenormal(self, ray):
		"""Find the surface normal based on ray's intersection with a non-plane surface"""	
		if self.s == "plane":
			return np.array([0,0,-1])
		else:
			Q = self.intercept(ray)
			surface_normal = Q - self.centre
			return surface_normal/np.sqrt(sum(n**2 for n in surface_normal))

	def refract(self, ray):
		"""Refracts the ray using snell's law calculating the new direction vector
		using the ray's direction vector upon intersection"""
		if self.s == "concave":
			n_unit = self.unitsurfacenormal(ray)
		else:
			n_unit = -self.unitsurfacenormal(ray)
		k1 = ray.k() 
		ref = self._n1/self._n2
		ndotk1 = np.dot(n_unit, k1)
		if np.sin(np.arccos(ndotk1)) > (1/ref):
			return None
		else:	
			return ref*k1 - (ref*ndotk1 - np.sqrt(1- (ref**2)*(1-ndotk1**2)))*n_unit

	def propagate_ray(self, ray):
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
		l = (self._zoutput - ray.p()[2]) / ray.k()[2]
		return ray.p() + l*ray.k()

	def propagate_ray(self, ray):
		p = self.intercept(ray)
		ray.append(p, None)
		return "Final Point: %s" %(ray.p()) + "and Final Direction: %s" %(ray.k())

class DispersiveRefraction(SphericalRefraction):
	"""Refracts rays based on their wavelengths"""

	def __init__(self, material, z0 = 0.0, c = 0.0, ar = 0.0, n1 = 0.0):
		"""Checks if the input material matches materials with known Sellmeier coefficients"""
		materials = ['water', 'borosilicate', 'fusedsilica', 'sapphire']
		if materials.count(material) == 0:
			raise Exception("Material does not have known Sellmeier coefficients")		
		self._material = material
		self._z0 = z0
		self._c = c
		self._ar = ar 
		self._r = self.radius()
		self._n1 = n1
		self.s = self.surface()
		self.centre = self.centre()
		
	def SellmeierCoefficients(self):
		"""Allocates known coefficients to the input material for refractive
		index calculation"""
		if self._material == 'water':
			a, b = [0.568909, 0.17190886, 0.0206250158], [0.00510302, 0.0182518, 0.026241589]
		elif self._material == 'borosilicate':
			a, b = [1.040, 0.2318, 1.0105], [0.0060, 0.0200, 103.5607]
		elif self._material == 'fusedsilica':
			a, b = [0.6962, 0.4079, 0.8975], [0.004679, 0.01351, 97.9340]
		elif self._material == 'sapphire':
			a, b = [1.431, 0.6505, 5.341], [0.005279, 0.01424, 325.0178]
		return (a, b)

	def n2(self, ray):
		"""Calculates the refractive index for the ray's wavelength"""
		w = ray._wavelength*10**(6)
		a, b, tmp = self.SellmeierCoefficients()[0], self.SellmeierCoefficients()[1], []
		for i in range(3):
			tmp.append((a[i]*w**2)/(w**2 - b[i]))
		return np.sqrt(1+sum(tmp))

	def refract(self, ray):
		"""Refracts the ray using snell's law calculating the new direction vector
		using the ray's direction vector upon intersection"""
		n_unit = -self.unitsurfacenormal(ray)
		k1 = ray.k() 
		ref = self._n1/self.n2(ray)
		ndotk1 = np.dot(n_unit, k1)
		if np.sin(np.arccos(ndotk1)) > (1/ref):
			return None
		else:	
			return ref*k1 - (ref*ndotk1 - np.sqrt(1- (ref**2)*(1-ndotk1**2)))*n_unit

	def propagate_ray(self, ray):
		if self.intercept(ray) is None or self.refract(ray) is None:
			return "Terminated"
		else:
			p = self.intercept(ray)
			k2 = self.refract(ray)
			ray.append(p, k2)
			return "Final Point: %s" %(ray.p()) + " and Final Direction: %s" %(ray.k()) 

class SphericalReflection(SphericalRefraction):
	"""inherits from SphericalRefraction because the intercept method for spheres can be reused"""

	def __init__(self, z0 = 0.0, c = 0.0, ar = 0.0):
		self._z0 = z0
		self._c = c
		self._ar = ar 
		self._r = self.radius()
		self.s = self.surface()
		self.centre = self.centre()

	def reflect(self, ray):
		"""Calculates the new direction vector upon reflection"""
		n_unit = self.unitsurfacenormal(ray)
		k1 = ray.k()
		ndotk1 = np.dot(n_unit, k1)
		return k1 - 2*ndotk1*n_unit 

	def propagate_ray(self, ray):
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
		self._wavelength = ray._wavelength
		self._x, self._y = self._ray.p()[0], self._ray.p()[1]
		self._Beam, self._Beampoints = [self._ray], [self._ray._points]

	def __repr__(self):
		return "Collimated Beam of radius = %s, centred at (%s,%s)" % (self._r,self._x,self._y)

	def create(self):
		"""Fills in the self._Beam list by creating a variable number of
		new ray instances"""
		radii_num = .5+np.sqrt(0.25 + self._n/np.pi)
		for r in np.linspace(0, self._r, radii_num, endpoint = True):
			for i in np.linspace(0, 2*np.pi, (2*np.pi*r*(radii_num-1))/self._r, endpoint = True):
				self._Beam.append(Ray([self._x+r*np.cos(i), self._y + r*np.sin(i), 0], self._ray.k(), self._ray._wavelength))
		return self._Beam

	def createdispersed(self):
		"""Fills in the self._Beam list by creating a variable number of
		new ray instances with different wavelengths"""
		radii_num = .5+np.sqrt(0.25 + self._n/np.pi)
		for r in np.linspace(0, self._r, radii_num, endpoint = True):
			for i in np.linspace(0, 2*np.pi, (2*np.pi*r*(radii_num-1))/self._r, endpoint = True):
				self._Beam.append(Ray([self._x+r*np.cos(i), self._y + r*np.sin(i), 0], self._ray.k(), random.randint(380, 750)*10**(-9)))
		return self._Beam

	def rainbow(self):
		"""Fills in the self._Beam list by creating a variable number of
		new ray instances with different wavelengths"""
		radii_num = .5+np.sqrt(0.25 + self._n/np.pi)
		for r in np.linspace(0, self._r, radii_num, endpoint = True):
			for i in np.linspace(0, 2*np.pi, (2*np.pi*r*(radii_num-1))/self._r, endpoint = True):
				for y in np.linspace(380, 750, self._n, endpoint = True):
					self._Beam.append(Ray([self._x+r*np.cos(i), self._y + r*np.sin(i), 0], self._ray.k(), y*10**(-9)))
		return self._Beam

	def trace(self, opticalelement, outputplane):
		"""propagates and plots the beam through 2 optical elements including
		an output plane"""
		self.createdispersed()
		for i in self._Beam:
			opticalelement.propagate_ray(i)
			outputplane.propagate_ray(i)
			self._Beampoints.append(i.vertices())
		self.plotdispersed()
		#self.spotplot(outputplane._zoutput)


	def trace2(self, opticalelement1, opticalelement2, outputplane):
		"""propagates and plots the beam through 3 optical elements
		including an output plane"""
		self.create()
		for i in self._Beam:
			opticalelement1.propagate_ray(i)
			opticalelement2.propagate_ray(i)
			outputplane.propagate_ray(i)
			self._Beampoints.append(i.vertices())
		#self.plotpath()
		#self.spotplot(outputplane._zoutput)
	
	def trace2dispersed(self, opticalelement1, opticalelement2, outputplane):
		"""propagates and plots the beam through 3 optical elements
		including an output plane"""
		self.rainbow()
		for i in self._Beam:
			opticalelement1.propagate_ray(i)
			opticalelement2.propagate_ray(i)
			opticalelement1.propagate_ray(i)
			outputplane.propagate_ray(i)
			self._Beampoints.append(i.vertices())
		self.plotdispersed()
		#self.spotplot(outputplane._zoutput)	

	def plotpath(self):
		"""plot the beam's path (y vs. z)"""
		z_coords, y_coords = [], []
		for list in self._Beampoints:
			z, y = [], []
			for i in list:
				z.append(i[2]), y.append(i[1])
			z_coords.append(z), y_coords.append(y)		
		fig1 = plt.figure()
		for z,y in zip(z_coords, y_coords):
			plt.plot(z, y, self._ray.Raycolour())
		plt.title('Beam Path for beam of %s rays, beam radius: %s' %(self._n, self._r))
		plt.xlabel('z')
		plt.ylabel('y')
		plt.xlim(0, 250)
		fig1.show()
		fig1.savefig('r_%s_and_n_%s_path.png' %(self._r, self._n))

	def plotdispersed(self):
		"""plot the beam's path (y vs. z), each ray plotted in the colour of its wavelength"""
		wavelengths = [self._ray.Raycolour()]
		for Ray in self._Beam:
			wavelengths.append(Ray.Raycolour())
		z_coords, y_coords = [], []
		for list in self._Beampoints:
			z, y = [], []
			for i in list:
				z.append(i[2]), y.append(i[1])
			z_coords.append(z), y_coords.append(y)	
		fig3 = plt.figure()
		for z,y in zip(z_coords, y_coords):
			plt.plot(z, y, wavelengths[z_coords.index(z)])
		plt.title('Beam Path for beam of %s rays, beam radius: %s' %(self._n, self._r))
		plt.xlabel('z')
		plt.ylabel('y')
		plt.xlim(0, 100)
		fig3.savefig('r_%s_and_n_%s.png' %(self._r, self._n))

	def spotplot(self, z = 0):
		"""plot the x-y distribution of rays in the beam at various z coordinates 
		along the beam's path"""
		coords = []
		for list in self._Beampoints:
			for i in list:
				if i[2] == z:
					coords.append((i[0], i[1]))
		fig2 = plt.figure()
		for i in coords:
			plt.scatter(i[0], i[1], color = "Red")
		plt.title('Spot Diagram at z = %s for beam of %s rays, beam radius: %s' %(z, self._n, self._r))
		plt.xlabel('x')
		plt.ylabel('y')
		fig2.show()
		fig2.savefig('r_%s_and_n_%s_spot.png' %(self._r, self._n))

	def rms(self, outputplane):
		"""Calculates the rms spot deviation (geometrical focus)"""
		distances = []
		for list in self._Beampoints:
			for i in list:
				if i[2] == outputplane._zoutput:
					distances.append(np.sqrt(i[0]**2 + i[1]**2))
		return np.sqrt(sum((n**2 for n in distances))/len(distances))

	def diffractionlimit(self, opticalelement, outputplane):
		return ((outputplane._zoutput - opticalelement._z0)*self._wavelength)/(2*self._r)







	





		


