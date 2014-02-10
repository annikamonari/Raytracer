import numpy as np
import matplotlib.pyplot as plt

"""

My module creating a 3D 
optical ray tracer

"""

class Ray:

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
		for item in self._points: print item

	def trace(self, SphericalRefraction, OutputPlane):
		SphericalRefraction.propagate_ray(self)
		OutputPlane.propagate_ray(self)
		self.plot()

	# a function for estimating the paralaxial focus using two rays
	def parallaxtrace(self, Ray, SphericalRefraction, OutputPlane):
		SphericalRefraction.propagate_ray(self)
		SphericalRefraction.propagate_ray(Ray)
		OutputPlane.propagate_ray(self)
		OutputPlane.propagate_ray(Ray)
		self.plotparallax(Ray)

	def parallaxtrace2(self, Ray, sr1, sr2, OutputPlane):
		sr1.propagate_ray(self)
		sr1.propagate_ray(Ray)
		sr2.propagate_ray(self)
		sr2.propagate_ray(Ray)
		OutputPlane.propagate_ray(self)
		OutputPlane.propagate_ray(Ray)
		self.plotparallax(Ray)

	def plot(self):
		z, y = [], []
		for i in self._points:
			z.append(i[2]), y.append(i[1])
		print z, y
		plt.plot(z, y, color = "Blue")
		plt.title('Beam Path')
		plt.xlabel('z')
		plt.ylabel('x')
		plt.show()

	def plotparallax(self, Ray):
		z, y, p, q = [], [], [], []
		for i in self._points:
			z.append(i[2]), y.append(i[1])
		for i in Ray._points:
			p.append(i[2]), q.append(i[1])
		print z, y
		print p, q
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
		if self.s == "plane":
			return np.array([0,0,-1])
		else:
			Q = self.intercept(ray)
			surface_normal = Q - self.centre
			return surface_normal/np.sqrt(sum(n**2 for n in surface_normal))

	def refract(self, ray):
		n_unit = -self.unitsurfacenormal(ray)
		k1 = ray.k() 
		ref = self.n1/self.n2
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
			print "Final Point: %s" %(ray.p()) + " and Final Direction: %s" %(ray.k())

class OutputPlane(OpticalElement):

	def __init__(self, z = 0.0):
		self.z_output = z

	def intercept(self, ray):
		l = (self.z_output - ray.p()[2]) / ray.k()[2]
		return ray.p() + l*ray.k()

	def propagate_ray(self, ray):
		p = self.intercept(ray)
		ray.append(p, None)
		return "Final Point: %s" %(ray.p()) + "and Final Direction: %s" %(ray.k())

class CollimatedBeam:

	def __init__(self, Ray, r = 10, n = 6):
		self.n = n
		self.r = r
		self.Ray = Ray
		self.x, self.y = self.Ray.p()[0], self.Ray.p()[1]
		self.Beam, self.Beam_points = [self.Ray], [self.Ray._points]
		self.d = self.raydensity()

	def __repr__(self):
		return "Collimated Beam of radius = %s, density = %s, centred at (%s,%s)" % (self.r,self.d,self.x,self.y)

	def create(self):
		for r in np.linspace(0, self.r, self.n, endpoint = True):
			for i in np.linspace(0, 2*np.pi, (2*np.pi*r*(self.n-1))/self.r, endpoint = True):
				self.Beam.append(Ray([self.x+r*np.cos(i), self.y + r*np.sin(i), 0], self.Ray.k()))
		return self.Beam

	def raydensity(self):
		return len(self.Beam)/(np.pi*(self.r**2))

	def trace(self, SphericalRefraction, OutputPlane):
		self.create()
		for i in self.Beam:
			SphericalRefraction.propagate_ray(i)
			OutputPlane.propagate_ray(i)
			self.Beam_points.append(i._points)
		self.plot()
		self.spotplot(OutputPlane.z_output)
		print 'Diffraction Limit = %s' %(((OutputPlane.z_output - SphericalRefraction.z0)*680*10**(-9))/(2*self.r))

	def trace2(self, sr1, sr2, OutputPlane):
		self.create()
		for i in self.Beam:
			sr1.propagate_ray(i)
			sr2.propagate_ray(i)
			OutputPlane.propagate_ray(i)
			self.Beam_points.append(i._points)
		self.plot()
		self.spotplot(OutputPlane.z_output)
		print 'Diffraction Limit = %s' %(((OutputPlane.z_output - sr1.z0)*680*10**(-9))/(2*self.r))

	def plot(self):
		z_coords, y_coords = [], []
		for list in self.Beam_points:
			z, y = [], []
			for i in list:
				z.append(i[2]), y.append(i[1])
			z_coords.append(z), y_coords.append(y)		
		for z,y in zip(z_coords, y_coords):
				plt.plot(z, y, color = "Blue")
		plt.title('Beam Path for beam of ray density: %s, beam radius: %s' %(self.d, self.r))
		plt.xlabel('z')
		plt.ylabel('y')
		plt.show()

	def spotplot(self, z = 0):
		coords, distances = [], []
		for list in self.Beam_points:
			for i in list:
				if i[2] == z:
					coords.append((i[0],i[1]))
		for i in coords:
			distances.append(np.sqrt(sum(n**2 for n in i)))
		print 'Rms deviation = %s' %(np.sqrt(sum((n**2 for n in distances))/len(coords)))
		plt.scatter(*zip(*coords), color = "Red")
		plt.title('Spot Diagram at z = %s for beam of ray density: %s, beam radius: %s' %(z, self.d, self.r))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()



	







		


