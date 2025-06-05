import ctypes

class IntegratorSEIGlobal(ctypes.Structure):
    """
    This class is an abstraction of the C-struct reb_integrator_sei.
    It controls the behaviour of the global symplectic SEI integrator for 
    simulating particles around a central object (like Saturn's rings).
    
    Unlike the original SEI integrator, this version automatically calculates
    the epicyclic frequency for each particle based on its distance from the
    central object and performs coordinate transformations between global 
    Cartesian and local shearing coordinates.

    :ivar float central_mass: 
        Mass of the central object. This is used to calculate individual 
        epicyclic frequencies for each particle. Must be set by user.
        Default: 1.0
        
    Example usage:
    
         >>> sim = rebound.Simulation()
     >>> sim.add(m=1e-6, a=1.0)   # Ring particle (central object not needed as particle)
     >>> sim.add(m=1e-6, a=1.5)   # Another ring particle  
     >>> sim.integrator = "sei_global"
     >>> sim.ri_sei_global.central_mass = 1.0  # Set central mass parameter
     >>> sim.integrate(100)
    
    For more details on the SEI integrator see Rein & Spiegel (2015).
    """
    
    # This struct contains the central mass parameter for epicyclic frequency calculation
    _fields_ = [
        ("central_mass", ctypes.c_double),  # Mass of central object
        ("lastdt", ctypes.c_double)         # Internal: cached timestep
    ]
    
    def __repr__(self):
        return '<{0}.{1} object at {2}>'.format(self.__module__, type(self).__name__, hex(id(self))) 