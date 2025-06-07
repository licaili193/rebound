#!/usr/bin/env python3
"""
Real Saturn Ring Simulation using SEI Global Integrator

This example simulates Saturn's rings using realistic physical parameters and scales.
Based on actual Saturn ring data from NASA and Cassini observations.

Physical Parameters:
- Saturn mass: 5.68 × 10²⁶ kg  
- Saturn equatorial radius: 60,268 km
- Ring distances: A-ring (122,170-136,780 km), B-ring (92,000-117,580 km)
- Particle sizes: 1-4 m radius with power law distribution
- Surface density: ~40-100 g/cm²
- G = 6.67428 × 10⁻¹¹ m³/kg/s²
"""

import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def powerlaw(slope, min_v, max_v):
    """Generate random numbers from power law distribution."""
    y = np.random.uniform()
    pow_max = pow(max_v, slope+1.)
    pow_min = pow(min_v, slope+1.)
    return pow((pow_max-pow_min)*y + pow_min, 1./(slope+1.))

def calculate_total_energy(sim, central_mass):
    """
    Calculate total energy including central mass potential.
    
    The sim.energy() method only considers particle-particle interactions,
    but our central mass is handled through the integrator parameter.
    According to REBOUND diagnostics documentation, we need to manually
    add the gravitational potential energy from the central mass.
    """
    # Kinetic energy and inter-particle potential energy
    E_particles = sim.energy()
    
    # Central mass potential energy: -G * M_central * m / r for each particle
    E_central = 0.0
    for particle in sim.particles:
        r = np.sqrt(particle.x**2 + particle.y**2 + particle.z**2)
        if r > 1e-10:  # Avoid division by zero
            E_central += -sim.G * central_mass * particle.m / r
    
    return E_particles + E_central

def main():
    print("Real Saturn Ring Simulation with SEI Global Integrator")
    print("=" * 55)
    
    # Create simulation
    sim = rebound.Simulation()
    
    # Physical constants (SI units)
    G = 6.67428e-11          # Gravitational constant [m³/kg/s²]
    M_saturn = 5.68e26       # Saturn mass [kg] 
    R_saturn = 60268e3       # Saturn equatorial radius [m]
    
    # Ring parameters based on Saturn's A-ring
    ring_inner = 122170e3    # A-ring inner edge [m]
    ring_outer = 136780e3    # A-ring outer edge [m]
    
    # Particle properties
    particle_density = 400.0 # Ring particle density [kg/m³]
    surface_density = 60.0   # Ring surface density [kg/m²] (A-ring typical)
    
    # Target orbital frequency for reference (at ~130,000 km)
    ref_distance = 130000e3  # Reference distance [m]
    ref_omega = np.sqrt(G * M_saturn / ref_distance**3)  # [rad/s]
    
    print(f"Physical Parameters:")
    print(f"  Saturn mass: {M_saturn:.2e} kg")
    print(f"  Saturn radius: {R_saturn/1000:.0f} km")
    print(f"  Ring range: {ring_inner/1000:.0f} - {ring_outer/1000:.0f} km")
    print(f"  Reference orbital frequency: {ref_omega:.2e} rad/s")
    print(f"  Reference orbital period: {2*np.pi/ref_omega/3600:.2f} hours")
    
    # Set simulation units (scale to more manageable numbers)
    # Use Saturn radius as length unit, Saturn mass as mass unit
    length_unit = R_saturn     # 1 unit = Saturn radius
    mass_unit = M_saturn       # 1 unit = Saturn mass
    time_unit = 1.0 / ref_omega  # 1 unit = reference orbital period
    
    print(f"\nSimulation Units:")
    print(f"  Length unit: {length_unit/1000:.0f} km")
    print(f"  Mass unit: {mass_unit:.2e} kg") 
    print(f"  Time unit: {time_unit/3600:.2f} hours")
    
    # Convert to simulation units
    sim.G = 1.0  # G*M_saturn*time_unit²/length_unit³ = 1
    central_mass = 1.0  # Saturn mass in units
    ring_inner_sim = ring_inner / length_unit
    ring_outer_sim = ring_outer / length_unit
    
    print(f"  Ring range (sim units): {ring_inner_sim:.3f} - {ring_outer_sim:.3f}")
    
    # Generate particles
    N_particles = 50  # More particles for realistic simulation
    print(f"\nGenerating {N_particles} ring particles...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    total_mass = 0.0
    
    for i in range(N_particles):
        # Random position in A-ring
        r = ring_inner_sim + (ring_outer_sim - ring_inner_sim) * np.random.random()
        theta = 2 * np.pi * np.random.random()
        
        # Convert to Cartesian coordinates  
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0.000002 * np.random.normal()  # Small vertical displacement
        
        # Circular orbital velocity (Keplerian)
        v_circ = np.sqrt(central_mass / r)  # In simulation units
        vx = -v_circ * np.sin(theta)
        vy = v_circ * np.cos(theta)
        vz = 0.0
        
        # Add small random velocity perturbations (typical ~1-10 m/s)
        v_random = 10.0 / (length_unit/time_unit)  # Convert 10 m/s to sim units
        vx += v_random * 0.1 * np.random.normal()
        vy += v_random * 0.1 * np.random.normal()
        
        # Particle size and mass using power law distribution
        radius_m = powerlaw(slope=-3, min_v=1.0, max_v=4.0)  # 1-4 meter radius
        volume = (4./3.) * np.pi * radius_m**3
        mass_kg = particle_density * volume
        
        # Convert to simulation units
        mass_sim = mass_kg / mass_unit
        radius_sim = radius_m / length_unit
        
        # Add particle
        sim.add(m=mass_sim, r=radius_sim, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        total_mass += mass_sim
    
    print(f"✓ Added {sim.N} particles")
    print(f"  Total particle mass: {total_mass:.2e} simulation units")
    print(f"  Total particle mass: {total_mass * mass_unit:.2e} kg")
    print(f"  Average particle radius: {np.mean([p.r for p in sim.particles]) * length_unit:.1f} m")
    
    # Configure SEI Global integrator
    sim.integrator = "sei_global"
    sim.ri_sei_global.central_mass = central_mass
    
    # Collision detection for ring particles
    sim.collision = "tree"
    sim.collision_resolve = "hardsphere"
    sim.gravity    = "tree"
    
    # Coefficient of restitution (Bridges et al. formula for ring particles)
    def cor_bridges(rel_velocity_magnitude):
        """Velocity-dependent coefficient of restitution for ring particles."""
        # Convert to m/s for the formula
        v_ms = rel_velocity_magnitude * (length_unit / time_unit)
        eps = 0.32 * pow(abs(v_ms)*100., -0.234)
        eps = max(0.0, min(1.0, eps))  # Clamp between 0 and 1
        return eps
    
    # Set timestep (much smaller than orbital period)
    orbital_period = 2 * np.pi  # In simulation time units  
    sim.dt = orbital_period / 1000.0  # 1/1000 of orbital period
    
    print(f"\nSimulation Configuration:")
    print(f"  Integrator: sei_global")
    print(f"  Central mass: {sim.ri_sei_global.central_mass}")
    print(f"  Timestep: {sim.dt:.6f} time units ({sim.dt * time_unit:.1f} s)")
    print(f"  Collision detection: {sim.collision}")
    
    # Move to center of mass frame (though should be close already)
    sim.move_to_com()
    
    # Calculate initial energy (including central mass potential)
    E_initial = calculate_total_energy(sim, central_mass)
    print(f"  Initial total energy: {E_initial:.6e}")
    
    # Integration parameters - much shorter for real-time scales
    t_max = 3.0 * orbital_period  # 3 orbital periods
    N_outputs = 300
    times = np.linspace(0, t_max, N_outputs)
    
    real_time_max = t_max * time_unit / 3600  # Convert to hours
    print(f"\nIntegrating for {t_max:.1f} time units ({real_time_max:.2f} hours)...")
    
    # Storage for results
    all_positions = []
    all_velocities = []
    all_times = []
    energies = []
    
    # Integration loop with progress tracking
    for i, t in enumerate(times):
        sim.integrate(t)
        
        # Store particle data
        positions = []
        velocities = []
        for p in sim.particles:
            positions.append([p.x, p.y, p.z])
            velocities.append([p.vx, p.vy, p.vz])
        
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_times.append(t)
        
        # Monitor energy conservation (including central mass potential)
        E = calculate_total_energy(sim, central_mass)
        energies.append(E)
        
        # Progress indicator
        if i % (N_outputs // 10) == 0:
            progress = 100 * i / len(times)
            dE = abs((E - E_initial) / E_initial) if E_initial != 0 else 0
            real_time = t * time_unit / 3600
            print(f"  Progress: {progress:5.1f}% | t = {real_time:6.2f} hrs | ΔE/E = {dE:.2e}")
    
    print(f"\nSimulation completed successfully!")
    print(f"  Final energy: {energies[-1]:.6e}")
    print(f"  Energy conservation: ΔE/E = {abs((energies[-1] - E_initial)/E_initial):.2e}")
    print(f"  Note: Energy calculation includes central mass potential energy")
    
    # Create comprehensive plots
    create_realistic_plots(sim, all_times, all_positions, all_velocities, energies, 
                          length_unit, time_unit, mass_unit, M_saturn, R_saturn)
    
    return sim, all_times, all_positions

def create_realistic_plots(sim, times, all_positions, all_velocities, energies,
                          length_unit, time_unit, mass_unit, M_saturn, R_saturn):
    """Create realistic visualization plots with proper units."""
    
    print("Creating realistic Saturn ring visualization...")
    
    # Convert to physical units for plotting
    positions = np.array(all_positions) * length_unit / 1000  # Convert to km
    times_hours = np.array(times) * time_unit / 3600  # Convert to hours
    N_particles = positions.shape[1]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Ring structure view (XY plane)
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.plasma(np.linspace(0, 1, N_particles))
    
    # Plot final positions with size reflecting particle mass
    final_x = positions[-1, :, 0]
    final_y = positions[-1, :, 1]
    final_r = np.sqrt(final_x**2 + final_y**2)
    
    # Scale marker sizes by particle mass (convert back to reasonable range)
    particle_masses = [p.m * mass_unit for p in sim.particles]  # Convert to kg
    marker_sizes = 20 + 100 * np.array(particle_masses) / np.max(particle_masses)  # Scale 20-120
    
    scatter = ax1.scatter(final_x, final_y, c=final_r, s=marker_sizes, alpha=0.8, cmap='plasma')
    plt.colorbar(scatter, ax=ax1, label='Distance from Saturn (km)')
    
    # Add Saturn and ring boundaries
    saturn_circle = plt.Circle((0, 0), R_saturn/1000, color='gold', alpha=0.9, zorder=10)
    ax1.add_patch(saturn_circle)
    
    # Add ring boundary circles
    ring_inner_km = 122170
    ring_outer_km = 136780
    inner_circle = plt.Circle((0, 0), ring_inner_km, fill=False, color='white', 
                             alpha=0.6, linestyle='--', linewidth=2)
    outer_circle = plt.Circle((0, 0), ring_outer_km, fill=False, color='white', 
                             alpha=0.6, linestyle='--', linewidth=2)
    ax1.add_patch(inner_circle)
    ax1.add_patch(outer_circle)
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_title('Saturn A-Ring Final Structure\n(Real Scale - Marker size ∝ particle mass)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-150000, 150000)
    ax1.set_ylim(-150000, 150000)
    
    # Plot 2: Particle trajectories (subset)
    ax2 = plt.subplot(2, 3, 2)
    step = max(1, N_particles // 8)  # Show subset of trajectories
    
    for i in range(0, N_particles, step):
        x_traj = positions[:, i, 0]
        y_traj = positions[:, i, 1]
        ax2.plot(x_traj, y_traj, color=colors[i], alpha=0.7, linewidth=1)
        # Use particle mass for marker size
        particle_size = marker_sizes[i] if i < len(marker_sizes) else 30
        ax2.scatter(x_traj[0], y_traj[0], color=colors[i], s=particle_size, marker='o', zorder=5)
        ax2.scatter(x_traj[-1], y_traj[-1], color=colors[i], s=particle_size, marker='s', zorder=5)
    
    saturn_circle2 = plt.Circle((0, 0), R_saturn/1000, color='gold', alpha=0.9)
    ax2.add_patch(saturn_circle2)
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('Particle Trajectories')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-150000, 150000)
    ax2.set_ylim(-150000, 150000)
    
    # Plot 3: Radial distance evolution
    ax3 = plt.subplot(2, 3, 3)
    for i in range(0, N_particles, step):
        radii = np.sqrt(positions[:, i, 0]**2 + positions[:, i, 1]**2 + positions[:, i, 2]**2)
        ax3.plot(times_hours, radii, color=colors[i], alpha=0.8, linewidth=1)
    
    ax3.axhline(y=ring_inner_km, color='red', linestyle='--', alpha=0.7, label='A-ring inner edge')
    ax3.axhline(y=ring_outer_km, color='red', linestyle='--', alpha=0.7, label='A-ring outer edge')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Distance from Saturn (km)')
    ax3.set_title('Orbital Radii Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Energy conservation
    ax4 = plt.subplot(2, 3, 4)
    if len(energies) > 1:
        energy_relative = [(E - energies[0])/energies[0] for E in energies]
        ax4.plot(times_hours, energy_relative, 'b-', linewidth=2)
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Relative Energy Change')
        ax4.set_title('Energy Conservation')
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 5: Orbital periods vs Kepler's 3rd law
    ax5 = plt.subplot(2, 3, 5)
    final_distances = np.sqrt(positions[-1, :, 0]**2 + positions[-1, :, 1]**2)
    # Convert distances to meters for calculation
    final_distances_m = final_distances * 1000
    
    # Theoretical orbital periods (Kepler's 3rd law)
    G = 6.67428e-11
    theoretical_periods_s = 2 * np.pi * np.sqrt(final_distances_m**3 / (G * M_saturn))
    theoretical_periods_h = theoretical_periods_s / 3600
    
    ax5.scatter(final_distances, theoretical_periods_h, c=colors[:N_particles], s=marker_sizes, alpha=0.8)
    ax5.set_xlabel('Orbital Distance (km)')
    ax5.set_ylabel('Orbital Period (hours)')
    ax5.set_title('Kepler\'s Third Law')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Ring cross-section (side view)
    ax6 = plt.subplot(2, 3, 6)
    final_x = positions[-1, :, 0]
    final_z = positions[-1, :, 2]
    final_r = np.sqrt(positions[-1, :, 0]**2 + positions[-1, :, 1]**2)
    
    scatter6 = ax6.scatter(final_r, final_z, c=colors[:N_particles], s=marker_sizes, alpha=0.8)
    ax6.axhline(y=0, color='gold', linewidth=3, alpha=0.8, label='Saturn equatorial plane')
    ax6.axvline(x=ring_inner_km, color='red', linestyle='--', alpha=0.7)
    ax6.axvline(x=ring_outer_km, color='red', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Distance from Saturn (km)')
    ax6.set_ylabel('Height above ring plane (km)')
    ax6.set_title('Ring Vertical Structure')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('real_saturn_ring_simulation.png', dpi=200, bbox_inches='tight')
    print(f"Realistic Saturn ring plots saved as 'real_saturn_ring_simulation.png'")
    
    # Create animation of ring evolution
    try:
        create_ring_animation(times_hours, positions, R_saturn, ring_inner_km, ring_outer_km, sim)
    except Exception as e:
        print(f"Could not create animation: {e}")
    
    plt.show()

def create_ring_animation(times, positions, R_saturn, ring_inner, ring_outer, sim=None):
    """Create animation of Saturn ring evolution."""
    
    print("Creating ring evolution animation...")
    
    # Setup animation figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-150000, 150000)
    ax.set_ylim(-150000, 150000)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Real Saturn A-Ring Evolution')
    
    # Add Saturn
    saturn = plt.Circle((0, 0), R_saturn/1000, color='gold', alpha=0.9)
    ax.add_patch(saturn)
    
    # Add ring boundaries
    inner_circle = plt.Circle((0, 0), ring_inner, fill=False, color='white', 
                             alpha=0.4, linestyle='--')
    outer_circle = plt.Circle((0, 0), ring_outer, fill=False, color='white', 
                             alpha=0.4, linestyle='--')
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)
    
    N_particles = positions.shape[1]
    colors = plt.cm.plasma(np.linspace(0, 1, N_particles))
    
    # Initialize scatter plot with particle sizes
    if sim is not None:
        # Calculate marker sizes for animation based on particle masses
        particle_masses = [p.m for p in sim.particles]
        anim_marker_sizes = 10 + 30 * np.array(particle_masses) / np.max(particle_masses)  # Scale 10-40 for animation
    else:
        anim_marker_sizes = 20  # Default size if no sim object
    
    scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], c=colors, s=anim_marker_sizes, alpha=0.8)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        # Update particle positions
        current_positions = positions[frame]
        scat.set_offsets(current_positions[:, :2])  # Only X and Y
        time_text.set_text(f'Time: {times[frame]:.2f} hours')
        return scat, time_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True, repeat=True)
    
    # Save animation
    try:
        anim.save('real_saturn_ring_animation.gif', writer='pillow', fps=20, dpi=100)
        print(f"Ring evolution animation saved as 'real_saturn_ring_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    return anim

if __name__ == "__main__":
    try:
        sim, times, positions = main()
        print(f"\nReal Saturn ring simulation completed successfully!")
        print(f"   Used realistic physical parameters")
        print(f"   Simulated {sim.N} ring particles in Saturn's A-ring")
        print(f"   Generated {len(times)} trajectory points over {times[-1]:.1f} time units")
        print(f"   Files created:")
        print(f"   - real_saturn_ring_simulation.png (comprehensive plots)")
        print(f"   - real_saturn_ring_animation.gif (ring evolution)")
        
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        import traceback
        traceback.print_exc() 