#!/usr/bin/env python3
"""
Saturn Disk Simulation using SEI Global Integrator

This example demonstrates the new SEI Global integrator by simulating 
particles in Saturn's rings. Unlike the original SEI integrator which 
requires a single OMEGA parameter, SEI Global automatically calculates 
the appropriate epicyclic frequency for each particle based on its 
distance from the central object.

The central object (Saturn) is not included as a particle - its gravitational
effect is represented through the central_mass parameter in the integrator.
"""

import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    print("Saturn Disk Simulation with SEI Global Integrator")
    print("=" * 50)
    
    # Create simulation
    sim = rebound.Simulation()
    
    # Add ring particles at various distances (no central particle needed)
    N_particles = 20
    a_min, a_max = 1.0, 2.5  # Ring extends from 1 to 2.5 Saturn radii
    central_mass = 1.0  # Mass of Saturn
    
    print(f"Adding {N_particles} ring particles...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    for i in range(N_particles):
        # Semi-major axis distribution
        a = a_min + (a_max - a_min) * i / (N_particles - 1)
        
        # Small random eccentricity and inclination
        e = 0.02 * np.random.random()  # Slightly higher eccentricity for more interesting dynamics
        inc = 0.01 * np.random.random()  # Small inclinations
        
        # Random angles
        theta = 2 * np.pi * np.random.random()  # Mean anomaly
        omega = 2 * np.pi * np.random.random()  # Argument of periapsis
        Omega = 2 * np.pi * np.random.random()  # Longitude of ascending node
        
        # More accurate conversion from orbital elements to Cartesian
        # Using standard orbital mechanics formulas
        E = theta  # For small eccentricity, mean anomaly ≈ eccentric anomaly
        nu = E + e * np.sin(E)  # True anomaly (first order)
        
        # Distance from central body
        r = a * (1 - e * np.cos(E))
        
        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0.0
        
        # Velocity in orbital plane - use simple circular velocity
        # For circular orbits: v = sqrt(GM/r) tangentially
        v_circular = np.sqrt(central_mass / r)
        vx_orb = -v_circular * np.sin(nu)  # Tangential velocity
        vy_orb = v_circular * np.cos(nu)
        vz_orb = 0.0
        
        # Rotation matrices for inclination and longitude of ascending node
        cos_Omega = np.cos(Omega)
        sin_Omega = np.sin(Omega)
        cos_inc = np.cos(inc)
        sin_inc = np.sin(inc)
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        
        # Transform to 3D space
        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_inc) * x_orb + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_inc) * y_orb
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_inc) * x_orb + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_inc) * y_orb
        z = (sin_omega * sin_inc) * x_orb + (cos_omega * sin_inc) * y_orb
        
        vx = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_inc) * vx_orb + \
             (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_inc) * vy_orb
        vy = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_inc) * vx_orb + \
             (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_inc) * vy_orb
        vz = (sin_omega * sin_inc) * vx_orb + (cos_omega * sin_inc) * vy_orb
        
        # Add particle using Cartesian coordinates
        sim.add(m=1e-12, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    
    print(f"✓ Added {sim.N} particles")
    
    # Configure SEI Global integrator with gravity
    sim.integrator = "sei_global"
    boxsize = 200.            # [m]
    sim.configure_box(boxsize)
    sim.N_ghost_x = 2
    sim.N_ghost_y = 2
    sim.G = 6.67428e-11
    sim.softening = 0.2
    sim.ri_sei_global.central_mass = central_mass
    sim.gravity    = "tree"
    sim.collision  = "tree"
    sim.collision_resolve = "hardsphere"
    sim.boundary   = "none"
    sim.dt = 0.005  # Smaller timestep for better accuracy
    
    print(f"✓ Set integrator to sei_global with gravity = {sim.gravity}")
    print(f"✓ Central mass parameter = {sim.ri_sei_global.central_mass}")
    
    # Move to center of mass frame
    sim.move_to_com()
    
    # Calculate some physical properties
    total_kinetic = sum(0.5 * p.m * (p.vx**2 + p.vy**2 + p.vz**2) for p in sim.particles)
    print(f"Initial kinetic energy: {total_kinetic:.10f}")
    
    # Integration parameters  
    t_max = 5.0  # Shorter integration for better visualization
    N_outputs = 200
    times = np.linspace(0, t_max, N_outputs)
    
    # Storage for results
    all_positions = []
    all_times = []
    
    print(f"\nIntegrating for {t_max} time units...")
    
    # Integration loop
    for i, t in enumerate(times):
        sim.integrate(t)
        
        # Store all particle positions and time
        positions = []
        for p in sim.particles:
            positions.append([p.x, p.y, p.z])
        all_positions.append(positions)
        all_times.append(t)
        
        # Progress indicator
        if i % 40 == 0:
            progress = 100 * i / len(times)
            print(f"  Progress: {progress:5.1f}%")
    
    print(f"\nSimulation completed!")
    print(f"Integrated {len(all_positions)} timesteps")
    
    # Create comprehensive plots (disable animation for now)
    create_trajectory_plots(sim, all_times, all_positions, central_mass)
    
    return sim, all_times, all_positions

def create_trajectory_plots(sim, times, all_positions, central_mass):
    """Create comprehensive visualization plots of the simulation results."""
    
    print("Creating trajectory plots...")
    
    # Convert positions to numpy arrays for easier manipulation
    positions = np.array(all_positions)  # Shape: (time_steps, particles, 3)
    N_particles = positions.shape[1]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: XY trajectory view with all orbits
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, N_particles))
    
    for i in range(N_particles):
        x_traj = positions[:, i, 0]
        y_traj = positions[:, i, 1]
        ax1.plot(x_traj, y_traj, color=colors[i], alpha=0.7, linewidth=1)
        # Mark starting position
        ax1.scatter(x_traj[0], y_traj[0], color=colors[i], s=30, marker='o', zorder=5)
        # Mark final position
        ax1.scatter(x_traj[-1], y_traj[-1], color=colors[i], s=30, marker='s', zorder=5)
    
    # Add Saturn at center
    saturn_circle = plt.Circle((0, 0), 0.05, color='gold', alpha=0.9, zorder=10)
    ax1.add_patch(saturn_circle)
    ax1.text(0, -0.15, 'Saturn', ha='center', fontsize=8, weight='bold')
    
    ax1.set_xlabel('X (Saturn radii)')
    ax1.set_ylabel('Y (Saturn radii)')
    ax1.set_title('Ring Particle Trajectories (XY View)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: XZ trajectory view (side view)
    ax2 = plt.subplot(2, 3, 2)
    for i in range(N_particles):
        x_traj = positions[:, i, 0]
        z_traj = positions[:, i, 2]
        ax2.plot(x_traj, z_traj, color=colors[i], alpha=0.7, linewidth=1)
        ax2.scatter(x_traj[0], z_traj[0], color=colors[i], s=20, marker='o')
        ax2.scatter(x_traj[-1], z_traj[-1], color=colors[i], s=20, marker='s')
    
    ax2.axhline(y=0, color='gold', linewidth=2, alpha=0.8, label='Saturn orbital plane')
    ax2.set_xlabel('X (Saturn radii)')
    ax2.set_ylabel('Z (Saturn radii)')
    ax2.set_title('Side View (XZ Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Orbital radii evolution
    ax3 = plt.subplot(2, 3, 3)
    for i in range(N_particles):
        radii = np.sqrt(positions[:, i, 0]**2 + positions[:, i, 1]**2 + positions[:, i, 2]**2)
        ax3.plot(times, radii, color=colors[i], alpha=0.8, linewidth=1)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Distance from Saturn (radii)')
    ax3.set_title('Orbital Radii Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final positions colored by distance
    ax4 = plt.subplot(2, 3, 4)
    final_x = positions[-1, :, 0]
    final_y = positions[-1, :, 1]
    final_r = np.sqrt(final_x**2 + final_y**2 + positions[-1, :, 2]**2)
    
    scatter = ax4.scatter(final_x, final_y, c=final_r, s=60, alpha=0.8, cmap='plasma')
    plt.colorbar(scatter, ax=ax4, label='Distance (Saturn radii)')
    
    # Add concentric circles to show ring zones
    for radius in [1.0, 1.5, 2.0, 2.5]:
        circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.4, linestyle='--')
        ax4.add_patch(circle)
    
    saturn_circle = plt.Circle((0, 0), 0.05, color='gold', alpha=0.9)
    ax4.add_patch(saturn_circle)
    
    ax4.set_xlabel('X (Saturn radii)')
    ax4.set_ylabel('Y (Saturn radii)')
    ax4.set_title('Final Ring Structure')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Particle velocities
    ax5 = plt.subplot(2, 3, 5)
    for i in range(N_particles):
        # Calculate velocity magnitudes over time (not stored, so use final velocities)
        pass
    
    # Show orbital periods vs distance
    final_distances = np.sqrt(positions[-1, :, 0]**2 + positions[-1, :, 1]**2)
    theoretical_periods = 2 * np.pi * np.sqrt(final_distances**3 / central_mass)
    
    ax5.scatter(final_distances, theoretical_periods, c=colors, s=50, alpha=0.8)
    ax5.set_xlabel('Orbital Distance (Saturn radii)')
    ax5.set_ylabel('Theoretical Orbital Period')
    ax5.set_title('Kepler\'s Third Law')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: 3D trajectory view
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    
    # Plot a subset of trajectories for clarity
    step = max(1, N_particles // 8)  # Show at most 8 trajectories
    for i in range(0, N_particles, step):
        x_traj = positions[:, i, 0]
        y_traj = positions[:, i, 1]
        z_traj = positions[:, i, 2]
        ax6.plot(x_traj, y_traj, z_traj, color=colors[i], alpha=0.8, linewidth=1.5)
        ax6.scatter(x_traj[0], y_traj[0], z_traj[0], color=colors[i], s=40, marker='o')
    
    # Add Saturn
    ax6.scatter([0], [0], [0], color='gold', s=100, marker='*', label='Saturn')
    
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.set_title('3D Trajectory View')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('saturn_ring_trajectories.png', dpi=200, bbox_inches='tight')
    print(f"Trajectory plots saved as 'saturn_ring_trajectories.png'")
    
    # Skip animation for now to avoid Figure 2 issue
    create_animation(times, positions, central_mass)
    
    plt.show()

def create_animation(times, positions, central_mass):
    """Create an animation showing the evolution of the ring system."""
    
    print("Creating animation...")
    
    # Setup animation figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (Saturn radii)')
    ax.set_ylabel('Y (Saturn radii)')
    ax.set_title('Saturn Ring System Evolution')
    
    # Add Saturn
    saturn = plt.Circle((0, 0), 0.05, color='gold', alpha=0.9)
    ax.add_patch(saturn)
    
    # Add reference circles
    for radius in [1.0, 1.5, 2.0, 2.5]:
        circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.2, linestyle=':')
        ax.add_patch(circle)
    
    N_particles = positions.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, N_particles))
    
    # Initialize particle scatter plot with first frame data
    initial_positions = positions[0]
    scat = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], c=colors, s=50, alpha=0.8)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        # Update particle positions
        current_positions = positions[frame]
        scat.set_offsets(current_positions[:, :2])  # Only X and Y
        time_text.set_text(f'Time: {times[frame]:.2f}')
        return scat, time_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True, repeat=True)
    
    # Save animation
    try:
        anim.save('saturn_ring_animation.gif', writer='pillow', fps=20, dpi=100)
        print(f"Animation saved as 'saturn_ring_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("  (Make sure pillow is installed: pip install pillow)")
    
    return anim

if __name__ == "__main__":
    try:
        sim, times, positions = main()
        print(f"\n Saturn disk simulation completed successfully!")
        print(f"   Used SEI Global integrator with {sim.N} particles")
        print(f"   Generated {len(times)} trajectory points")
        print(f"   Files created:")
        print(f"   - saturn_ring_trajectories.png (comprehensive plots)")
        print(f"   (Animation disabled to avoid Figure 2 popup)")
        
    except Exception as e:
        print(f"\n Simulation failed: {e}")
        import traceback
        traceback.print_exc() 