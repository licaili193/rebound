/**
 * @file 	integrator_sei_global.c
 * @brief 	Global Symplectic Epicycle Integrator (SEI).
 * @author 	Extended from original SEI by Hanno Rein <hanno@hanno-rein.de>
 * @details	This file implements a global version of the Symplectic Epicycle Integrator 
 * (SEI). Unlike the original SEI which uses a single OMEGA parameter for all particles,
 * this version calculates the epicyclic frequency for each particle based on its 
 * current distance from the central object. It also performs coordinate transformations
 * between global Cartesian coordinates and local shearing coordinates for each particle.
 * 
 * This allows simulating systems like Saturn's rings where particles at different
 * distances have different epicyclic frequencies.
 * 
 * @section 	LICENSE
 * Copyright (c) 2011 Hanno Rein, Shangfei Liu
 *
 * This file is part of rebound.
 *
 * rebound is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rebound is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rebound.h"
#include "particle.h"
#include "gravity.h"
#include "boundary.h"
#include "integrator.h"
#include "integrator_sei_global.h"

static void operator_H012_global(double dt, double OMEGA, double OMEGAZ, struct reb_particle* p);
static void operator_phi1_global(double dt, struct reb_particle* p);
static void global_to_local_coordinates(struct reb_particle* p, double* x_local, double* y_local, double* z_local, double* vx_local, double* vy_local, double* vz_local, double central_mass, double G);
static void local_to_global_coordinates(double x_local, double y_local, double z_local, double vx_local, double vy_local, double vz_local, struct reb_particle* p, double central_mass, double G);
static double calculate_omega(double r, double central_mass, double G);
static void apply_global_rotation(struct reb_particle* p, double OMEGA, double dt_half);

void reb_integrator_sei_global_init(struct reb_simulation* const r){
    /**
     * Initialize the SEI global integrator.
     * Set default central mass if not already set.
     */
    if (r->ri_sei_global.central_mass == 0.0) {
        r->ri_sei_global.central_mass = 1.0;  // Default central mass
    }
    r->ri_sei_global.lastdt = 0;
}

void reb_integrator_sei_global_part1(struct reb_simulation* const r){
    r->gravity_ignore_terms = 0;
    const int N = r->N;
    struct reb_particle* const particles = r->particles;
    
    if (N < 1) return; // Need at least 1 particle
    
    const double central_mass = r->ri_sei_global.central_mass;
    
    // Process all particles 
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++){
        // Convert global coordinates to local shearing coordinates
        double x_local, y_local, z_local, vx_local, vy_local, vz_local;
        global_to_local_coordinates(&particles[i], &x_local, &y_local, &z_local, 
                                   &vx_local, &vy_local, &vz_local, central_mass, r->G);
        
        // Calculate orbital frequency using local y coordinate (radial distance)
        double OMEGA = calculate_omega(y_local, central_mass, r->G);
        double OMEGAZ = OMEGA; // For Keplerian motion, vertical frequency equals radial frequency
        
        // Create a temporary particle with local coordinates
        struct reb_particle temp_p = {0};
        temp_p.x = x_local;
        temp_p.y = y_local;
        temp_p.z = z_local;
        temp_p.vx = vx_local;
        temp_p.vy = vy_local;
        temp_p.vz = vz_local;
        
        // Apply SEI operator H012 in local coordinates
        operator_H012_global(r->dt, OMEGA, OMEGAZ, &temp_p);
        
        // Convert back to global coordinates
        local_to_global_coordinates(temp_p.x, temp_p.y, temp_p.z, 
                                   temp_p.vx, temp_p.vy, temp_p.vz, 
                                   &particles[i], central_mass, r->G);
        
        // Apply global rotation of the local patch
        apply_global_rotation(&particles[i], OMEGA, r->dt / 2.0);
    }
    
    r->t += r->dt/2.;
}

void reb_integrator_sei_global_part2(struct reb_simulation* r){
    const int N = r->N;
    struct reb_particle* const particles = r->particles;
    
    if (N < 1) return;
    
    const double central_mass = r->ri_sei_global.central_mass;
    
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++){
        // Apply perturbation kick (phi1 operator) in global coordinates
        operator_phi1_global(r->dt, &particles[i]);
        
        // Convert to local coordinates
        double x_local, y_local, z_local, vx_local, vy_local, vz_local;
        global_to_local_coordinates(&particles[i], &x_local, &y_local, &z_local, 
                                   &vx_local, &vy_local, &vz_local, central_mass, r->G);
        
        // Calculate orbital frequency using local y coordinate (radial distance)
        double OMEGA = calculate_omega(y_local, central_mass, r->G);
        double OMEGAZ = OMEGA;
        
        // Create temporary particle with local coordinates
        struct reb_particle temp_p = {0};
        temp_p.x = x_local;
        temp_p.y = y_local;
        temp_p.z = z_local;
        temp_p.vx = vx_local;
        temp_p.vy = vy_local;
        temp_p.vz = vz_local;
        
        // Apply SEI operator H012 in local coordinates
        operator_H012_global(r->dt, OMEGA, OMEGAZ, &temp_p);
        
        // Convert back to global coordinates
        local_to_global_coordinates(temp_p.x, temp_p.y, temp_p.z, 
                                   temp_p.vx, temp_p.vy, temp_p.vz, 
                                   &particles[i], central_mass, r->G);
        
        // Apply global rotation of the local patch
        apply_global_rotation(&particles[i], OMEGA, r->dt / 2.0);
    }
    
    r->t += r->dt/2.;
    r->dt_last_done = r->dt;
}

void reb_integrator_sei_global_synchronize(struct reb_simulation* r){
    // Do nothing - particles are always synchronized in global coordinates
}

void reb_integrator_sei_global_reset(struct reb_simulation* r){
    r->ri_sei_global.lastdt = 0;	
}

/**
 * @brief Calculate orbital frequency for a particle at distance r from central mass
 * @param r Distance from central object
 * @param central_mass Mass of central object
 * @return Orbital frequency OMEGA
 */
static double calculate_omega(double r, double central_mass, double G){
    // For Keplerian motion: OMEGA = sqrt(GM/r^3)
    // NEGATIVE sign for counter-clockwise rotation (standard astronomical convention)
    if (r <= 1e-10) return -1e10;  // Avoid division by zero, set high frequency for very close particles
    return -sqrt(G * central_mass / (r*r*r));  // Negative for counter-clockwise motion
}

/**
 * @brief Transform from global Cartesian to local shearing coordinates
 * @param p Particle in global coordinates
 * @param x_local, y_local, z_local Local position coordinates (output)
 * @param vx_local, vy_local, vz_local Local velocity coordinates (output)  
 * @param central_mass Mass of central object
 */
static void global_to_local_coordinates(struct reb_particle* p, double* x_local, double* y_local, double* z_local,
                                       double* vx_local, double* vy_local, double* vz_local, double central_mass, double G){
    // Transform from global Cartesian to local shearing sheet coordinates
    // In SEI, the local patch is centered at the guiding center, so particles
    // are at the origin (0,0) in local coordinates with velocity perturbations
    
    // Calculate distance and angle from center for the guiding center
    double r = sqrt(p->x * p->x + p->y * p->y);
    
    if (r < 1e-10) {
        // Handle particles very close to center
        *x_local = 0.0;
        *y_local = r;  // Keep radial distance for OMEGA calculation
        *z_local = p->z;
        *vx_local = p->vx;
        *vy_local = p->vy;
        *vz_local = p->vz;
        return;
    }
    
    double theta = atan2(p->y, p->x);  // Azimuthal angle
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    
    // In local shearing sheet: particle is at origin of the local patch
    *x_local = 0.0;   // Azimuthal offset from guiding center
    *y_local = r;     // Radial distance (for OMEGA calculation)  
    *z_local = p->z;  // Vertical (small, preserved)
    
    // Transform velocities to local frame (subtract orbital motion)
    // Orbital velocity at radius r: v_orbital = sqrt(GM/r) in azimuthal direction
    double v_orbital = sqrt(G * central_mass / r);
    
    // Rotate velocities to local coordinate system
    double vx_rotated = -sin_theta * p->vx + cos_theta * p->vy;  // Azimuthal velocity
    double vy_rotated =  cos_theta * p->vx + sin_theta * p->vy;  // Radial velocity
    
    // Subtract the orbital motion to get perturbation velocities
    *vx_local = vx_rotated - v_orbital;  // Azimuthal perturbation velocity
    *vy_local = vy_rotated;              // Radial perturbation velocity
    *vz_local = p->vz;                   // Vertical velocity (unchanged)
}

/**
 * @brief Transform from local shearing coordinates back to global Cartesian
 * @param x_local, y_local, z_local Local position coordinates
 * @param vx_local, vy_local, vz_local Local velocity coordinates
 * @param p Particle to update with global coordinates
 * @param central_mass Mass of central object
 */
static void local_to_global_coordinates(double x_local, double y_local, double z_local, 
                                       double vx_local, double vy_local, double vz_local,
                                       struct reb_particle* p, double central_mass, double G){
    // Transform from local shearing sheet coordinates back to global Cartesian
    // The local patch was centered at the guiding center, so we need to add back
    // the global orbital motion and account for any rotation of the patch
    
    // Get the original guiding center position (before H012 integration)
    double r_orig = sqrt(p->x * p->x + p->y * p->y);
    
    if (r_orig < 1e-10) {
        // Handle particles very close to center
        p->x = x_local;
        p->y = y_local;  
        p->z = z_local;
        p->vx = vx_local;
        p->vy = vy_local;
        p->vz = vz_local;
        return;
    }
    
    double theta_orig = atan2(p->y, p->x);  // Original azimuthal angle
    
    // The guiding center may have rotated during the integration step
    // Calculate new orbital frequency at the new radial distance y_local
    double OMEGA = calculate_omega(y_local, central_mass, G);
    
    // Update the azimuthal angle: particles move with the local patch
    // Note: OMEGA is negative for counter-clockwise motion
    // We need to add the rotation during the timestep, but we don't have dt here
    // So we'll use the original orientation for now and let the integrator handle the rotation
    double theta_new = theta_orig;
    
    double cos_theta = cos(theta_new);
    double sin_theta = sin(theta_new);
    
    // Position in global coordinates: place at the guiding center + local offsets
    // For SEI, x_local and y_local represent small perturbations
    double r_new = y_local;  // New radial distance
    
    p->x = r_new * cos_theta + x_local * (-sin_theta);  // Add azimuthal perturbation
    p->y = r_new * sin_theta + x_local * cos_theta;     // Add azimuthal perturbation  
    p->z = z_local;                                     // Vertical unchanged
    
    // Velocity in global coordinates: add back orbital motion + rotated perturbations
    double v_orbital = sqrt(G * central_mass / r_new);  // Orbital velocity at new radius
    
    // Rotate perturbation velocities back to global frame
    double vx_global_pert = -sin_theta * vx_local + cos_theta * vy_local;
    double vy_global_pert =  cos_theta * vx_local + sin_theta * vy_local;
    
    // Add back the orbital motion
    p->vx = vx_global_pert + v_orbital * (-sin_theta);  // Add orbital velocity
    p->vy = vy_global_pert + v_orbital * cos_theta;     // Add orbital velocity
    p->vz = vz_local;                                   // Vertical unchanged
}

/**
 * @brief SEI H012 operator for a single particle using individual OMEGA values
 * @param dt Timestep
 * @param OMEGA Orbital frequency for this particle
 * @param OMEGAZ Vertical frequency for this particle
 * @param p Particle (in local coordinates)
 */
static void operator_H012_global(double dt, double OMEGA, double OMEGAZ, struct reb_particle* p){
    // Check for reasonable OMEGA values to avoid numerical issues
    if (fabs(OMEGA) <= 1e-10 || fabs(OMEGA) > 1e10) return;  // Skip if OMEGA magnitude is unreasonable
    if (fabs(OMEGAZ) <= 1e-10 || fabs(OMEGAZ) > 1e10) return;
    
    // Pre-calculate trigonometric functions
    double sindt = sin(OMEGA * (-dt/2.));
    double tandt = tan(OMEGA * (-dt/4.));
    double sindtz = sin(OMEGAZ * (-dt/2.));
    double tandtz = tan(OMEGAZ * (-dt/4.));

    // Integrate vertical motion (z-direction)
    const double zx = p->z * OMEGAZ;
    const double zy = p->vz;

    // Rotation implemented as 3 shear operators to avoid round-off errors
    const double zt1 = zx - tandtz * zy;			
    const double zyt = sindtz * zt1 + zy;
    const double zxt = zt1 - tandtz * zyt;	
    p->z = (fabs(OMEGAZ) > 1e-10) ? zxt / OMEGAZ : p->z;  // Avoid division by zero
    p->vz = zyt;

    // Integrate motion in xy directions (epicyclic motion)
    const double aO = 2. * p->vy + 4. * p->x * OMEGA;	// Center of epicyclic motion
    const double bO = p->y * OMEGA - 2. * p->vx;	

    const double ys = (p->y * OMEGA - bO) / 2.; 		// Epicycle vector
    const double xs = (p->x * OMEGA - aO); 

    // Rotation implemented as 3 shear operators to avoid round-off errors
    const double xst1 = xs - tandt * ys;			
    const double yst = sindt * xst1 + ys;
    const double xst = xst1 - tandt * yst;	

    p->x = (fabs(OMEGA) > 1e-10) ? (xst + aO) / OMEGA : p->x;  // Avoid division by zero			
    p->y = (fabs(OMEGA) > 1e-10) ? (yst * 2. + bO) / OMEGA - 3./4. * aO * dt : p->y;	
    p->vx = yst;
    p->vy = -xst * 2. - 3./2. * aO;
}

/**
 * @brief Apply acceleration due to perturbations (PHI1 term)
 * @param dt Timestep  
 * @param p Particle to update
 */
static void operator_phi1_global(double dt, struct reb_particle* p){
    // Apply external accelerations (from gravity calculations, etc.)
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->vz += p->az * dt;
}

/**
 * @brief Apply global rotation to particle due to rotating local patch
 * @param p Particle to rotate
 * @param OMEGA Orbital frequency 
 * @param dt_half Half timestep (dt/2)
 */
static void apply_global_rotation(struct reb_particle* p, double OMEGA, double dt_half){
    // The local shearing sheet itself rotates around the central mass
    double theta_rotation = OMEGA * dt_half;
    double cos_rot = cos(theta_rotation);
    double sin_rot = sin(theta_rotation);
    
    // Rotate position
    double x_rot = cos_rot * p->x - sin_rot * p->y;
    double y_rot = sin_rot * p->x + cos_rot * p->y;
    p->x = x_rot;
    p->y = y_rot;
    
    // Rotate velocity  
    double vx_rot = cos_rot * p->vx - sin_rot * p->vy;
    double vy_rot = sin_rot * p->vx + cos_rot * p->vy;
    p->vx = vx_rot;
    p->vy = vy_rot;
} 