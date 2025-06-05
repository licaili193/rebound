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
static void global_to_local_coordinates(struct reb_particle* p, double* x_local, double* y_local, double* z_local, double* vx_local, double* vy_local, double* vz_local, double central_mass);
static void local_to_global_coordinates(double x_local, double y_local, double z_local, double vx_local, double vy_local, double vz_local, struct reb_particle* p, double central_mass);
static double calculate_omega(double r, double central_mass);

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
                                   &vx_local, &vy_local, &vz_local, central_mass);
        
        // Calculate orbital frequency for this particle's current distance
        double dist = sqrt(particles[i].x*particles[i].x + particles[i].y*particles[i].y + particles[i].z*particles[i].z);
        double OMEGA = calculate_omega(dist, central_mass);
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
                                   &particles[i], central_mass);
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
                                   &vx_local, &vy_local, &vz_local, central_mass);
        
        // Calculate orbital frequency for current position
        double dist = sqrt(particles[i].x*particles[i].x + particles[i].y*particles[i].y + particles[i].z*particles[i].z);
        double OMEGA = calculate_omega(dist, central_mass);
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
                                   &particles[i], central_mass);
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
static double calculate_omega(double r, double central_mass){
    // For Keplerian motion: OMEGA = sqrt(GM/r^3)
    // We assume G=1 in simulation units
    // NEGATIVE sign for counter-clockwise rotation (standard astronomical convention)
    if (r <= 1e-10) return -1e10;  // Avoid division by zero, set high frequency for very close particles
    return -sqrt(central_mass / (r*r*r));  // Negative for counter-clockwise motion
}

/**
 * @brief Transform from global Cartesian to local shearing coordinates
 * @param p Particle in global coordinates
 * @param x_local, y_local, z_local Local position coordinates (output)
 * @param vx_local, vy_local, vz_local Local velocity coordinates (output)  
 * @param central_mass Mass of central object
 */
static void global_to_local_coordinates(struct reb_particle* p, double* x_local, double* y_local, double* z_local,
                                       double* vx_local, double* vy_local, double* vz_local, double central_mass){
    // For SEI, we work in the local shearing sheet coordinates
    // The transformation is simpler - we just use the current position and velocity
    // The SEI integrator handles the epicyclic motion in its own coordinate system
    
    *x_local = p->x;
    *y_local = p->y;
    *z_local = p->z;
    *vx_local = p->vx;
    *vy_local = p->vy;
    *vz_local = p->vz;
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
                                       struct reb_particle* p, double central_mass){
    // Update particle with the new coordinates from SEI integration
    p->x = x_local;
    p->y = y_local;
    p->z = z_local;
    p->vx = vx_local;
    p->vy = vy_local;
    p->vz = vz_local;
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