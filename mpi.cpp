#include "common.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <list>

std::list<particle_t> ghost_parts; 
std::list<particle_t> actual_parts;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Function to send the out particles to the right neighbor
void send_out_particles(std::list<particle_t>& parts, int num_parts, int rank_begin, int rank_end) {
    // total number of particles to send
    int num_send = 0;
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].x > size / rank_end) {
            num_send++;
        }
    }

    // Send the number of particles to the right neighbor
    MPI_Send(&num_send, 1, MPI_INT, rank_end, 0, MPI_COMM_WORLD);

    // Send the particles to the right neighbor
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].x > size / rank_end) {
            MPI_Send(&parts[i], 1, PARTICLE, rank_end, 0, MPI_COMM_WORLD);
        }
    }
}

// Function to receive the in particles from the left neighbor
void receive_in_particles(std::list<particle_t>& parts, int num_parts, int rank_begin) {
    // Receive the number of particles from the left neighbor
    int num_recv;
    MPI_Recv(&num_recv, 1, MPI_INT, rank_begin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Receive the particles from the left neighbor
    for (int i = 0; i < num_recv; i++) {
        particle_t p;
        MPI_Recv(&p, 1, PARTICLE, rank_begin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        parts.push_back(p);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // Create bins such that each bin is a row and the number of bins is equal to the number of processors
    std::vector<std::list<particle_t>> near_bounds;
    near_bounds.resize(num_procs);
    double width = size / num_procs;

    // Distribute particles to the current bin based on their x position
    // Collect particles that we will send to the right neighbor
    if (rank == 0){
        // Collect particles that we will send to the right neighbor
        std::vector<std::list<particle_t>> send_parts;
        send_parts.resize(num_procs);
        // clear the particles forces 
        for (int i = 0; i < num_parts; i++) {
            parts[i].ax = parts[i].ay = 0;
            // Distribute particles to the current bin based on their y position 
            int processor = parts[i].y / width;
            send_parts[processor].push_back(parts[i]);
            // Check to see if the bin that particle belongs to is the same as the processor's rank. If it is, add the particle to the bin
            if (processor == rank) {
                actual_parts.push_back(parts[i]);
            }
            //Check to see if the particle is within the cutoff distance from the boundary of the neighboring bin and the current bin
            if (processor == rank - 1 && abs(parts[i].y - processor * width) <= cutoff) {
               ghost_parts.push_back(parts[i]); 
            }
            if (processor == rank + 1 && abs(parts[i].y - (processor + 1) * width) <= cutoff) {
                ghost_parts.push_back(parts[i]);
            }
            // collect ghost particles that we will send to the right neighbor
            // check to see if the proessor is greater or equal to 1 and if the particle is within a certain cutoff distance from the top of its current bin 
            if (processor - 1 >= 0 && abs(parts[i].y - processor * width) <= cutoff) {
                near_bounds[processor - 1].push_back(parts[i]);
            }
            // check to see if the proessor is less than the number of processors and if the particle is within a certain cutoff distance from the bottom of its current bin
            if (processor + 1 < num_procs && abs(parts[i].y - (processor + 1) * width) <= cutoff) {
                near_bounds[processor + 1].push_back(parts[i]);
            }
        }
        // Send the particles to the right neighbor
        for (int i = 1; i < num_procs; i++){
            send_out_particles(send_parts[i], send_parts[i].size(), 0, i);
        }
    }
    else {
        // Receive the particles from the left neighbor
        receive_in_particles(actual_parts, near_bounds[rank].size(), rank - 1);
    }
    if (rank == 0){
        for(int i = 1; i <num_procs; i++){
            send_out_particles(near_bounds[i], near_bounds[i].size(), 0, i);
        }
    }
    else {
        receive_in_particles(ghost_parts, near_bounds[rank].size(), rank - 1);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
        // 
}