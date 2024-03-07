#include "common.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>

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
void send_out_particles(std::vector<particle_t>& parts, int num_parts, int rank_begin, int rank_end, double size) {
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
void receive_in_particles(std::vector<particle_t>& parts, int num_parts, int rank_begin, int rank_end, double size) {
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

    // get each row 
    double row_size = size / num_procs;
    std::vector<std::list<particle_t>> ghost_parts; 
    ghost_parts.resize(num_procs);

    // Each processor will handle the bins in each row. Just initialize, don't need to do anything else
    if (rank == 0) {
        // list of particles that will be sent out 
        std::vector<std::list<particle_t>> send_out_parts;
        send_out_parts.resize(num_procs);
        // find the bins that will be sent out
        for (int i = 0; i < num_parts; i++) {
            int bin_x = parts[i].x / row_size;
            if (bin_x == num_procs) {
                bin_x--;
            }
            if (bin_x != 0) {
                send_out_parts[bin_x - 1].push_back(parts[i]);
            }
        }
    }
}

// Assign particles to bins
void assign_particles_to_bins(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
}