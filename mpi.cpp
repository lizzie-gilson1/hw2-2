#include "common.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>

// Define MPI Particle Type
// MPI_Datatype PARTICLE;

// Put any static global variables here that you will use throughout the simulation.
// constexpr int MAX_PARTICLES = 1000;

// Define a struct for linked list node
struct ListNode {
    int index;
    ListNode* next;
    ListNode(int idx) : index(idx), next(nullptr) {}
};

// Global declaration of bins
// std::array<std::array<ListNode*, MAX_PARTICLES>, MAX_PARTICLES> bins; // Use fixed-size array instead of vector
std::vector<std::vector<ListNode*>> bins;
int binCountX, binCountY; // Number of bins in each dimension
int binSize;

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

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here


    // binSize = std::max(static_cast<double>(std::ceil(cutoff * 0.5)), 1.0);
    binSize = std::max(static_cast<int>(std::ceil(cutoff * 0.5)), 1);
    binCountX = std::ceil(size / binSize);
    binCountY = std::ceil(size / binSize);
    // Adjust bin counts if necessary to ensure all particles are covered
    if (binCountX * binSize < size) {
        binCountX++;
    }
    if (binCountY * binSize < size) {
        binCountY++;
    }
    bins.resize(binCountX, std::vector<ListNode*>(binCountY, nullptr));
}

// Assign particles to bins
void assign_particles_to_bins(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Clear bins
    for (auto& row : bins) {
        for (ListNode* node : row) {
            delete node; // Delete existing nodes
            // node = nullptr;
        }
        row.clear(); // Clear row
    }

    // double check off by one --> rank = 0 already doing something

    // Assign particles to bins
    int start = rank * num_parts / num_procs;
    int end = (rank + 1) * num_parts / num_procs;
    for (int i = start; i < end; ++i) {
        // int binX = std::min(static_cast<int>(parts[i].x / binSize), binCountX - 1); // Clip binX to valid range
        // int binY = std::min(static_cast<int>(parts[i].y / binSize), binCountY - 1); // Clip binY to valid range
        // binX = std::max(binX, 0); // Ensure binX is not negative
        // binY = std::max(binY, 0); // Ensure binY is not negative
        // bins[binX][binY] = new ListNode(i); // Assign a new node to the bin
        int binX = parts[i].x / binSize;
        int binY = parts[i].y / binSize;
        // if division is incorrect manually add a bin?
        // MPI
        bins[binX][binY] = new ListNode(i); // Assign a new node to the bin
    }

    // Exchange ghost particles with neighboring processes
    int left_proc = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_proc = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // Non-blocking communication for exchanging ghost particles
    MPI_Request reqs[4];
    MPI_Status stats[4];
    MPI_Isend(bins.front().data(), bins.front().size(), MPI_INT, left_proc, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(bins.back().data(), bins.back().size(), MPI_INT, right_proc, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(bins.back().data(), bins.back().size(), MPI_INT, right_proc, 0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(bins.front().data(), bins.front().size(), MPI_INT, left_proc, 0, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, stats);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Re-assign particles to bins to account for movement
    assign_particles_to_bins(parts, num_parts, size, rank, num_procs);
    // check openmp code and add MPI_Barrier

    // Compute forces with binning
    int start = rank * num_parts / num_procs;
    int end = (rank + 1) * num_parts / num_procs;
    for (int i = start; i < end; ++i) {
        parts[i].ax = parts[i].ay = 0;
        int binX = parts[i].x / binSize;
        int binY = parts[i].y / binSize;

        // Iterate through neighboring bins
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int newX = binX + dx, newY = binY + dy;
                if (newX >= 0 && newX < binCountX && newY >= 0 && newY < binCountY) {
                    ListNode* node = bins[newX][newY]; // Get the head of the linked list
                    while (node != nullptr) {
                        // add MPI stuff here
                        apply_force(parts[i], parts[node->index]);
                        node = node->next;
                    }
                }
            }
        }
    }

    // Move particles
    for (int i = start; i < end; ++i) {
        move(parts[i], size);
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Gather all particles to root process (rank 0)
    particle_t* all_particles = nullptr;
    if (rank == 0) {
        all_particles = new particle_t[num_parts * num_procs];
    }

    MPI_Gather(parts, num_parts, PARTICLE, all_particles, num_parts, PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Now all_particles array contains all particles from all processes
        // You can process or save them as needed
        // Remember to free the memory allocated for all_particles
        // save(fsave, all_particles, num_parts * num_procs, size); // Implement this function as needed
        delete[] all_particles;
    }
}