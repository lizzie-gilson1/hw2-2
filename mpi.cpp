#include "common.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

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

//Integrate the ODE
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
            node = nullptr;
        }
        row.clear(); // Clear row
    }

    // double check off by one --> rank = 0 already doing something

    // Assign particles to bins
    int start = rank * num_parts / num_procs;
    int end = (rank + 1) * num_parts / num_procs;
    for (int i = start; i < end; ++i) {
        int binX = parts[i].x / binSize;
        int binY = parts[i].y / binSize;
        std::cout << "Particle " << i << " assigned to bin (" << binX << ", " << binY << ")" << std::endl;

        // Perform bounds checking
        if (binX >= 0 && binX < binCountX && binY >= 0 && binY < binCountY) {
            // Assign a new node to the bin
            bins[binX][binY] = new ListNode(i);
        } else {
            // Handle cases where binX or binY are out of range
            // For example, you might want to skip assigning to this bin or adjust the indices
            // Here, I'll adjust the indices to ensure they fall within the valid range
            binX = std::max(0, std::min(binX, binCountX - 1));
            binY = std::max(0, std::min(binY, binCountY - 1));
            std::cout << "Adjusted indices: (" << binX << ", " << binY << ")" << std::endl;

            // Assign a new node to the adjusted bin
            bins[binX][binY] = new ListNode(i);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}



void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // MPI_Barrier(MPI_COMM_WORLD);
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
  // Synchronize all processes before moving particles
    MPI_Barrier(MPI_COMM_WORLD);

    // Move particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }

    // Synchronize all processes before re-assigning particles to bins
    MPI_Barrier(MPI_COMM_WORLD);

    // Recalculate bins for particles that moved in the previous time step
    // assign_particles_to_bins(parts, num_parts, size, rank, num_procs);
  
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Allocate memory for recv_parts on all ranks
    particle_t* recv_parts = new particle_t[num_parts * num_procs];

    // Calculate recv_counts and displacements
    int recv_counts[num_procs];
    int displacements[num_procs];
    for (int i = 0; i < num_procs; i++) {
        recv_counts[i] = num_parts;
        displacements[i] = i * num_parts;
    }

    // Gather particles from all processors to the main processor using MPI_Gatherv
    MPI_Gatherv(parts, num_parts, PARTICLE, recv_parts, recv_counts, displacements, PARTICLE, 0, MPI_COMM_WORLD);

    // Sort received particles by particle id (only master processor)
    if (rank == 0) {
        std::sort(recv_parts, recv_parts + num_parts * num_procs,
                  [](const particle_t& a, const particle_t& b) { return a.id < b.id; });


        std::copy(recv_parts, recv_parts + num_parts, parts);
    }

    // Clean up memory on all ranks
    delete[] recv_parts;
}
