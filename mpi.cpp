#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>

struct ListNode {
    int index;
    ListNode* next;
    ListNode(int idx) : index(idx), next(nullptr) {}
};

std::vector<std::vector<ListNode*>> bins;
int binCountX, binCountY; // Number of bins in each dimension
double binSize = cutoff; 
int totalSubRegions;
int numRows, numCols;

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
    // divide the space into bins of size cutoff-by-cutoff 
    binCountX = size/cutoff;
    binCountY = size/cutoff;
    bins.resize(binCountX*binCountY); 

    int totalSubRegions = size/num_procs; 
    numRows = totalSubRegions/binSize;
    numCols = size/binSize; 
}

void assign_particles_to_bins(particle_t* parts, int num_parts, double size, int rank, int num_procs){
    // Clear bins 
    for (auto& row : bins) {
        for (ListNode* node : row) {
            delete node; // Delete existing nodes
        }
        row.clear(); // Clear row
    }

    // Determine the range of rows assigned to this MPI process 
    int rows_per_proc = numRows / num_procs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == num_procs - 1) ? numRows : ((rank + 1) * rows_per_proc);

    // Assign particles to bins locally
    for (int i = 0; i < num_parts; i++) {
        // Check if particle is within simulation domain and assigned rows
        if (parts[i].x >= 0 && parts[i].x <= size && parts[i].y >= 0 && parts[i].y <= size &&
            parts[i].y / binSize >= start_row && parts[i].y / binSize < end_row) {
            int x = parts[i].x / binSize;
            int y = parts[i].y / binSize;
            // Check if particle is within the valid bin range
            if (x >= 0 && x < binCountX && y >= 0 && y < binCountY) {
                int binIndex = x * binCountY + y;
                ListNode* node = new ListNode(i);
                // Create a vector if the bin is empty
                if (bins[binIndex].empty()) {
                    bins[binIndex].emplace_back(node);
                } else {
                    // Append to the existing vector
                    bins[binIndex].back()->next = node;
                }
            }
        }
    }

    // // Exchange ghost particles with neighboring processes
    // int left_proc = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    // int right_proc = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // // Non-blocking communication for exchanging ghost particles
    // MPI_Request reqs[4];
    // MPI_Status stats[4];
    // MPI_Isend(bins.front().data(), bins.front().size(), MPI_INT, left_proc, 0, MPI_COMM_WORLD, &reqs[0]);
    // MPI_Isend(bins.back().data(), bins.back().size(), MPI_INT, right_proc, 0, MPI_COMM_WORLD, &reqs[1]);
    // MPI_Irecv(bins.back().data(), bins.back().size(), MPI_INT, right_proc, 0, MPI_COMM_WORLD, &reqs[2]);
    // MPI_Irecv(bins.front().data(), bins.front().size(), MPI_INT, left_proc, 0, MPI_COMM_WORLD, &reqs[3]);
    // MPI_Waitall(4, reqs, stats);
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Re-assign particles to bins to account for movement
    assign_particles_to_bins(parts, num_parts, size, rank, num_procs);
    // check openmp code and add MPI_Barrier

    // Compute forces with binning
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