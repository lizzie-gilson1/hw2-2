#include "common.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <list>
#include <algorithm>
#include <array>





std::list<particle_t> ghost_parts; // ghost pars
std::list<particle_t> actual_parts; // actual_parts
std::list<particle_t> in_parts; // in_parts
std::list<particle_t> out_parts1; // above outgoing
std::list<particle_t> out_parts2; // below outgoing
std::list<particle_t> ghost_parts1; // above ghost
std::list<particle_t> ghost_parts2; // below ghost



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

void copy_particle(particle_t& part1, particle_t& part2)
{
   
    part2.x = part1.x;
    part2.vx = part1.vx;
    part2.ax = part1.ax;


    part2.y = part1.y;
    part2.vy = part1.vy;
    part2.ay = part1.ay;
   
    part2.id = part1.id;
   
}

void receive(std::list<particle_t>& in_parts, int incoming_rank)
{
    MPI_Status status;
    int num_particles;
    
    MPI_Recv(&num_particles, 1, MPI_INT, incoming_rank, 0, MPI_COMM_WORLD, &status);

    for(int i = 0; i < num_particles; i++)
    {
        particle_t received_particle;
        MPI_Recv(&received_particle, 1, PARTICLE, incoming_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        in_parts.push_back(received_particle);
    }
}


void send_out(std::list<particle_t>* outgoing_parts, int send_rank, int destination_rank)
{
    int num_particles = outgoing_parts->size(); 

    MPI_Send(&num_particles, 1, MPI_INT, destination_rank, 0, MPI_COMM_WORLD);

    for (const auto& particle : *outgoing_parts) // foreach loop
    {
        MPI_Send(&particle, 1, PARTICLE, destination_rank, 0, MPI_COMM_WORLD);
    }
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) { // Giovi does init (switch)
    double row_width = size / num_procs;
    std::vector<std::list<particle_t>> otherghost;
    otherghost.resize(num_procs);
    if(rank == 0)
    {
        std::vector<std::list<particle_t>> outsending; 
        outsending.resize(num_procs);
        for (int i = 0; i < num_parts; ++i)
        {
            parts[i].ax = parts[i].ay = 0;
            int bin = static_cast<int>(parts[i].y / row_width);
            outsending[bin].push_back(parts[i]);
            if (bin - 1 >= 0 && abs(parts[i].y - bin * row_width) <= cutoff)
            {
                otherghost[bin - 1].push_back(parts[i]);
            }

            if (bin + 1 < num_procs && abs(parts[i].y - (bin + 1)*row_width) <= cutoff)
            {
                otherghost[bin + 1].push_back(parts[i]);
            }

            if (bin == rank){
                actual_parts.push_back(parts[i]);
            }

            if (bin == rank - 1 && abs(parts[i].y - rank * row_width) <= cutoff)
            {
                ghost_parts.push_back(parts[i]);
            }

            if (bin == rank + 1 && abs(parts[i].y - (rank + 1)*row_width) <= cutoff)
            {
                ghost_parts.push_back(parts[i]);
            }
        }

        for (int out_proc = 1; out_proc < num_procs; out_proc++)
        {
            std::list<particle_t> parts_out = outsending[out_proc];
            send_out(&parts_out, 0, out_proc);
            parts_out.clear();
        }
    }
    else 
    {
        receive(actual_parts, 0);
    }
    if(rank == 0) {
        for (int out_proc = 1; out_proc < num_procs; out_proc++)
        {
            std::list<particle_t> parts_out = otherghost[out_proc];
            send_out(&parts_out, 0, out_proc);
            parts_out.clear();
        }
    }
    else
    {
        receive(ghost_parts, 0);
    }
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) { // giovino

    double row_width = size / num_procs;
    for( particle_t& particle : actual_parts)   // try diff for loop
    {
        for (particle_t& neighbor : actual_parts)
        {
            apply_force(particle, neighbor);
        }
        for (particle_t& neighbor : ghost_parts)
        {
            apply_force(particle, neighbor);
        }
    }

    ghost_parts.clear();
    for (auto it = actual_parts.begin(); it != actual_parts.end();) //try diff
    {
        particle_t& particle = *it;
        move(particle, size);
        particle.ax= particle.ay = 0;
        int bin = static_cast<int>(particle.y / row_width);
        if(bin == rank - 1) // redo if statements
        {
            out_parts2.push_back(particle);
            it = actual_parts.erase(it);
        }
        else if (bin == rank + 1)
        {
            out_parts1.push_back(particle);
            it = actual_parts.erase(it);
        }
        else
        {
            ++it;
        } 

    }

    if(rank % 2 == 0)
    {
        if(rank + 1 < num_procs) //switchs
        {
            send_out(&out_parts1, rank, rank + 1);
            out_parts1.clear();
        }
        if (rank - 1 >=0)
        {
            send_out(&out_parts2, rank, rank - 1);
            out_parts2.clear();
        }
        if (rank + 1 < num_procs)
        {
            receive(in_parts, rank + 1);
            for (particle_t& particle: in_parts)
            {
                actual_parts.push_back(particle);
            }
            in_parts.clear();
        }

        if (rank - 1 >= 0)
        {
            receive(in_parts, rank - 1);
            for (particle_t& particle: in_parts)
            {
                actual_parts.push_back(particle);
            }
            in_parts.clear();
        }

    } else {
        if (rank + 1 < num_procs)
        {
            receive(in_parts, rank + 1);
            for (particle_t& particle : in_parts)
            {
                actual_parts.push_back(particle);
            }
            in_parts.clear();
        }
        if (rank - 1 >= 0)
        {
            receive(in_parts, rank - 1);
            for (particle_t& particle : in_parts)
            {
                actual_parts.push_back(particle);
            }
            in_parts.clear();
        }
        if (rank + 1 < num_procs)
        {
            send_out(&out_parts1, rank, rank + 1);
            out_parts1.clear();
        }
        if (rank - 1 >= 0)
        {
            send_out(&out_parts2, rank, rank - 1);
            out_parts2.clear();
        }
    }

    for (particle_t& particle : actual_parts)
    {
        if (rank - 1 >= 0 && abs(particle.y - rank * row_width) <= cutoff)
        {
            ghost_parts2.push_back(particle);
        }
        if (rank + 1 < num_procs && abs(particle.y - (rank + 1) * row_width) <= cutoff)
        {
            ghost_parts1.push_back(particle);
        }
    }
    if(rank % 2 == 0)
    {
        if (rank + 1 < num_procs)
        {
            send_out(&ghost_parts1, rank, rank + 1);
            ghost_parts1.clear();
        }
        if (rank - 1 >= 0)
        {
            send_out(&ghost_parts2, rank, rank - 1);
            ghost_parts2.clear();
            // ERRROR!

        }
        if (rank - 1 >= 0)
        {
            in_parts.clear();
            receive(in_parts, rank - 1);
            for (particle_t& particle : in_parts)
            {
                ghost_parts.push_back(particle);
            }
            in_parts.clear();
        }
        if (rank + 1 < num_procs)
        {
            in_parts.clear();
            receive(in_parts, rank + 1);
            for (particle_t& particle : in_parts)
            {
                ghost_parts.push_back(particle);
            }
            in_parts.clear();
        }
    } else {
        if (rank - 1 >= 0)
        {
            receive(in_parts, rank - 1);
            for (particle_t& particle : in_parts)
            {
                ghost_parts.push_back(particle);
            }
            in_parts.clear();
        }

        if (rank + 1 < num_procs)
        {
            receive(in_parts, rank + 1);
            for (particle_t& particle : in_parts)
            {
                ghost_parts.push_back(particle);
            }
            in_parts.clear();
        }
        if (rank + 1 < num_procs)
        {
            // ERROR!
            send_out(&ghost_parts1, rank, rank + 1);
            ghost_parts1.clear();
        }
        if (rank - 1 >= 0)
        {
            send_out(&ghost_parts2, rank, rank - 1);
            ghost_parts2.clear();
        }   
    }
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) { // change loops?

    std::list<particle_t> receive_buffer;

    particle_t* local_parts_array = new particle_t[actual_parts.size()];
    std::copy(actual_parts.begin(), actual_parts.end(), local_parts_array);
    if (rank == 0)
    {
        for (particle_t& particle : actual_parts)
        {
           
            copy_particle(particle, parts[(int)(particle.id - 1)]);
        }
        for (int i = 1; i < num_procs; i++){
            receive(receive_buffer, i);
            for(particle_t& particle : receive_buffer){
              
                copy_particle(particle, parts[(int)(particle.id - 1)]);
            }
            receive_buffer.clear();
        }
    } else {
        send_out(&actual_parts, rank, 0);
    }
}











