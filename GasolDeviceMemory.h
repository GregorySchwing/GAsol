#ifndef GASOLDEVICEMEMORY
#define GASOLDEVICEMEMORY

#include <cuda_runtime.h>
#include <cassert>
class GasolDeviceMemory {
public:
    GasolDeviceMemory() {
        d_fitness = nullptr;
        d_x_index = nullptr;
        d_y_index = nullptr;
        d_z_index = nullptr;
        d_g = nullptr;
        d_max_g = nullptr;
        d_points_radii = nullptr;
        d_forbid_combination = nullptr;
        d_population = nullptr;
    }

    ~GasolDeviceMemory() {
        freeDeviceMemory();
    }

    void allocateAndCopyToDevice(double *h_fitness, int *h_x_index, int *h_y_index, int *h_z_index, float *h_g, float *h_max_g, float *h_points_radii, int **h_forbid_combination, int** h_population, int n_points, int h_current_point, int h_max_ind) {
        current_point = h_current_point;
        max_ind = h_max_ind;
        freeDeviceMemory();

        cudaMalloc((void**)&d_fitness, max_ind * sizeof(double));
        cudaMalloc((void**)&d_x_index, n_points * sizeof(int));
        cudaMalloc((void**)&d_y_index, n_points * sizeof(int));
        cudaMalloc((void**)&d_z_index, n_points * sizeof(int));
        cudaMalloc((void**)&d_g, n_points * sizeof(float));
        cudaMalloc((void**)&d_max_g, n_points * sizeof(float));
        cudaMalloc((void**)&d_points_radii, current_point * sizeof(float));

        cudaMalloc((void**)&d_forbid_combination, current_point * current_point * sizeof(int));

        for (int i = 0; i < current_point; i++) {
            cudaMemcpy(&d_forbid_combination[i*current_point], h_forbid_combination[i], current_point * sizeof(int), cudaMemcpyHostToDevice);
        }

        cudaMalloc((void**)&d_population, max_ind * (current_point+1) * sizeof(int));

        for (int i = 0; i < max_ind; i++) {
            cudaMemcpy(&d_population[i*(current_point+1)], h_population[i], current_point+1 * sizeof(int), cudaMemcpyHostToDevice);
        }

        

        int * forbid_combination = (int *) calloc(sizeof(int), current_point*current_point);
        int * population = (int *) calloc(sizeof(int), max_ind * (current_point+1));

        for (int i = 0; i < current_point; i++) {
            memcpy(&forbid_combination[i*current_point], h_forbid_combination[i], current_point * sizeof(int));
        }

        for (int i = 0; i < max_ind; i++) {
            memcpy(&population[i*(current_point+1)], h_population[i], current_point+1 * sizeof(int));
        }

        for (int i = 0; i < current_point; i++) 
            for (int j = 0; j < current_point; ++j)
                assert(forbid_combination[i*(current_point) + j]==h_forbid_combination[i][j]);
        
        for (int i = 0; i < max_ind; i++) 
            for (int j = 0; j < (current_point+1); ++j)
                assert(population[i*(current_point+1) + j]==h_forbid_combination[i][j]);
        

        cudaMemcpy(d_fitness, h_fitness, max_ind * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_index, h_x_index, n_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_index, h_y_index, n_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z_index, h_z_index, n_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g, n_points * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_g, h_max_g, n_points * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points_radii, h_points_radii, current_point * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Add any other member functions and data members as needed
    int current_point;
    int max_ind;
    double* d_fitness;
    int* d_x_index;
    int* d_y_index;
    int* d_z_index;
    float* d_g;
    float* d_max_g;
    float* d_points_radii;
    int* d_forbid_combination;
    int* d_population;

private:


    void freeDeviceMemory() {
        if (d_fitness) cudaFree(d_fitness);
        if (d_x_index) cudaFree(d_x_index);
        if (d_y_index) cudaFree(d_y_index);
        if (d_z_index) cudaFree(d_z_index);
        if (d_g) cudaFree(d_g);
        if (d_max_g) cudaFree(d_max_g);
        if (d_points_radii) cudaFree(d_points_radii);
        if (d_forbid_combination) cudaFree(d_forbid_combination);
        if (d_population) cudaFree(d_population);
    }
};

#endif
