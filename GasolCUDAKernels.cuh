#ifndef GASOLCUDAKERNELS
#define GASOLCUDAKERNELS
__constant__ double w[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

__global__ void evaluatePopulationKernel(double *fitness, int max_ind, int current_point, double all_g, float *max_g, int *population, int *forbid_combination) {
    int nind = blockIdx.x * blockDim.x + threadIdx.x;

    if (nind < max_ind) {
        if (fitness[nind] < -1.0) {
            double d1 = 0.0;
            double d2 = 1.0;
            double p2 = 0.0001;
            double current_g = 0.0;
            int bad_w = 0;

            for (int j = 0; j < current_point; j++) {
                if (population[nind*(current_point+1)+j] == 1) {
                    current_g += max_g[j];

                    for (int k = j + 1; k < current_point; k++) {
                        if (population[nind*(current_point+1)+k] == 1 && forbid_combination[j*current_point+k] == 1) {
                            d2 = 0.0;
                            bad_w += 1;
                        }
                    }
                }
            }

            d1 = current_g / all_g;
            double w_sum = 0.0;
            double e = 1.0;
            p2 = (double)bad_w / (double)current_point;
            double penalty = 1.0;

            for (int j = 0; j < 2; j++) {
                w_sum += w[j];
            }

            penalty *= p2;
            e *= pow(d1, w[0]);
            e *= pow(d2, w[1]);
            e = pow(e, 1.0 / w_sum);
            penalty = p2 - 0.0001;
            e -= penalty;
            fitness[nind] = e;
        }
    }
}


#endif