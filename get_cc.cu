#include <stdio.h>
int main() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("%d.%d\n", prop.major, prop.minor);
    return 0;
}
