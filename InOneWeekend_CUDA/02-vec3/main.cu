//header files
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

//local headers
#include "color.h"
#include "vec3.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

//global variables
int width = 256;
int height = 256;
int numPixels = width * height;
size_t fbSize = 3 * numPixels * sizeof(vec3);

//kernel function
__global__ void render(vec3 *fb, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int pixelIndex = (j * max_x + i) * 3;
    fb[pixelIndex] = vec3(float(i) / max_x, float(j) / max_y, 0.2);
}

int main(void)
{
    //allocate FB
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, fbSize));

    //kernel
    int tx = 8;
    int ty = 8;

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    render <<< blocks, threads >>> (frameBuffer, width, height);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //render
    cout << "P3\n" << width << ' ' << height << "\n255\n";

    for (int j = height - 1; j >= 0; --j)
    {
        cerr << "\rScanlines remaining: " << j << ' ' << flush;
        for (int i = 0; i < width; ++i)
        {
            size_t pixelIndex = (j * width + i) * 3;
            // float r = frameBuffer[pixelIndex];
            // float g = frameBuffer[pixelIndex + 1];
            // float b = frameBuffer[pixelIndex + 2];

            // int ir = static_cast<int>(255.999 * r);
            // int ig = static_cast<int>(255.999 * g);
            // int ib = static_cast<int>(255.999 * b);

            // cout << ir << ' ' << ig << ' ' << ib << "\n";
            write_color(std::cout, frameBuffer[pixelIndex]);
        }
    }
    checkCudaErrors(cudaFree(frameBuffer));

    cerr << "\nDone.\n";
}