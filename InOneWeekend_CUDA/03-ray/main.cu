//header files
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

//local headers
#include "color.h"
#include "vec3.h"
#include "ray.h"

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
__device__ vec3 rayColor(const ray& r)
{
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int pixelIndex = (j * max_x + i) * 3;

    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    ray r(origin, lowerLeftCorner + u * horizontal + v * vertical - origin);

    fb[pixelIndex] = rayColor(r);
}

int main(void)
{
    //allocate FB
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, fbSize));

    //image dimensions
    const float aspectRatio = 16.0f / 9.0f;
    const int imageWidth = 1280;
    const int imageHeight = int(imageWidth / aspectRatio);

    //camera
    float viewportHeight = 2.0f;
    float viewportWidth = viewportHeight * aspectRatio;
    float focalLength = 1.0f;

    vec3 origin = point3(0.0f, 0.0f, 0.0f);
    vec3 horizontal = vec3(viewportWidth, 0.0f, 0.0f);
    vec3 vertical = vec3(0.0f, viewportHeight, 0.0f);
    vec3 lowerLeftCorner = origin - (horizontal / 2.0f) - (vertical / 2.0f) - vec3(0.0f, 0.0f, focalLength);

    //kernel
    int tx = 8;
    int ty = 8;

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    render <<< blocks, threads >>> (frameBuffer, width, height, lowerLeftCorner, horizontal, vertical, origin);

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