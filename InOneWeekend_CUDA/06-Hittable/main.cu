//header files
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

//local headers
#include "./RTutils_GPU/color.h"
#include "./RTutils_GPU/vec3.h"
#include "./RTutils_GPU/ray.h"
#include "./RTutils_GPU/hittable_list.h"
#include "./RTutils_GPU/sphere.h"
#include "./RTutils_GPU/rtweekend.h"

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

//kernel functions
__device__ vec3 rayColor(const ray& r, hittable **world)
{
    hit_record rec;
    if ((*world)->hit(r, 0.0f, FLT_MAX, rec))
    {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 2.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin, hittable **world)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int pixelIndex = j * max_x + i;

    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    ray r(origin, lowerLeftCorner + u * horizontal + v * vertical - origin);

    fb[pixelIndex] = rayColor(r, world);
}

__global__ void createWorld(hittable **d_list, hittable **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new sphere(point3(0, 0, -1), 0.5);
        d_list[1] = new sphere(point3(0, -100.5, -1), 100);

        *d_world = new hittable_list(d_list, 2);
    }
}

__global__ void freeWorld(hittable **d_list, hittable **d_world) {
   delete *(d_list);
   delete *(d_list+1);
   delete *d_world;
}

int main(void)
{
    //image dimensions
    const float aspectRatio = 16.0f / 9.0f;
    const int imageWidth = 1280;
    const int imageHeight = int(imageWidth / aspectRatio);

    //world
    hittable **d_list;
    int numSpheres = 2;
    checkCudaErrors(cudaMalloc((void **)&d_list, numSpheres * sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    createWorld <<< 1, 1 >>> (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //camera
    float viewportHeight = 2.0f;
    float viewportWidth = viewportHeight * aspectRatio;
    float focalLength = 1.0f;

    vec3 origin = point3(0.0f, 0.0f, 0.0f);
    vec3 horizontal = vec3(viewportWidth, 0.0f, 0.0f);
    vec3 vertical = vec3(0.0f, viewportHeight, 0.0f);
    vec3 lowerLeftCorner = origin - (horizontal / 2.0f) - (vertical / 2.0f) - vec3(0.0f, 0.0f, focalLength);

    // create frame buffer
    int numPixels = imageWidth * imageHeight;
    size_t fbSize = numPixels * sizeof(vec3);
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, fbSize));

    //kernel
    int tx = 32;
    int ty = 32;

    dim3 blocks(imageWidth / tx + 1, imageHeight / ty + 1);
    dim3 threads(tx, ty);

    render <<< blocks, threads >>> (frameBuffer, imageWidth, imageHeight, lowerLeftCorner, horizontal, vertical, origin, d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //render
    cout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";

    for (int j = imageHeight - 1; j >= 0; --j)
    {
        cerr << "\rScanlines remaining: " << j << ' ' << flush;
        for (int i = 0; i < imageWidth; ++i)
        {
            size_t pixelIndex = (j * imageWidth + i);
            write_color(std::cout, frameBuffer[pixelIndex]);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    freeWorld<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(frameBuffer));

    cerr << "\nDone.\n";
}