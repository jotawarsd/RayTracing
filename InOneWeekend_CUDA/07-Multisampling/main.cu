//header files
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>
using namespace std;

//local headers
#include "./RTutils_GPU/color.h"
#include "./RTutils_GPU/vec3.h"
#include "./RTutils_GPU/ray.h"
#include "./RTutils_GPU/hittable_list.h"
#include "./RTutils_GPU/sphere.h"
#include "./RTutils_GPU/rtweekend.h"
#include "./RTutils_GPU/camera.h"

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

__global__ void renderInit(int max_x, int max_y, curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_x))
        return;

    int pixelIndex = j * max_x + i;

    curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int numSamples, camera **cam, hittable **world, curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int pixelIndex = j * max_x + i;
    curandState localRandState = randState[pixelIndex];
    vec3 color(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < numSamples; s++)
    {
        float u = float(i + curand_uniform(&localRandState)) / float(max_x);
        float v = float(j + curand_uniform(&localRandState)) / float(max_y);

        ray r = (*cam)->get_ray(u, v);
        color += rayColor(r, world);
    }
    fb[pixelIndex] = color / float(numSamples);
}

__global__ void createWorld(hittable **d_list, hittable **d_world, camera **d_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new sphere(point3(0, 0, -1), 0.5);
        d_list[1] = new sphere(point3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
        *d_camera = new camera();
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
    const int numSamples = 100;

    // create frame buffer
    int numPixels = imageWidth * imageHeight;
    size_t fbSize = numPixels * sizeof(vec3);
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, fbSize));

    //allocate random state
    curandState *d_randState;
    checkCudaErrors(cudaMalloc((void**)&d_randState, numPixels * sizeof(curandState)));

    //world
    hittable **d_list;
    int numSpheres = 2;
    checkCudaErrors(cudaMalloc((void **)&d_list, numSpheres * sizeof(hittable *)));

    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    createWorld <<< 1, 1 >>> (d_list, d_world, d_camera);
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

    //time
    clock_t start, end;
    start = clock();

    //kernel
    int tx = 32;
    int ty = 32;

    dim3 blocks(imageWidth / tx + 1, imageHeight / ty + 1);
    dim3 threads(tx, ty);

    renderInit <<<blocks, threads>>> (imageWidth, imageHeight, d_randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render <<< blocks, threads >>> (frameBuffer, imageWidth, imageHeight, numSamples, d_camera, d_world, d_randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    end = clock();
    double timeTaken = double(end - start) / CLOCKS_PER_SEC;
    cerr << "\nKernel Execution Complete\n";
    cerr << "Time taken : " << fixed << timeTaken << " seconds" << endl;

    //render image
    cout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";
    for (int j = imageHeight - 1; j >= 0; --j)
    {
        for (int i = 0; i < imageWidth; ++i)
        {
            size_t pixelIndex = (j * imageWidth + i);
            write_color(std::cout, frameBuffer[pixelIndex]);
        }
    }
    cerr << "\nDone.\n";

    checkCudaErrors(cudaDeviceSynchronize());
    freeWorld<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(frameBuffer));
}