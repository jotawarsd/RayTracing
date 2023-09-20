//header files
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

//local headers
#include "./RTutils_GPU/color.h"
#include "./RTutils_GPU/vec3.h"
#include "./RTutils_GPU/ray.h"

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
__device__ float hitSphere(const point3& center, float radius, const ray& r)
{
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(r.direction(), oc);
    float c = oc.length_squared() - (radius * radius);

    float discriminant = (half_b * half_b) - (a * c);

    if (discriminant < 0)
        return -1.0;
    else
        return (-half_b - sqrt(discriminant)) / a;
}

__device__ vec3 rayColor(const ray& r)
{
    float t = hitSphere(point3(0.0f, 0.0f, -1.0f), 0.5f, r);
    if (t > 0.0)
    {
        vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
        return 0.5f * color(N.x() + 1.0f, N.y() + 1.0f, N.z() + 1.0f);
    }

    vec3 unit_direction = unit_vector(r.direction());
    t = 0.5f * (unit_direction.y() + 2.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int pixelIndex = j * max_x + i;

    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    ray r(origin, lowerLeftCorner + u * horizontal + v * vertical - origin);

    fb[pixelIndex] = rayColor(r);
}

int main(void)
{
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

    render <<< blocks, threads >>> (frameBuffer, imageWidth, imageHeight, lowerLeftCorner, horizontal, vertical, origin);

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
    checkCudaErrors(cudaFree(frameBuffer));

    cerr << "\nDone.\n";
}