#include <iostream>
using namespace std;

#include "RTutils/rtweekend.h"
#include "RTutils/color.h"
#include "RTutils/hittable_list.h"
#include "RTutils/sphere.h"
#include "RTutils/camera.h"
#include "RTutils/material.h"

color rayColor(const ray& r, const hittable& world, int depth)
{
    hit_record rec;

    if (depth <= 0)
        return color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec))
    {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * rayColor(scattered, world, depth-1);
        return color(0, 0, 0);
    } 
    vec3 unitDirection = unit_vector(r.direction());
    auto t = 0.5 * (unitDirection.y() + 2.0);

    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main(void)
{
    //image dimensions
    const auto aspectRatio = 16.0 / 9.0;
    const int imageHeight = 600;
    const int imageWidth = static_cast<int>(imageHeight * aspectRatio);
    const int samplesPerPixel = 10;
    const int maxDepth = 10;

    //world
    hittable_list world;
    
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<dielectric>(1.3);
    auto material_left = make_shared<metal>(color(0.8, 0.8, 0.8), 0.2);
    auto material_right = make_shared<metal>(color(0.8, 0.5, 0.2), 0.0);

    world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), -0.4, material_center));
    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    //camera
    point3 lookfrom(3,3,2);
    point3 lookat(0,0,-1);
    vec3 vup(0,1,0);
    auto dist_to_focus = (lookfrom-lookat).length();
    auto aperture = 2.0;

    camera cam(lookfrom, lookat, vup, 20, aspectRatio, aperture, dist_to_focus);

    //render
    cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for (int j = imageHeight - 1; j >= 0; --j)
    {
        cerr << "\rScanlines remaining: " << j << ' ' << flush;
        for (int i = 0; i < imageWidth; ++i)
        {
            color pixelColor(0, 0, 0);
            for (int s = 0; s < samplesPerPixel; ++s)
            {
                auto u = (double(i) + random_double()) / (imageWidth - 1);
                auto v = (double(j) + random_double()) / (imageHeight - 1);

                ray r = cam.get_ray(u, v);
                pixelColor += rayColor(r, world, maxDepth);
            }
            write_color(std::cout, pixelColor, samplesPerPixel);
        }
    }
    cerr << "\nDone.\n";
}
