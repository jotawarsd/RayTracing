#include <iostream>
#include <time.h>
using namespace std;

#include "../RTutils/rtweekend.h"
#include "../RTutils/color.h"
#include "../RTutils/hittable_list.h"
#include "../RTutils/sphere.h"
#include "../RTutils/camera.h"
#include "../RTutils/material.h"

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

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));
    world.add(make_shared<sphere>(point3(0, 1, 0), -0.9, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

int main(void)
{
    //image dimensions
    const auto aspectRatio = 3.0 / 2.0;
    const int imageHeight = 1000;
    const int imageWidth = static_cast<int>(imageHeight * aspectRatio);
    const int samplesPerPixel = 500;
    const int maxDepth = 50;

    //world
    auto world = random_scene();

    //camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.15;

    camera cam(lookfrom, lookat, vup, 20, aspectRatio, aperture, dist_to_focus);

    //time
    time_t start, end;

    //render
    cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    time(&start);
    ios_base::sync_with_stdio(false);
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
    time(&end);
    double timeTaken = double(end - start);

    cerr << "\nDone.\n";
    cerr << "Time taken : " << fixed << timeTaken << " seconds" << endl;
}
