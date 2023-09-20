#include <iostream>
using namespace std;

#include "RTutils/rtweekend.h"
#include "RTutils/color.h"
#include "RTutils/hittable_list.h"
#include "RTutils/sphere.h"

color rayColor(const ray& r, const hittable& world)
{
    hit_record rec;
    if (world.hit(r, 0, infinity, rec))
        return 0.5 * (rec.normal + color(1, 1, 1));
      
    vec3 unitDirection = unit_vector(r.direction());
    auto t = 0.5 * (unitDirection.y() + 2.0);

    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main(void)
{
    //image dimensions
    const auto aspectRatio = 16.0 / 9.0;
    const int imageHeight = 480;
    const int imageWidth = static_cast<int>(imageHeight * aspectRatio);

    //world
    hittable_list world;
    world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));

    //camera
    auto viewportHeight = 2.0;
    auto viewportWidth = viewportHeight * aspectRatio;
    auto focalLength = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewportWidth, 0, 0);
    auto vertical = vec3(0, viewportHeight, 0);
    auto lowerLeftCorner = origin - (horizontal / 2) - (vertical / 2) - vec3(0, 0, focalLength);

    //render
    cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for (int j = imageHeight - 1; j >= 0; --j)
    {
        cerr << "\rScanlines remaining: " << j << ' ' << flush;
        for (int i = 0; i < imageWidth; ++i)
        {
            auto u = double(i) / (imageWidth - 1);
            auto v = double(j) / (imageHeight - 1);

            ray r(origin, lowerLeftCorner + (u * horizontal) + (v * vertical) - origin);
            color pixelColor = rayColor(r, world);
            write_color(std::cout, pixelColor);
        }
    }
    cerr << "\nDone.\n";
}
