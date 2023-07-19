#include <iostream>
using namespace std;

#include "color.h"
#include "ray.h"
#include "vec3.h"

color rayColor(const ray& r)
{
    vec3 unitDirection = unit_vector(r.direction());
    auto t = 0.5 * (unitDirection.y() * unitDirection.x() + 2.0);

    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main(void)
{
    //image dimensions
    const auto aspectRatio = 16.0 / 9.0;
    const int imageWidth = 1280;
    const int imageHeight = static_cast<int>(imageWidth / aspectRatio);

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
            color pixelColor = rayColor(r);
            write_color(std::cout, pixelColor);
        }
    }
    cerr << "\nDone.\n";
}
