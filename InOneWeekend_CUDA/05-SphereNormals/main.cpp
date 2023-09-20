#include <iostream>
using namespace std;

#include "RTutils_CPU/color.h"
#include "RTutils_CPU/ray.h"
#include "RTutils_CPU/vec3.h"

double hitSphere(const point3& center, double radius, const ray& r)
{
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(r.direction(), oc);
    auto c = oc.length_squared() - (radius * radius);

    auto discriminant = (half_b * half_b) - (a * c);
    if (discriminant < 0)
        return -1.0;
    else
        return (-half_b - sqrt(discriminant)) / a;
}

color rayColor(const ray& r)
{
    auto t = hitSphere(point3(0, 0, -1), 0.5, r);
    if (t > 0.0)
    {
        vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }   
    vec3 unitDirection = unit_vector(r.direction());
    t = 0.5 * (unitDirection.y() + 2.0);

    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main(void)
{
    //image dimensions
    const auto aspectRatio = 16.0 / 9.0;
    const int imageHeight = 1080;
    const int imageWidth = static_cast<int>(imageHeight * aspectRatio);

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
