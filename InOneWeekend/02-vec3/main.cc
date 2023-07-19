//header files
#include <iostream>
using namespace std;

//local headers
#include "color.h"
#include "vec3.h"

int main(void)
{
    //dimensions
    int width = 256;
    int height = 256;

    //render
    cout << "P3\n" << width << ' ' << height << "\n255\n";

    for (int j = height - 1; j >= 0; --j)
    {
        cerr << "\rScanlines remaining: " << j << ' ' << flush;
        for (int i = 0; i < width; ++i)
        {
            color pixel_color(double(i) / (width - 1), double(j) / (height - 1), 0.25);
            write_color(std::cout, pixel_color);
        }
    }

    cerr << "\nDone.\n";
}