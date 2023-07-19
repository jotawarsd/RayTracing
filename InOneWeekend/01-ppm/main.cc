#include <iostream>
using namespace std;

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
            auto r = double(i) / (width - 1);
            auto g = double(j) / (height - 1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            cout << ir << ' ' << ig << ' ' << ib << "\n";
        }
    }

    cerr << "\nDone.\n";
}