#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"

struct hit_record;

class material
{
public:
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, 
        color& attenuation, ray& scattered, curandState *randState
    ) const = 0;
};

class lambertian : public material
{
public:
    __device__ lambertian(const color& a) : albedo(a) {};
    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *randState) const override
    {
        vec3 scatter_direction = rec.normal + random_unit_vector(randState);

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private: 
    color albedo;
};

class metal : public material
{
public: 
    __device__ metal(const color& a) : albedo(a) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *randState) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;

        return true;
    }

private:
    color albedo;
};

#endif
