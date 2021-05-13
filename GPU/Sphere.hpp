//
//  Sphere.hpp
//  Simple Rendering
//
//  Created by Zachary Norfolk on 4/5/21.
//

#ifndef Sphere_hpp
#define Sphere_hpp

#include "Solid.hpp"
#include "Plane.hpp"
#include <cmath>

#define SQR(x) ((x)*(x))

/*
 Defines the Sphere class. A Sphere is defined by its center Point and a radius.
*/
class Sphere: public Solid{
    Point c;
    double radius;
    
public:

    __host__ __device__
    Sphere(Point c, double r, RGB color): Solid(color), c(c), radius(r) {};
    
    __device__
    virtual Point intersects(Ray r) const override{
        Point u = r.direction;
        Point o = r.origin;
        double del = SQR(u.dot(o-c)) + SQR(radius) - SQR((o-c).norm());
        if(del < 0){
            return POINT_NAN;
        }
        double d = -(u.dot(o-c)) - sqrt(del);
        if(d <=0-EPSILON){
            return POINT_NAN;
        }
        return o + d*u;
    }
    
    __device__
    virtual UnitVec normal(Point p) const override{
        return Ray(c, p).direction;
    }
    
    __device__
    virtual Ray reflect(Ray r) const override{
        Point intersection = intersects(r);
        Ray pos(intersection, (2*intersection)-c);
        Plane plane(pos);
        Ray a = plane.reflect(r);
        if(Point::isNan(a.origin)){
            return Ray(r.origin+EPSILON*r.direction, r.origin+r.direction);
        }
        return plane.reflect(r);
    }
};


    __global__
    void initSphere(Point p, double r, RGB c, Solid** solids, unsigned n){
        Sphere* sphere = new Sphere(p, r, c);
        solids[n] = sphere;
    }


    __host__
    void createSphereGPU(Point p, double r, RGB c){
        initSphere<<<1,1>>>(p,r,c,GPU_SOLIDS, GPU_SOLIDS_SIZE);
        GPU_SOLIDS_SIZE++;
    }
    

#endif /* Sphere_hpp */

