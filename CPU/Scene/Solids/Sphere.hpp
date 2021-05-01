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
#include <iostream>

#define SQR(x) ((x)*(x))

/*
 Defines the Sphere class. A Sphere is defined by its center Point and a radius.
*/


class Sphere: public Solid{
    Point c;
    double radius;
    
    constexpr const static double EPSILON = .000001;
public:
    Sphere(Point c, double r, RGB color): Solid(color), c(c), radius(r) {};
    
    virtual Point intersects(Ray r) const override{
        Point u = r.direction;
        Point o = r.origin;
        double del = SQR(u.dot(o-c)) + SQR(radius) - SQR((o-c).norm());
        if(del ==0){
            return POINT_NAN;
        }
        double d = -(u.dot(o-c)) - sqrt(del);
        if(d <=0-EPSILON){
            return POINT_NAN;
        }
        return o + d*u;
    }
    
    virtual Ray reflect(Ray r) const override{
        Point intersection = intersects(r);
        Ray pos(intersection, (2*intersection)-c);
        Plane plane(pos);
        return plane.reflect(r);
    }
};


#endif /* Sphere_hpp */
