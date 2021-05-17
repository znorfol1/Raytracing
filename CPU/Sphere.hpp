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
 Defines the Sphere class. A Sphere is determined by its center Point and a radius value.
 
 See Solid.hpp for virtual function documentation
*/
class Sphere: public Solid{
    Point c; // center of Sphere
    double radius;  // radius
public:
    Sphere(Point c, double r, RGB color=Color::WHITE, double refl=0): Solid(color, refl), c(c), radius(r) {};
    
    virtual Point intersect(const Ray &r) const override{
        Point u = r.direction;
        Point o = r.origin;
        double del = SQR(u.dot(o-c)) + SQR(radius) - SQR((o-c).norm());
        if(del < 0){
            return POINT_NAN;
        }
        double d = -(u.dot(o-c)) - sqrt(del);
        if(d <= 0-SOLID_EPSILON){
            return POINT_NAN;
        }
        return o + d*u;
    }
    
    virtual UnitVec normal(const Point &p) const override{
        return Ray(c, p).direction;
    }
    
    virtual Ray reflect(const Ray &r) const override{
        Point intersection = intersect(r);
        Ray pos(intersection, (2*intersection)-c);
        Plane plane(pos);
        Ray a = plane.reflect(r);
        if(Point::isNan(a.origin)){
            return Ray(r.origin+SOLID_EPSILON*r.direction, r.origin+r.direction);
        }
        return plane.reflect(r);
    }
};

#endif /* Sphere_hpp */
