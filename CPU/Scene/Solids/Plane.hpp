//
//  Plane.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 4/5/21.
//

#ifndef Plane_hpp
#define Plane_hpp

#include "Solid.hpp"

/*
 Defines the Plane class and its subclass Disk. A Plane is defined by a singular Ray who's
 direction is the normal and origin is some point on the plane. A Disk is a flat circle.
 */

class Plane: public Solid{
protected:
    Ray p;
    constexpr const static double EPSILON = .000001;
public:
    Plane(Ray p, RGB c = Color::VOID): Solid(c), p(p) {};
    Plane(const Plane &a): p(a.p) {};
    
    virtual Point intersects(Ray r) const override{
        double d = r.direction.dot(p.direction);
        if(d == 0){
            return POINT_NAN;
        }
        d = (p.origin - r.origin).dot(p.direction)/d;
        if(d <= 0+EPSILON){
            return POINT_NAN;
        }
        return r.origin + (r.direction*d);
    }
    
    virtual Ray reflect(Ray r) const override{
        Point a = p.direction.reflect(r.direction);
        Point intersection = intersects(r);
        return Ray(intersection, a+intersection);
        
        double d = - p.direction.dot(p.origin);
        a = a - (2*d)*p.direction;
        return Ray(intersection, a);
    }
};


class Disk: public Plane{
    double radius;
public:
    Disk(Ray p, double r, RGB c): Plane(p, c), radius(r) {};
    Disk(const Disk &d): Plane(d.p, d.color), radius(d.radius) {};
    
    virtual Point intersects(Ray r) const override{
        Point x = Plane::intersects(r);
        return (x.distance(p.origin) < radius) ? x : POINT_NAN;
    }
};

#endif /* Plane_hpp */
