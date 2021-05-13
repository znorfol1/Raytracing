//
//  Plane.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 4/5/21.
//

#ifndef Plane_hpp
#define Plane_hpp

#include "Solid.hpp"
#include "Scene.hpp"
#include <iostream>

/*
 Defines the Plane class and its subclass Disk. A Plane is defined by a singular Ray who's
 direction is the normal and origin is some point on the plane. A Disk is a flat circle.
 */

class Plane: public Solid{
protected:
    Ray p;
public:


        
    __device__
    Plane(Ray p, RGB c = BLACK, double r = .999): Solid(c, r), p(p) {};
    
    __device__
    Plane(const Plane &a): Solid(a), p(a.p) {};
    
    __device__
    virtual ~Plane() {};
    
    __device__
    virtual Point intersects(Ray r) const override{
        //printf("In Plane Intersects\n");
        //printf("    %lf , %lf, %lf\n", r.direction.x,  r.direction.y, r.direction.z);
        double d = r.direction.dot(p.direction);
        
        if(d == 0){
            return POINT_NAN;
        }
        d = (p.origin - r.origin).dot(p.direction)/d;
        if(d <= 0+SOLID_EPSILON){
            return POINT_NAN;
        }
        return r.origin + (d*r.direction);
    }
    
    __device__
    virtual UnitVec normal(Point p) const override{
        return this->p.direction;
    }
    
    __device__
    virtual Ray reflect(Ray r) const override{
        Point a = p.direction.reflect(r.direction);
        Point intersection = intersects(r);
        return Ray(intersection, a+intersection);
    }
};


    __global__
    void initPlane(Ray p, RGB c, Solid** solids, unsigned n){
        Plane* plane = new Plane(p, c);
        solids[n] = plane;
    }


    __host__
    void createPlaneGPU(Ray p, RGB c){
        initPlane<<<1,1>>>(p,c,GPU_SOLIDS, GPU_SOLIDS_SIZE);
        GPU_SOLIDS_SIZE++;
    }
    




#endif /* Plane_hpp */

