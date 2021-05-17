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
 Defines the Plane class. A Plane is determined by a singular Ray who's
 direction is the surface normal and who's origin is some point on the plane.
 
 See Solid.hpp for virtual function documentation
 */
class Plane: public Solid{
protected:
    Ray p; // normal ray
public:
    __device__
    Plane(Ray p, RGB c = WHITE, double r = 0): Solid(c, r), p(p) {};
    
    __device__
    Plane(const Plane &a): Solid(a), p(a.p) {};
    
    __device__
    virtual ~Plane() {};
    
    __device__
    virtual Point intersect(const Ray &r) const override{
        double d = r.direction.dot(p.direction);
        if(d == 0){
            return POINT_NAN;
        }
        d = (p.origin - r.origin).dot(p.direction)/d;
        if(d <= 0+SOLID_EPSILON){
            return POINT_NAN;
        }
        return r.origin + (r.direction*d);
    }
    
    __device__
    virtual UnitVec normal(const Point &p) const override{
        return this->p.direction;
    }
    
    __device__
    virtual Ray reflect(const Ray &r) const override{
        Point a = p.direction.reflect(r.direction);
        Point intersection = intersect(r);
        return Ray(intersection, a+intersection);
    }
};

/*
Device initalizers
*/

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


