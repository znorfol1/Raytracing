//
//  Solid.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Solid_hpp
#define Solid_hpp

#include "Ray.hpp"
#include "Point.hpp"
#include "Color.hpp"
#include <cmath>

#define SOLID_EPSILON 0.00001

/*
 Defines the abstract Solid class. Solids are the objects in the scene which interact with
 light. These are the material objects to be rendered.
 */

class Solid{
protected:
    constexpr const static double EPSILON = 0.000001;
public:
    RGB color;
    double reflectivity; //Must be in [0,1]
    
    __device__
    Solid(RGB c , double r = .999) : color(c), reflectivity(r){};
    
    __device__
    Solid(const Solid &s):  Solid(s.color, s.reflectivity) {};
    
    __device__
    virtual ~Solid() {};
    
    /*
     Determimes the intersection Point of a Ray with the Solid. If the Ray does not
     intersect, POINT_NAN is returned.
     */
    __device__
    virtual Point intersects(Ray r) const = 0;
    
    /*
     Returns the vector normal to the surface at Point p.
     */
    __device__
    virtual UnitVec normal(Point p) const = 0;
    
    /*
     Finds the reflection of the given Ray off of the Solid. The origin of the resulting Ray
     is the Point of intersection.
     */
    __device__
    virtual Ray reflect(Ray r) const = 0;
    
    /*
     Returns the amount of light from an incoming Ray that gets reflected in the outgoing
     direction.
     */
    __device__
    double brdf(Ray in, UnitVec out){
        Ray refl = reflect(in);
        double x = out.dot(refl.direction);
        if(x>1){
            x = 1;
        }
        if(x < 0){
            x = 0;
        }
        x = (x+1)/2;

        return -5*log(1-reflectivity) * (1-x)+.0001;
    }
};


#endif /* Solid_hpp */

