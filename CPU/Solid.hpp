//
//  Solid.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Solid_hpp
#define Solid_hpp

#include "Ray.hpp"
#include "Color.hpp"

#define SOLID_EPSILON 0.000001

/*
 Defines the abstract Solid class. Solids are the objects in the scene which interact with
 light. These are the objects that show up in the rendering. Solids have a color and reflectivity value.
 Reflectivity is currently unsupported, so all objects are diffuse.
 */
class Solid{
public:
    RGB color;
    double reflectivity; //Must be in [0,1]
    //  TODO: Add support for reflectivity
    
    Solid(RGB c = Color::WHITE, double r = 0) : color(c), reflectivity(r){};
    
    Solid(const Solid &s):  Solid(s.color, s.reflectivity) {};
    
    /*
     Determimes the intersection Point of a Ray with the Solid. If the Ray does not
     intersect, POINT_NAN is returned.
     */
    virtual Point intersect(const Ray &r) const = 0;
    
    /*
     Returns the vector normal to the surface at Point p on the surface.
     */
    virtual UnitVec normal(const Point &p) const = 0;
    
    /*
     Finds the reflection of the given Ray off of the Solid. The origin of the resulting Ray
     is the Point of intersection.
     */
    virtual Ray reflect(const Ray &r) const = 0;
    
    /*
     Returns the amount of light from an incoming Ray that gets reflected in the outgoing
     direction. Currenty it is constant as only diffuse objects are supported.
     */
    double brdf(const Ray &in, const UnitVec &out) const{
        return 1;
    }
};

#endif /* Solid_hpp */
