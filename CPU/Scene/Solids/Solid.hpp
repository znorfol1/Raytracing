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
#include "Image.hpp"

/*
 Defines the abstract Solid class. Solids are the objects in the scene which interact with
 light. These are the material objects to be rendered.
 */

class Solid{
public:
    RGB color;
    Solid(RGB c = Color::VOID) : color(c) {};
    
    /*
     Determimes the intersection Point of a Ray with the Solid. If the Ray does not
     intersect, POINT_NAN is returned.
     */
    virtual Point intersects(Ray r) const = 0;
    
    /*
     Finds the reflection of the given Ray off of the Solid. The origin of the resulting Ray
     is the Point of intersection.
     */
    virtual Ray reflect(Ray r) const = 0;
};


#endif /* Solid_hpp */
