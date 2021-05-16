//
//  Ray.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Ray_hpp
#define Ray_hpp

#include "Point.hpp"

/*
 Defines the Ray struct. A ray originates from a point and continues in a single direction.
 The Ray struct contains a Point representing its origin, and a UnitVec describing its
 direction.
*/
struct Ray{
    
    Point origin;
    UnitVec direction;

    Ray(Point origin, Point other) : origin(origin), direction(other-origin) {};
    
    Ray(const Ray &r) : origin(r.origin), direction(r.direction) {};
    
    /*
     Returns the Ray originating from the same point, but traveling in the opposite direction.
     */
    Ray opposite() const{
        return Ray(origin, origin-direction);
    }
};

#endif /* Ray_hpp */
