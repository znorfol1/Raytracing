//
//  Light.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Light_hpp
#define Light_hpp

#include "Color.hpp"
#include "Ray.hpp"

/*
 A struct representing a light source. It has a location and a color.
 */
struct Light{

    RGB color = Color::WHITE;
    Point source;
    
    Light(Point s = Point(0,0,0)): source(s) {};
    
    Light(const Light &o): Light(o.source) {};
    
    /*
     Determines if a Ray hits the Light.
     */
    bool hits(const Ray &r) const{
        double d =  distanceTo(r);
        return -0.001 < d && d < .35;
    }
    
    /*
     Returns the distance from this Light to the closest point on a given Ray.
     */
    double distanceTo(const Ray &r) const{
        double lambda = (source.dot(r.direction)) - (r.origin.dot(r.direction));
        if(lambda < 0 - 0.0000001){
            return -1;
        }
        return (source - (r.origin + lambda*r.direction)).norm();
    }
};

/*
 Struct representing a Ray that also carries a color.
 */
struct LightRay{
    
    RGB color = Color::WHITE;
    Ray r;
    
    LightRay(Ray r = Ray(Point(0,0,0), Point(0,0,0)), RGB c = Color::WHITE): r(r), color(c) {};
    
    LightRay(Light l, Point to): r(l.source,to), color(l.color){};
    
    LightRay(const LightRay& o): LightRay(o.r, o.color) {};
};

#endif /* Light_hpp */
