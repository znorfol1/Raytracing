//
//  Scene.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Scene_hpp
#define Scene_hpp

#include <vector>
#include "Solid.hpp"
#include "Light.hpp"
#include <cstddef>
#include <stdlib.h>

/*
 Defines the Scene class. A Scene is the container that holds Solids and Lights which will be
 used to render the image. Solids and Lights can be created independently and registered to the
 Scene using add(). To be rendered, the scene must be passed to a Camera object.
*/
class Scene{

public:
    std::vector<Solid*> solids; // Solids in the scene
    std::vector<Light*> lights; // Lights in the scene (only 1 light is currently supported)

    Scene(std::vector<Solid*> solids, std::vector<Light*> lights): solids(solids), lights(lights){};
    
    Scene(): solids(), lights() {};
    
    /*
     Adds the Solid to the Scene
     */
    void add(Solid &s){
        solids.push_back(&s);
    }
    
    /*
     Adds the Light to the Scene
     */
    void add(Light &b){
        lights.push_back(&b);
    }

    /*
     Traces the light backwards through the scene starting with the given Ray.
     
     Returns: the incoming color opposite to the given Ray
     */
    RGB trace(Ray r) const{
        return traceLight(r, 3, 5);
    }

private:
    
    /*
     Finds the first object the given ray intersects.
     If none are hit, NULL is returned.
     */
    Solid* closestIntersection(Ray r) const{
        Solid* closest = NULL;
        double dist = INFINITY;
        for(Solid* a : solids){
            Point p = a->intersect(r);
            if(Point::isNan(p)){
                continue;
            }
            double d = p.distanceTo(r.origin);
            if(d < dist){
                closest = a;
                dist = d;
            }
        }
        return closest;
    }
    
    /*
     Generates a random double between -d and d
     */
    static double ran(double d = 1){
        double ran = (double)rand() / (double)RAND_MAX;
        return 2*d*ran - d;
    }
    
    /*
     Generates a random ray off a surface given an intersection point.
     
     i: The Point on the surface
     n: the normal to the surface
     b1: the first local basis element to the surface
     b2: the second local basis element to the  surface
     */
    static Ray randomRay(Point i, UnitVec n, UnitVec b1, UnitVec b2, double d = 1){
        return Ray(i, i+n + ran(d)*b1 + ran(d)*b2);
    }
    
    /*
     Gives a UnitVec living on the plane normal to the input vector
     */
    static UnitVec getBasis1(UnitVec n){
        if(n.x != 0){
            double a = -n.dot(Point(0,1,1))/n.x;
            return UnitVec(a, 1, 1);
        }
        return UnitVec(1, 0, 0);
    }

    /*
     Traces the ray throughout the scene with the specified number of bounces and number of random samples
     */
    RGB traceLight(Ray r, int numBounces, int numSamples) const{
        if(numSamples < 2){
            numSamples = 2;
        }
        LightRay incoming[numSamples];
        Solid* s = closestIntersection(r);
    
        if(s == NULL){
            if(lights[0]->hits(r)){
                return lights[0]->color;
            }
            return Color::ATMOSPHERE;
        }
        if(numBounces == 0){
            return Color::SHADOW;
        }

        Ray refl = s->reflect(r);
        Ray toLight = Ray(refl.origin,  lights[0]->source + Point(ran(.2),0, ran(.2)));
        UnitVec n = s->normal(refl.origin);
        UnitVec b1 = getBasis1(n);
        UnitVec b2 = n.cross(b1);

        int count =  0;
        if(toLight.direction.dot(s->normal(toLight.origin)) >= 0 - .0001 && closestIntersection(toLight)==NULL){
            incoming[0] = LightRay(toLight, traceLight(toLight, numBounces-1,  numSamples));
            count++;
        }
        for(int i = count; i < numSamples; i++){
            Ray r1 = randomRay(refl.origin, n, b1, b2);
            incoming[i] = LightRay(r1,traceLight(r1, numBounces - 1, numSamples));
        }
        return calcColor(incoming, numSamples, s, r.opposite().direction);
    }
    
    /*
     Given an array of incoming LightRays, and an outwards direction, returns the outgoing color.
     */
    static RGB calcColor(LightRay* in, unsigned n, Solid* s, UnitVec out){
        double prob[n];
        double sum = 0;
        for(int i = 0; i < n; i++){
            Ray incoming(in[i].r.origin + in[i].r.direction, in[i].r.origin);
            prob[i] = s->brdf(incoming, out);
            prob[i] *= in[i].r.direction.dot(s->normal(in[0].r.origin));
            sum+=prob[i];
        }
        for(int i = 0; i < n; i++){
            prob[i] /= sum;
        }
        double r = 0;
        double g = 0;
        double b = 0;
        for(int i = 0; i < n; i++){
            double c = prob[i];
            r += c * in[i].color.r;
            g += c * in[i].color.g;
            b += c * in[i].color.b;
        }
        return RGB(r*(s->color.r/255.0), g*(s->color.g/255.0), b*(s->color.b/255.0));
    }
    
};

#endif /* Scene_hpp */
