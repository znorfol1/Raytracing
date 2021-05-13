//
//  Scene.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Scene_hpp
#define Scene_hpp


#include "Solid.hpp"
#include "Light.hpp"
#include <assert.h>
#include <iostream>

/*
 Defines the Scene class. A Scene is the container that holds Solids and Lights which will be
 used to render the image. Solids and Lights can be created independently and registered to the
 Scene using add(). To be rendered, the scene must be passed to a Camera object.
*/


static Solid** GPU_SOLIDS = NULL;
static unsigned GPU_SOLIDS_SIZE = 0;
//static Light* GPU_LIGHT = NULL;
static Light GPU_LIGHT;



    __host__
    void createSceneGPU(unsigned n){
        assert(GPU_SOLIDS == NULL);
        
        cudaMalloc(&GPU_SOLIDS, sizeof(Solid*) * n);
        //cudaMalloc(&GPU_LIGHT, sizeof(Light));
    }
    
    __host__
    void createLightGPU(Point s, RGB c){
        GPU_LIGHT = Light(s);
        //initLight<<<1,1>>>(s,c,GPU_LIGHT);
    }



    __device__
    Solid* closestIntersection(Ray r, Solid** solids, unsigned n) {
        Solid* closest = NULL;
        double dist = INFINITY;
        for(int i = 0; i < n; i++){
            Solid* a = solids[i];
            Point p = a->intersects(r);
            if(Point::isNan(p)){
                continue;
            }
            double d = p.distance(r.origin);
            if(d < dist){
                closest = a;
                dist = d;
            }
        }
        return closest;
    }
    

    
    __device__
    RGB calcColor(LightRay* in, unsigned n, Solid* s, UnitVec out){
        double prob[2]; // MAKE SURE n <= 2
        double sum = 0;
        for(int i = 0; i < n; i++){
            Ray incoming(in[i].r.origin + in[i].r.direction, in[i].r.origin);
            prob[i] = s->brdf(incoming, out);
            sum+=prob[i];
        }
        for(int i = 0; i < n; i++){
            prob[i] /= sum;
        }
        double r = 0;
        double g = 0;
        double b = 0;
        for(int i = 0; i < n; i++){
            double c = prob[i] * in[i].r.direction.dot(s->normal(in[0].r.origin));
            r += c * in[i].color.r;
            g += c * in[i].color.g;
            b += c * in[i].color.b;
        }
        return RGB(r*(s->color.r/255.0), g*(s->color.g/255.0), b*(s->color.b/255.0));
    }
    

    __device__
        RGB traceLightNoRecursion(Ray r, int numBounces, Solid** solids, Light light, unsigned n){
        LightRay incoming[2];
        Solid* s = closestIntersection(r, solids, n);
        if(s == NULL){          
            return BLACK;
        }

        Ray refl = s->reflect(r);
        Ray toLight = Ray(refl.origin,  light.source);
        if(closestIntersection(toLight, solids, n) !=  NULL){
            return BLACK;
        }
        incoming[0] = LightRay(toLight, WHITE);
        //return  s->color;
        return calcColor(incoming,1, s,  r.opposite().direction);
    }

/*
        __device__
    RGB traceLight(Ray r, int numBounces, Solid** solids, Light light, unsigned n){
        printf("IN TRACELIGHT BOUNCE %d\n", numBounces);
        LightRay incoming[2];
        Solid* s = closestIntersection(r, solids, n);
        printf("    R direction: %lf , %lf, %lf\n", r.direction.x,  r.direction.y, r.direction.z);

        if(s == NULL){
        printf("Did't hit solid, bounce %d\n", numBounces);
            if(light.hits(r)){
            
                return light.color;
            }
            
            return BLACK;
        }
        printf("Hit solid, bounce %d\n", n);
        if(numBounces == 0){
            return BLACK;
        }
        
        Ray refl = s->reflect(r);
        printf("Reflected\n");
        printf("    Direction: %lf , %lf, %lf\n", refl.direction.x,  refl.direction.y, refl.direction.z);
        
        Ray toLight = Ray(refl.origin,  light.source);
        printf("    ToLight: %lf , %lf, %lf\n", toLight.direction.x,  toLight.direction.y, toLight.direction.z);
        
        printf("Created incoming Rays\n");
        
        Point normal = s->normal(refl.origin);
        
        RGB color = traceLight(refl, numBounces-1, solids, light, n);
        incoming[0] = LightRay(refl, color);
        printf("Color: %d\n", incoming[0].color.r);
        
        //incoming[0] = LightRay(refl, traceLight(refl, numBounces-1, solids, light, n));
        
        printf("Tracing reflection\n");
        toLight.direction.dot(s->normal(toLight.origin));
        /*
        if(toLight.direction.dot(s->normal(toLight.origin)) >= 0 - .0001){
            incoming[1] = LightRay(toLight, traceLight(toLight, numBounces-1, solids, light, n));
        } 
        
        else {
            incoming[1] = LightRay(refl, traceLight(refl, numBounces-1, solids,light,n));
        }
        
        return calcColor(incoming, 2, s, r.opposite().direction);
        
        return BLACK;
    }
    
*/
    
    __global__
    void trace(RGB* pixels, Ray* r, Solid** solids, Light light, unsigned n) {
        unsigned blockId = blockIdx.x + blockIdx.y * gridDim.x; 
        unsigned tid = blockId * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;

        pixels[tid] = traceLightNoRecursion(r[tid], 1, solids, light, n);
    }




#endif /* Scene_hpp */

