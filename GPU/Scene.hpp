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
#include <curand_kernel.h>

#define NUM_SAMPLES 5
#define NUM_BOUNCES 3

#define ATMOSPHERE_COLOR RGB(120,120,120)
#define SHADOW_COLOR  RGB(40,40,40)


/*
 Defines the Scene class. A Scene is the container that holds Solids and Lights which will be
 used to render the image. Solids and Lights can be created independently and registered to the
 Scene using add(). To be rendered, the scene must be passed to a Camera object.
*/


static Solid** GPU_SOLIDS = NULL;
static unsigned GPU_SOLIDS_SIZE = 0;
static Light GPU_LIGHT;



    __host__
    void createSceneGPU(unsigned n){
        assert(GPU_SOLIDS == NULL);
        
        cudaMalloc(&GPU_SOLIDS, sizeof(Solid*) * n);
        
    }
    
    __host__
    void createLightGPU(Point s, RGB c){
        GPU_LIGHT = Light(s);
    }

    
    /*
     Finds the first object the given ray intersects.
     If none are hit, NULL is returned.
     */
    __device__
    Solid* closestIntersection(Ray r, Solid** solids, int n) {
        Solid* closest = NULL;
        double dist = INFINITY;
        for(int i = 0; i < n;  i++){
            Point p = solids[i]->intersect(r);
            if(Point::isNan(p)){
                continue;
            }
            double d = p.distanceTo(r.origin);
            if(d < dist){
                closest = solids[i];
                dist = d;
            }
        }
        return closest;
    }
    
    /*
     Generates a random double between -d and d
     */
    __device__
    double ran(curandState* state,  double d = 1){
        return 2*d*curand_uniform(state)-d;
    }
    
    /*
     Generates a random ray off a surface given an intersection point.
     
     i: The Point on the surface
     n: the normal to the surface
     b1: the first local basis element to the surface
     b2: the second local basis element to the  surface
     */
    __device__
    Ray randomRay(curandState* state, Point i, UnitVec n, UnitVec b1, UnitVec b2, double d = 1){
        return Ray(i, i+n + ran(state, d)*b1 + ran(state, d)*b2);
    }
    
    /*
     Gives a UnitVec living on the plane normal to the input vector
     */
    __device__
    UnitVec getBasis1(UnitVec n){
        if(n.x != 0){
            double a = -n.dot(Point(0,1,1))/n.x;
            return UnitVec(a, 1, 1);
        }
        return UnitVec(1, 0, 0);
    }
    
    /*
     Given an array of incoming LightRays, and an outwards direction, returns the outgoing color.
     */
    __device__
    RGB calcColor(LightRay* in, Solid* s, UnitVec out){
        double prob[NUM_SAMPLES];
        double sum = 0;
        for(int i = 0; i < NUM_SAMPLES; i++){
            Ray incoming(in[i].r.origin + in[i].r.direction, in[i].r.origin);
            //prob[i] = s->brdf(incoming, out);
            prob[i] = 1;
            prob[i] *= in[i].r.direction.dot(s->normal(in[0].r.origin));
            sum+=prob[i];
        }
        for(int i = 0; i < NUM_SAMPLES; i++){
            prob[i] /= sum;
        }
        double r = 0;
        double g = 0;
        double b = 0;
        for(int i = 0; i < NUM_SAMPLES; i++){
            double c = prob[i];
            r += c * in[i].color.r;
            g += c * in[i].color.g;
            b += c * in[i].color.b;
        }
        return RGB(r*(s->color.r/255.0), g*(s->color.g/255.0), b*(s->color.b/255.0));
    }

    /*
     Traces the ray throughout the scene with the specified number of bounces and number of random samples
     */
    __device__
    RGB traceLight(Ray r, int numBounces, Solid** solids, int n, Light light, curandState* state) {
    
        LightRay incoming[NUM_SAMPLES];
        Solid* s = closestIntersection(r, solids, n);
    
        if(s == NULL){
            if(light.hits(r)){
                return light.color;
            }
            return ATMOSPHERE_COLOR;
        }
        if(numBounces == 0){
            return SHADOW_COLOR;
        }

        Ray refl = s->reflect(r);
        Ray toLight = Ray(refl.origin,  light.source + Point(ran(state, .2),0, ran(state, .2)));
        UnitVec nor = s->normal(refl.origin);
        UnitVec b1 = getBasis1(nor);
        UnitVec b2 = nor.cross(b1);

        int count = 0;
        if(toLight.direction.dot(s->normal(toLight.origin)) >= 0 - .0001 && closestIntersection(toLight, solids, n)==NULL){
            incoming[0] = LightRay(toLight, traceLight(toLight, numBounces-1, solids, n, light, state));
            count++;
        }
        for(int i = count; i < NUM_SAMPLES; i++){
            Ray r1 = randomRay(state,refl.origin, nor, b1, b2);
            incoming[i] = LightRay(r1,traceLight(r1, numBounces - 1, solids, n, light, state));
        }
        return calcColor(incoming, s, r.opposite().direction);
    }
    

    __global__
    void trace(RGB* pixels, Ray* r, Solid** solids, int n, Light light, curandState* state) {
        unsigned blockId = blockIdx.x + blockIdx.y * gridDim.x; 
        unsigned tid = blockId * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;

        pixels[tid] = traceLight(r[tid], 3, solids,n,light, state+tid);
    }

#endif /* Scene_hpp */

