#include "Scene.hpp"
#include "Ray.hpp"
#include "Plane.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "Sphere.hpp"
#include "Camera.hpp"
    
    
int main() {
    //Set stack
    cudaDeviceSetLimit(cudaLimitStackSize, 32768ULL);

    createSceneGPU(3);
    
    Ray r1(Point(0,0,0), Point(0,1,0));
    createPlaneGPU(r1, RGB(255,255,255));
    createSphereGPU(Point(0,1,0),1, Color::YELLOW);
    createSphereGPU(Point(-4,1,1), 1, Color::GREEN);
    createLightGPU(Point(0,4,1), WHITE);
  
    Camera cam;
    Ray camPosition(Point(4,3,4), Point(0,1,0));
    cam.render(camPosition).writeTo("out.ppm");
       
    return  0;
}
