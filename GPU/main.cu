
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
    createSceneGPU(2);
    
    Ray r1(Point(0,0,0), Point(0,1,0));
    createPlaneGPU(r1, RGB(40,40,40));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
 
    createSphereGPU(Point(0,1,0), 1, RGB(255, 188, 5));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    createLightGPU(Point(0,6,0), WHITE);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    
    Camera cam;
    cam.render(Ray(Point(3,5,3), Point(0,1,0))).writeTo("out.ppm");
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
