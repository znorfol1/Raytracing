//
//  Camera.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Camera_hpp
#define Camera_hpp

#include "Ray.hpp"
#include "Scene.hpp"
#include "Image.hpp"
#include <iostream>

__global__
void initRand(curandState* state){
    unsigned blockId = blockIdx.x + blockIdx.y * gridDim.x; 
    unsigned tid = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    curand_init(0, tid, 0, state+tid);
}
    

/*
 Defines the Camera class. A Camera is defined by its focal length (mimicing the focal length
 of a real camera) and the dimensions of its viewport. The viewport is the rectangle through
 which light can pass through to reach the Camera. By default it is initalized to a 16:9 aspect
 ratio with y being 1 unit in lenth relative to the coordinate system used by the scene.
 */
class Camera{
    double focalLength;
    double viewPortX;
    double viewPortY;
public:
    
    __host__
    Camera(double f = 1, double x=1.77777777, double y=1): viewPortX(x), viewPortY(y), focalLength(f) {};

    /*
        Uses this camera to produce an image of the scene. The resolution
        should conform to the aspect ratio of the viewport
     
        xRes: x-resolution of the output image
        yRes: y-resolution of the output image
        scene: the Scene to be rendered
        position: the position of the camera relative to the scene
     
        Returns: the Image object storing the results of the render
     */
    __host__
    Image render(Ray position, int xRes = 1920, int yRes = 1080){
        Point xBasis = UnitVec(position.direction.projectedRotate())*(viewPortX/xRes);
        Point yBasis = UnitVec(position.direction.cross(xBasis))*(-viewPortY/yRes);
        Point o = position.origin + position.direction*focalLength - (xRes/2)*xBasis + (yRes/2)*yBasis;
        // o is the location of the top left of the outputted image
        
        RGB* pixels = new RGB[xRes*yRes];
        RGB* GPU_pixels;
        Ray* rays = (Ray*) malloc(sizeof(Ray)*xRes*yRes);
        Ray* GPU_rays;
        curandState* states;
        
        cudaMalloc(&GPU_pixels, sizeof(RGB)*(xRes*yRes));
        cudaMalloc(&GPU_rays, sizeof(Ray)*xRes*yRes);
        cudaMalloc(&states , xRes*yRes*sizeof(curandState));
        
        float timeMS = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        for(int j = 0; j < yRes; j++){
            for (int i = 0; i < xRes; i++){
                Ray p(position.origin, o + i*xBasis - j*yBasis);
                rays[j * xRes + i] = p;
            }
        }
        
        dim3 blockSize = dim3(8,8);
        dim3 gridSize = dim3(xRes/8, yRes/8);      
        
        cudaMemcpy(GPU_rays, rays, sizeof(Ray)*xRes*yRes, cudaMemcpyHostToDevice);
        initRand<<<gridSize, blockSize>>>(states);   
        
        cudaEventRecord(start, 0);
        trace<<<gridSize, blockSize>>>(GPU_pixels, GPU_rays, GPU_SOLIDS, GPU_SOLIDS_SIZE, GPU_LIGHT, states);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeMS, start, stop);
                   
        cudaMemcpy(pixels, GPU_pixels, sizeof(RGB)*xRes*yRes, cudaMemcpyDeviceToHost);
        
        printf("Render time taken: %.3lf seconds\n\n", timeMS/1000);
        delete[] rays;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(GPU_pixels);
        cudaFree(GPU_rays);
        cudaFree(states);
        
        Image img(xRes, yRes, pixels);
        return img;
    }
    
};

#endif /* Camera_hpp */

