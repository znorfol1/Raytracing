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

    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

    __global__
    void initRand(curandState* state){
        unsigned blockId = blockIdx.x + blockIdx.y * gridDim.x; 
        unsigned tid = blockId * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        curand_init(0, tid, 0, state+tid);
    }
    
   

/*
 Defines the Camera class. A Camera is defined by its focal length (minicing the focal length
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
     Renders the Scene with this Camera placed in the specified Ray position. The Camera will be pointing in the direction
     given by the Ray position, and will be level to the  xz plane.
     */
    __host__
    Image render(Ray position, int xRes = 1920, int yRes = 1080){
        //xRes = 128;
        //yRes = 72;
 
        Point xBasis = UnitVec(position.direction.projectedRotate())*(viewPortX/xRes);
        Point yBasis = UnitVec(position.direction.cross(xBasis))*(-viewPortY/yRes);
        Point o = position.origin + position.direction*focalLength - (xRes/2)*xBasis + (yRes/2)*yBasis;
        
        // o is the location of the top left of the outputted image
        RGB* pixels = new RGB[(int)xRes* (int)yRes];
        Ray* rays = (Ray*)  malloc(sizeof(Ray)*(int)xRes*(int)yRes);
        RGB* GPU_pixels;
        Ray* GPU_rays;
        cudaMalloc(&GPU_pixels, sizeof(RGB)*(xRes*yRes));
        cudaMalloc(&GPU_rays, sizeof(Ray)*xRes*yRes);
        
        curandState* states;
        cudaMalloc(&states , xRes*yRes*sizeof(curandState));
        
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
        trace<<<gridSize, blockSize>>>(GPU_pixels, GPU_rays, GPU_SOLIDS, GPU_SOLIDS_SIZE, GPU_LIGHT, states);
                   
        cudaMemcpy(pixels, GPU_pixels, sizeof(RGB)*xRes*yRes, cudaMemcpyDeviceToHost);
        delete[] rays;
        Image img(xRes, yRes, pixels);
        return img;
    }
};

#endif /* Camera_hpp */

