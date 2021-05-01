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
#include "Solid.hpp"
#include "Image.hpp"

/*
 Defines the Camera class. A Camera is defined by its focal length (minicing the focal length
 of a real camera) and the dimensions of its viewport. The viewport is the rectangle through
 which light can pass through to reach the Camera. By default it is initalized to a 16:9 aspect
 ratio with y being 1 unit in lenth.
 */

class Camera{
    double focalLength;
    double viewPortX;
    double viewPortY;
    
public:
    Camera(double f, double x=1.77777777, double y=1): viewPortX(x), viewPortY(y), focalLength(f) {};
    
    Image render(double xRes, double yRes, Scene scene, Ray position){
        Image img(xRes, yRes);
        Point xBasis = UnitVec(position.direction.projectedRotate())*(viewPortX/xRes);
        Point yBasis = UnitVec(position.direction.cross(xBasis))*(-viewPortY/yRes);
        //Calculate basis
        Point o = position.origin + position.direction*focalLength - (xRes/2)*xBasis + (yRes/2)*yBasis;
        
        for(int i = 0; i < xRes; i++){
            for (int j = 0; j < yRes; j++){
                Ray p(position.origin, o + i*xBasis - j*yBasis);
                img[j][i] = scene.trace(p);
            }
        }
        return img;
    }
};

#endif /* Camera_hpp */
