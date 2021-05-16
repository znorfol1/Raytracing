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
    
    Camera(double f = 1, double x=1.77777777, double y=1): viewPortX(x), viewPortY(y), focalLength(f) {};

    /*
        Uses this camera to produce an image of the scene.
     
        xRes: x-resolution of the output image
        yRes: y-resolution of the output image
        scene: the Scene to be rendered
        position: the position of the camera relative to the scene
     
        Returns: the Image object storing the results of the render
     */
    Image render(unsigned xRes, unsigned yRes, Scene scene, Ray position){
        Image img(xRes, yRes);
        Point xBasis = UnitVec(position.direction.projectedRotate())*(viewPortX/xRes);
        Point yBasis = UnitVec(position.direction.cross(xBasis))*(-viewPortY/yRes);
        Point o = position.origin + position.direction*focalLength - (xRes/2)*xBasis + (yRes/2)*yBasis;
        // o is the location of the top left of the outputted image
        double last = 0;
        for(int i = 0; i < xRes; i++){
            for (int j = 0; j < yRes; j++){
                Ray p(position.origin, o + i*xBasis - j*yBasis);
                img[j][i] = scene.trace(p);
            }
            double progress = (((double)i)/xRes);
            if(progress > last){
                printf("%.0lf percent complete\n", progress*100);
                last += .1;
            }
        }
        return img;
    }
    
};

#endif /* Camera_hpp */
