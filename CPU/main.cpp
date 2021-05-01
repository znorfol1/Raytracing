//
//  main.cpp
//  Simple Rendering
//
//  Created by Zachary Norfolk on 3/26/21.
//

#include "Solid.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Plane.hpp"
#include "Sphere.hpp"
#include <iostream>

int main(int argc, const char * argv[]) {
    Point c(0, 1, 0);
    Ray r1(Point(0,0,0), c);
    
    Sphere s(c, 1, Color::ORANGE);
    Disk d(r1, 3, Color::GRAY);
    Camera cam(1);  
    Light light(Point(0,3,1));
    Ray camPos(Point(4, 3, 4),  c);

    Scene scene;
    scene.add(s);
    scene.add(d);
    scene.add(light);
    
    cam.render(1920, 1080, scene, camPos).writeTo("/Output/out.ppm", 256);
    
    return 0;
}
