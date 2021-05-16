//
//  main.cpp
//  Simple Rendering
//
//  Created by Zachary Norfolk on 3/26/21.
//

#include "Scene.hpp"
#include "Camera.hpp"
#include "Plane.hpp"
#include "Sphere.hpp"
#include <iostream>

int main(int argc, const char * argv[]) {
    
    //Example Scene
    
    //Create Solids
    Plane floor(Ray(Point(), Point(0,1,0)), Color::WHITE);
    Sphere s1(Point(0,1,0),1, Color::YELLOW);
    Sphere s2(Point(-4,1,1), 1, Color::GREEN,.5);
    Sphere s3(Point(2,.2,1), .2, Color::PURPLE,.5);
    
    //Create Light
    Light light(Point(0,4,1));
    
    //Add to Scene
    Scene scene;
    scene.add(floor);
    scene.add(s1);
    scene.add(s2);
    scene.add(s3);
    scene.add(light);
    
    //Create camera
    Camera cam(1);
    Ray camPosition(Point(4,3,4), Point(0,1,0));
    
    //Render and write to file
    cam.render(1920, 1080, scene, camPosition).writeTo("/Users/zacharynorfolk/Projects/Simple Rendering/Output/out.ppm");
    
    return 0;
}
