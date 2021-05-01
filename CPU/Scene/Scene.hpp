//
//  Scene.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Scene_hpp
#define Scene_hpp

#include <vector>
#include "Solid.hpp"
#include "Light.hpp"
#include "Image.hpp"

/*
 Defines the Scene class. A Scene is the container that holds Solids and Lights which will be
 used to render the image. Solids and Lights can be created independently and registered to the
 Scene using add(). To be rendered, the scene must be passed to a Camera object.
*/

class Scene{

public:
    std::vector<Solid*> solids;
    std::vector<Light*> lights;

    Scene(std::vector<Solid*> solids, std::vector<Light*> lights): solids(solids), lights(lights){};
    Scene(): solids(), lights() {};
    
    void add(Solid &s){
        solids.push_back(&s);
    }
    
    void add(Light &b){
        lights.push_back(&b);
    }
    
    Solid* closestIntersection(Ray r) const{
        Solid* closest = NULL;
        double dist = INFINITY;
        for(Solid* a : solids){
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
    
    RGB trace(Ray r) const{
        Solid* s = closestIntersection(r);
        if(s == NULL){
            return Color::VOID;
        }
        Ray reflection = s->reflect(r);
        Ray toLight(reflection.origin, lights[0]->source);
        if(closestIntersection(toLight) != NULL){
            return Color::BLACK;
        }
        return s->color.scale(reflection.direction.dot(toLight.direction));
    }
    
};

#endif /* Scene_hpp */
