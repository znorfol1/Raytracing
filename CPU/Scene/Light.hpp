//
//  Light.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Light_hpp
#define Light_hpp

#include "Image.hpp"
#include "Point.hpp"

struct Light{
    
    double luminosity = 0;
    RGB color = Color::WHITE;
    Point source;
    
    Light(Point s): source(s){};
    

};

#endif /* Light_hpp */
