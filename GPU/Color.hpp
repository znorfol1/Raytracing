//
//  Color.hpp
//  Simple Rendering
//
//  Created by Zachary Norfolk on 5/1/21.
//

#ifndef Color_hpp
#define Color_hpp

/*
 Struct representing an RGB valued color.
 */
struct RGB{
    
    unsigned char r;
    unsigned char g;
    unsigned char b;
    
    __host__  __device__
    RGB(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0): r(r), g(g), b(b) {};
};

/*
 Defines some common colors for ease of use.
 */
namespace Color{
    const static RGB WHITE(255, 255, 255);
    const static RGB BLACK(3, 3, 3);
    const static RGB RED(240, 65, 97);
    const static RGB ORANGE(255, 85, 51);
    const static RGB YELLOW(255, 188, 5);
    const static RGB GRAY(125, 125, 125);
    const static RGB BLUE(24, 61, 182);
    const static RGB PINK(200, 40, 150);
    const static RGB PURPLE(201, 59, 245);
    const static RGB GREEN(56, 245, 122);
    const static RGB ATMOSPHERE(120,120,110);
    const static RGB SHADOW(50,50,50);
}

#define WHITE RGB(255,255,255)
#define BLACK RGB(3,3,3)

#endif /* Color_hpp */


