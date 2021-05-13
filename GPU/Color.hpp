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
    int r;
    int g;
    int b;
    
    __host__  __device__
    RGB(int r = 0, int g = 0, int b = 0): r(r), g(g), b(b) {};
};

/*
 Defines some common colors for ease of use.
 */

#define WHITE RGB(255,255,255)
#define BLACK RGB(3,3,3)


#endif /* Color_hpp */

