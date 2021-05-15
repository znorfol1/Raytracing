//
//  Image.hpp
//  Raytracing
//
//  Created by Zachary Norfolk on 4/1/21.
//

#ifndef Image_hpp
#define Image_hpp

#include <stdio.h>
#include <string>
#include "Color.hpp"

/*
 An Image represents a PPM image file loaded in memory. It can be written to an output location using
 the function writeTo().
*/
class Image{
    
    unsigned int x;
    unsigned int y;
    RGB* pixels;
    
public:
 
    Image(int x=1920, int y=1080): x(x), y(y) {
        pixels = new RGB[x*y];
    }
    
    ~Image(){
        delete[] pixels;
    }
    
    RGB* operator[](unsigned const int i){
        return pixels+x*i;
    }
   
    void writeTo(std::string filename){
        FILE *f = fopen(filename.data(), "w");
        fprintf(f, "P3 %d %d %d\n", x, y, 255);
        for(int i = 0; i < y; i++){
            for(int j = 0; j < x; j++){
                RGB p = pixels[i*x + j];
                fprintf(f,"%d %d %d\n", p.r, p.g, p.b);
            }
            fputc('\n', f);
        }
        fclose(f);
    }
};

#endif /* Image_hpp */
