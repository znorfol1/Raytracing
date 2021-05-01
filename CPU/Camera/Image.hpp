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

/*
 Defines the RGB struct and Image class. The RGB struct represents an RGB color value. The Color
 namespace contains common color values as RGB objects. 
 */

struct RGB{
    int r;
    int g;
    int b;
    
    RGB(int r, int g, int b): r(r), g(g), b(b) {};
    RGB():r(0), g(0), b(0) {};
    
    RGB scale(double x) const{
        if (x < 0){
            x = 0;
        } else if(x > 1){
            x = 1;
        }
        double a= 255/2;
        if(r==125 &&b==125  && g==125){
            return RGB(scaleFunc2(r, x), scaleFunc2(g, x), scaleFunc2(b, x));
        }
        return RGB(scaleFunc(r, x), scaleFunc(g, x), scaleFunc(b, x));
    }
private:
    static int scaleFunc(double orig, double x){
        double a = 4*255;
        double b = -4*orig-a;
        double c = -.75*a-b;
        double y = a*x*x*x + b*x*x + c*x;
        return y/2+20;
        
    }
                       
                       static int scaleFunc2(double orig, double x){
                           double a = 4*255;
                           double b = -4*orig-a;
                           double c = -.75*a-b;
                           double y = a*x*x*x + b*x*x + c*x;
                           return y/3+40;
                           
                       }
};

namespace Color{
    const static RGB WHITE(255, 255, 255);
    const static RGB BLACK(3, 3, 3);
    const static RGB VOID(36, 36, 36);
    const static RGB ORANGE(255, 188, 5);
    const static RGB GRAY(125, 125, 125);
const static RGB BLUE(24, 61, 182);
}



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
   
    void writeTo(std::string filename, int max){
        FILE *f = fopen(filename.data(), "w");
        fprintf(f, "P3 %d %d %d\n", x, y, max);
        for(int i = 0; i < y; i++){
            for(int j = 0; j < x; j++){
                RGB p = pixels[i*x + j];
                fprintf(f,"%d %d %d ", p.r, p.g, p.b);
            }
            fputc('\n', f);
        }
        fclose(f);
    }
};

#endif /* Image_hpp */
