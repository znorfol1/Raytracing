//
//  Point.hpp
//  Raytracing
//
//
//  Created by Zachary Norfolk on 3/26/21.
//

#ifndef Point_hpp
#define Point_hpp

#include <cmath>

/*
 Defines the Point class. A Point is a 3D point/vector supporting most standard mathematical operations.
 There is a special constant POINT_NAN which is the Point equivalent of NaN
 */
class Point{

public:
    
    double x;
    double y;
    double z;

    Point(double x=0, double y=0, double z=0): x(x), y(y), z(z) {};
    
    Point(const Point &p): Point(p.x, p.y, p.z){};
    
    double dot(const Point &v) const{
        return x*v.x + y*v.y + z*v.z;
    }
    
    Point cross(const Point &v) const{
        return Point(y*v.z-z*v.y, z*v.x-x*v.z,x*v.y-y*v.x);
    }
    
    double norm() const{
        return sqrt(dot(*this));
    }
        
    Point operator+(const Point &p) const{
        return Point(x+p.x, y+p.y, z+p.z);
    }
    
    Point operator-(const Point &p) const{
        return Point(x-p.x, y-p.y, z-p.z);
    }
    
    Point operator*(const double a) const{
        return Point(a*x, a*y, a*z);
    }
    
    Point operator/(const double a) const{
        return Point(x/a, y/a, z/a);
    }
    
    bool operator==(const Point &p) const{
        return x==p.x && y==p.y && z==p.z;
    }
    
    bool operator!=(const Point &p) const{
        return !(*this == p);
    }
    
    double distanceTo(const Point &p) const{
        return sqrt((p.x-x)*(p.x-x) + (p.y-y)*(p.y-y) + (p.z-z)*(p.z-z));
    }
    
    Point projectedRotate() const{
        return Point(-z, 0, x);
    }
    
    static bool isNan(const Point &p){
        return std::isnan(p.x) && std::isnan(p.y) && std::isnan(p.z);
    }
};

static const Point POINT_NAN(NAN,NAN,NAN);

static Point operator*(const double a, const Point &p) {
    return p*a;
}



/*
 Subclass of a Point representing a vector of length 1.
 */
class UnitVec: public Point {
    
public:
    
    UnitVec(double x=1, double y=0, double z=0): Point(Point(x, y, z)/Point(x, y, z).norm()){};
    
    UnitVec(const Point &p): Point(p/p.norm()) {};
    
    /*
     Reflects a given UnitVec through the plane normal to this UnitVec.
     */
    UnitVec reflect(const UnitVec& v) const {
        double a = (1-2*x*x)*v.x - 2*x*y*v.y - 2*x*z*v.z;
        double b = (1-2*y*y)*v.y - 2*y*x*v.x - 2*y*z*v.z;
        double c = (1-2*z*z)*v.z - 2*z*x*v.x - 2*z*y*v.y;
        return UnitVec(a, b, c);
    }
};

#endif /* Point_hpp */
