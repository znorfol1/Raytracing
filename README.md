# Raytracing
Author: Zachary Norfolk

Description: This is a simple Raytracer coded from scratch in both C++14 and CUDA. One version runs solely on the CPU, the other uses the GPU to parallelize. 

The progect is currently: **INCOMPLETE**

## Usage:
All the code is very object-oriented. A Scene is the general container that holds the objects to be rendered. These objects can be divided into Lights and Solids. Lights are the objects which emit light into the Scene, and Solids are the objects with which light interacts. Once all the Lights and Solids have been placed in a Scene, a Camera must be created to extract a rendered image. 
