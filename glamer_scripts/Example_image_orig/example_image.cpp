/* ParticleExample
  no recenter on particle files

  copied from https://github.com/glenco/ExampleImage/blob/master/example_image.cpp
  */

#include <slsimlib.h>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <thread>
#include <mutex>
#include "gridmap.h"
#include <math.h>

using namespace std;

/**
 
 This is a simple example program that uses the GLAMER library to make an 
 image of lensed object.  The source is an image of a galaxy read in from 
 a fits file.  The lens is an NFW profile.
 
 This example was written by Giulia Despali, July 2017
 
 Further information on all classes can be found at:
 http://glenco.github.io/glamer/Doc/html/
 
 Other examples that do more complicated things can be found at:
 https://github.com/glenco
 
 **/

int main(int arg,char **argv){
  
  // set cosmology
  // COSMOLOGY cosmo(Planck1yr);
  COSMOLOGY cosmo(CosmoParamSet::Planck1yr);
  Point_2d rotation_vector;
  PosType zl=0.5;
  PosType zs = 2; //** redshift of source
  long seed = -28976391;
  // create lens with no mass in it yet
  Lens lens(&seed,zs,cosmo,false);
  // create an NFW halo
  LensHaloNFW halo(1.e+15,1,zl,5,1.,0.,cosmo);
  // set position of halo in angular coordinates
  halo.setTheta(0,0);
  // put halo into lens
  lens.insertMainHalo(halo,true,true);

  PixelMap<float> map_lens;

  PosType factor = 1;
  
  // read in an image of the source
  //const PixelMap map_source("galaxy.fits",4.8e-7);
  const PixelMap<float> map_source("galaxy.fits",4.8e-7);
  std::cout << " read source from fits "  << std::endl;
  // convert this image into a source class
  //SourcePixelled source(map_source,zs,factor);
  PosType zero_point=30 ; // the magnitude zero point -> arbitrary
  SourcePixelled source(map_source,zs,factor,zero_point);
  std::cout << " sourcePixelled created"  << std::endl;
  

  Point_2d center;
  center[0] = 0.0;
  center[1] = 0.0;

  PosType pi = 3.14159 ;
  // shoot a regular grid of rays through the lens
  GridMap gridmap(&lens,512,center.x,0.25*pi/180/10);
  // add the surface brightnesses of the source
  gridmap.RefreshSurfaceBrightnesses(&source);

  std::cout << " gridmap done"  << std::endl;

  // output image to a PixelMap
  PixelMap<float> map_image = gridmap.getPixelMapFlux<float>(); //1.0);

  std::cout << "pixelmap for image created"  << std::endl;

  // print final image to a fits file, the "!" will cause it to overwrite the file
  map_image.printFITS("!test_run.fits");

  return 0;
}


