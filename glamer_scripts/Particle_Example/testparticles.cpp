 /*
  */

#include <slsimlib.h>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <thread>
#include <mutex>

#include "particle_halo.h"
#include "gridmap.h"

using namespace std;

int main(int arg,char **argv){
  
  COSMOLOGY cosmo(CosmoParamSet::Planck1yr);
  Point_2d rotation_vector(0,0);
  PosType zl=0.4;           // redshift of lens
  PosType z_source = 2.0;           // redshift of lens
  int Nsmooth = 16; // number of neighbors for smoothing scale

  //LensHaloParticles phalo("particles.dm.txt",zl,Nsmooth,cosmo,rotation_vector, true, true);
  
  long seed = -28976391;
  /**********************************************************/
   
  Lens lens(&seed,z_source,cosmo);

  Point_3d<double> center;
  {
      // rotation of the halo about its center of mass
    rotation_vector[0] = PI/2;
    rotation_vector[1] = PI/5;

    LensHaloParticles<ParticleType<float> > halo("particles.dm.txt"
                                                 ,SimFileFormat::ascii
                                                 ,zl
                                                 ,Nsmooth
                                                 ,cosmo
                                                 ,rotation_vector
                                                 ,true
                                                 ,true
                                                 ,0);
    
    //center = halo.CenterOfMass();
    
    // insert halos into lens
     // this is moved instead of inserted to avoid a copy
    lens.moveinMainHalo(halo, true);
    
    // Another way of creating particle halos that is more flexible with
    // more file types is commented out here.
//    MakeParticleLenses halomaker("particles.dm.txt",ascii,Nsmooth,true);
//    halomaker.CreateHalos(cosmo,zl);
//    //  There is a seporate halo for each type of particle
//    for(auto h : halomaker.halos){
//       lens.insertMainHalo(h,zl, true);
//    }
//
  }

  // here you can rotate each simulation independently
  //for(int i=0 ; i < lens.getNMainHalos<LensHP>()  ; ++i){
  //  lens.getMainHalo<LensHP>(i)->rotate(theta);
  //}
  
  center /= cosmo.angDist(zl);
 
  std::cout << "making gridmap ... ";
  double range = 30.0 * arcsecTOradians; // range of grids in radians
  GridMap gridmap(&lens,512,center.x,range/2);
  std::cout << "done." << std::endl;

  // set the redshift of the source plane
  lens.ResetSourcePlane(z_source,false);
  
  // output some maps
  gridmap.writeFits<float>(LensingVariable::KAPPA,"!particles_kappa.fits");
  gridmap.writeFits<float>(LensingVariable::INVMAG,"!particles_invmag.fits");
  gridmap.writeFits<float>(LensingVariable::ALPHA1,"!particles_ALPHA1.fits");
  gridmap.writeFits<float>(LensingVariable::ALPHA2,"!particles_ALPHA2.fits");
  gridmap.writeFits<float>(LensingVariable::ALPHA,"!particles_ALPHA.fits");
  
  // find the critical curves
  std::vector<ImageFinding::CriticalCurve> crit_curve;
  ImageFinding::find_crit(lens, gridmap,crit_curve);
  
  
  //*** plot caustic curves
  if(crit_curve.size() > 0){
    Point_2d p1,p2;
    Point_2d tp1,tp2;
    
    //*** find good boundaries for plot
    for(int i=0;i<crit_curve.size();++i){
      crit_curve[i].CausticRange(tp1,tp2);
      if(p1[0] > tp1[0]) p1[0] = tp1[0];
      if(p1[1] > tp1[1]) p1[1] = tp1[1];
      
      if(p2[0] < tp2[0]) p2[0] = tp2[0];
      if(p2[1] < tp2[1]) p2[1] = tp2[1];
      
    }
    Point_2d center = crit_curve[0].caustic_center;
    PixelMap<float> map(center.x,512*2,1.1*MAX(p2[0]-p1[0],p2[1]-p1[1])/512/2);
    
    for(int i=0;i<crit_curve.size();++i){
      //map.AddCurve(crit_curve[i].caustic_curve_outline, i+1);  // this is a outline of the caustic that does not intersect itself
      map.AddCurve(crit_curve[i].caustic_curve_intersecting,i+1);
    }
    map.printFITS("!caustics.fits");
  }
  
  //*** plot critical curves
  
  double crit_range=0;
  if(crit_curve.size() > 0){
    Point_2d p1,p2;
    Point_2d tp1,tp2;
    for(int i=0;i<crit_curve.size();++i){
      crit_curve[i].CritRange(tp1,tp2);
      if(p1[0] > tp1[0]) p1[0] = tp1[0];
      if(p1[1] > tp1[1]) p1[1] = tp1[1];
      
      if(p2[0] < tp2[0]) p2[0] = tp2[0];
      if(p2[1] < tp2[1]) p2[1] = tp2[1];
      
    }
    Point_2d center = crit_curve[0].critical_center;
    crit_range = 1.1*MAX(p2[0]-p1[0],p2[1]-p1[1]);
    PixelMap<float> map(center.x,512,crit_range/512);
    
    for(int i=0;i<crit_curve.size();++i)
      map.AddCurve(crit_curve[i].critcurve,i+1);// .critical_curve,i+1);
    
    map.printFITS("!critical.fits");
  }
  
  std::cout << "Number of caustics : "<< crit_curve.size() << std::endl;
  
  if(crit_curve.size() == 0){
    cout << "Exiting" << endl;
    exit(1);
  }
  //*** print information about the critical curves that were found
  PosType rmax,rmin,rave;
  if(crit_curve.size() > 0){
    std::string type;
    for(int i=0;i<crit_curve.size();++i){
      type = to_string( crit_curve[i].type );
      std::cout << "  " << i << " type " << to_string(crit_curve[i].type) << std::endl;
      crit_curve[i].CausticRadius(rmax,rmin,rave);
      std::cout << "      caustic " << crit_curve[i].caustic_center << " | " << crit_curve[i].caustic_area << " " << rmax << " " << rmin << " " << rave << std::endl;
      crit_curve[i].CriticalRadius(rmax,rmin,rave);
      std::cout << "      critical " << crit_curve[i].critical_center << " | " << crit_curve[i].critical_area << " " << rmax << " " << rmin << " " << rave << std::endl;
    }
  }
  
  //**** put a source in and map its images
  //****************************************
  
  //*** find a source position within the tangential caustic
  Utilities::RandomNumbers_NR random(seed);   //*** random number generator
  std::vector<Point_2d> y;                    //*** vector for source positions
  crit_curve[0].RandomSourcesWithinCaustic(1,y,random); //*** get random points within first caustic

  PosType zs = 2; //** redshift of source
  //** make a Sersic source, there are a number of other ones that could be used
  SourceSersic source(23,0.02,0,1,0.5,zs,23,Band::EUC_VIS);
  
  source.setTheta(y[0]);
  
  /** reset the source plane in the lens from the one given in the
   parameter file to this source's redshift
   */
  lens.ResetSourcePlane(zs,false);
  
  std::vector<ImageInfo> imageinfo;
  int Nimages;

  std::cout << "Mapping source ..." << std::endl;
  
  // we could make a PixelMap<> from the GridMap with  PixelMap<> map = gridmap.writePixelMap(LensingVariable::SurfBrightness);
  // but let us re-center the PixelMap<> on the critical curve
  Point_2d crit_center = crit_curve[0].critical_center;
  PixelMap<float> map(crit_center.x,512,crit_range/512,PixelMapUnits::surfb);

  
  // The following produces an image of the lensed source using the rays that
  // were created within the grid and refined when finding the caustics.
  
  gridmap.RefreshSurfaceBrightnesses(&source);
  map.AddGridMapBrightness(gridmap);
  
  // You can add a source directly to the PixelMap<> without lensing with
  
  source.setTheta(crit_center);
  map.AddSource(source);
  
  map.printFITS("!image_unrefined.fits");
  
  
  return 0;
}
