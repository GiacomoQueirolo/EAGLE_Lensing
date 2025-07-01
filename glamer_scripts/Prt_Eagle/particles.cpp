 /*
     Adapted copy from testparticles.cpp
     We want to take particles from EAGLE simulation
     
         - note we give 
             - z_source (arbitrary)
             - z_lens   (given by EAGLE sim)
             - Nsmoothing -> should be given by EAGLE? -> for now kept at 16
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
  PosType z_source = 2.0;   // redshift of source
  int Nsmooth = 16;         // number of neighbors for smoothing scale
  double range = 30.0 * arcsecTOradians; // range of grids in radians
  int Nx_gridmap = 512 ; //x-size of gridmap -> define the resolution w range
  std::string particle_file = "data/particles_EAGLE.csv" ; // csv4 -> x,y,z,M # could be csv6 w smoothin and particle type but DM has no smoothing scale
  //LensHaloParticles phalo("particles.dm.txt",zl,Nsmooth,cosmo,rotation_vector, true, true);
  
  long seed = -28976391; //no idea why we need this one
  /**********************************************************/
   
  Lens lens(&seed,z_source,cosmo);

  Point_3d<double> center;
  {
      // rotation of the halo about its center of mass -> for now ignored
    rotation_vector[0] = PI/2;
    rotation_vector[1] = PI/5;
    /*
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
    
    */
      
    // Another way of creating particle halos that is more flexible with
    // more file types 
    // csv4 -> CSV ascii format without header.  The first three columns
    // are the positions.  Next columns are used for the other formats being and
    // interpreted as (column 4) masses are in Msun/h, (column 5) the paricle smoothing
    // size in Mpc/h and (column 6) an integer for type of particle.  There can be more
    // columns in the file than are used.  In the case of csv6, when there are more then one
    // type of halo each type will be in a differeent LensHaloParticles with different smoothing.
      
            // -> instead of 1 single file, could I load various files and just "insertMainhalo" multiple times for each of them? -> hard to compute the center then
    MakeParticleLenses halomaker(particle_file,
                                  SimFileFormat::csv4,
                                  Nsmooth,true); 
    // the last true is bool recenter -> recenter so that the LensHalos are centered on the center of mass -> to consider -> should just be arbitrary, st center = 0 ?
    Point_3d <double> c_mass = halomaker.getCenterOfMass();

    //Point_2d center; // doesn't matter if it's 2D or 3D -> only use x
    
    center[0] = c_mass[0];
    center[1] = c_mass[1];

    // the last entry is inv_area, indicated by "// inverse area for mass compensation"
    // where "bool compensate=false"-> if true a negative mass will be included so that  the region wil have zero mass
    // in my case I think it's safe to set it to 0
    halomaker.CreateHalos(cosmo,zl,0);
    //  There is a separate halo for each type of particle
    for(auto h : halomaker.halos){
       //lens.insertMainHalo(*h,zl, true);
        lens.moveinMainHalo(*h,zl, true);
    }
    // DEBUG: arrives here no problems
      
  }
  
  // here you can rotate each simulation independently
  //for(int i=0 ; i < lens.getNMainHalos<LensHP>()  ; ++i){
  //  lens.getMainHalo<LensHP>(i)->rotate(theta);
  //}
    

  center /= cosmo.angDist(zl);
  std::cout << "making gridmap ... ";
  // 2 points: 
    // -gridmap is not adaptive -> less expensive but very likely we will need the adaptive one (called grid?)
    // - the second entry is Nx, the Initial number of grid points in X dimension. -> define w. range the resolution
  GridMap gridmap(&lens,Nx_gridmap,center.x,range/2);
  std::cout << "done." << std::endl;

  // set the redshift of the source plane
  lens.ResetSourcePlane(z_source,false);
  
  // output some maps
  //gridmap.writeFits<float>(LensingVariable::KAPPA,"!particles_kappa.fits");
  //gridmap.writeFits<float>(LensingVariable::INVMAG,"!particles_invmag.fits");
  gridmap.writeFits<float>(LensingVariable::ALPHA1,"!results/particles_ALPHA1.fits");
  gridmap.writeFits<float>(LensingVariable::ALPHA2,"!results/particles_ALPHA2.fits");
  //gridmap.writeFits<float>(LensingVariable::ALPHA,"!results/particles_ALPHA.fits");

  // the following is kept for now to check if it works -> note: for the test it doesn't

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
    cout << "No caustics found - no lensing" << endl;
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
