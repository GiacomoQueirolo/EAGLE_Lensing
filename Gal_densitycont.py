# for now this programs will plot the density contours of a given galaxy,
# inspired heavily by create_isocont
from Gen_PM_PLL import LensPart,kwlens_part_AS,cutoff_radius,z_source_max,pixel_num
from python_tools.get_res import LoadClass
from remade_gal import get_rnd_NG




def get_lens(path_gal_name=None,reload=True ,pixel_num=pixel_num):
    if path_gal_name is None:
        Gal = get_rnd_NG()
    else:
        Gal = LoadClass(path_gal_name)
    lensGal = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       cutoff_radius=cutoff_radius,z_source_max=z_source_max, 
                       pixel_num=pixel_num,reload=reload)
    lensGal.run()
    return lensGal

from create_isocont import plot_dens_map_hist
if __name__ == "__main__":
    
    #Lens = get_lens()
    #Gal = LoadClass(
    Lens = get_lens(path_gal_name="/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_23/Gn3SGn0.pkl")
    name_proj = Lens.savedir+"/proj_mass.pdf"
    plot_dens_map_hist(Lens.Gal,Lens.proj_index,Lens.pixel_num,cutoff_radius=Lens.cutoff_radius,namefig=name_proj)