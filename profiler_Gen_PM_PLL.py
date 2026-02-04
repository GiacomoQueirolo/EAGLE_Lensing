from pyinstrument import Profiler

from python_tools.get_res import LoadClass
from particle_galaxy import get_rnd_PG
from Gen_PM_PLL import LensPart,kwlens_part_AS,cutoff_radius,z_source_max,pixel_num

print("test Gen_PM_PLL.LensPart with a random galaxy")
Gal = get_rnd_PG()
profiler = Profiler()
profiler.start()


mod_LP = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       cutoff_radius=cutoff_radius,z_source_max=z_source_max,
                       pixel_num=pixel_num,reload=False)
#float(profiler.output_text().split("Model_PART.get_lensed_image")[0].split("|-")[-1])
profiler.stop()
print("First: loading data")
print(profiler.output_text(color=True,show_all=False))
profiler.start()
mod_LP.run()
profiler.stop()
print("Second: Computing lens model")
print(profiler.output_text(color=True,show_all=False))
