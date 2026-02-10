# Bird Watcher (placeholder name - for now, the repo is still called EAGLE_Lensing)

This work-in-progress pipeline aims to take hydrodynamical simulation outputs, i.e. galaxy catalogues and particle datasets, locate gravitational lenses, and simulate lensed images by computing the lensing effects of the single particles. The lensing code relies on the lenstronomy framework. 
The place-holder name references the fact that this code was developed on the EAGLE simulation, following a similar approach to the SEAGLE project, and aims to be flexible enough to be easily adapted to the COLIBRE simulation.


## Quickstart

```bash
git clone git@github.com:GiacomoQueirolo/EAGLE_Lensing
cd EAGLE_Lensing
echo "for the following 2 commands you can alternatively use conda instead of mamba"
mamba env create -f bird_watcher_env.yaml
mamba activate bird_watcher_env
pip install -e .
jupyter notebook Tutorial.ipynb 
```

