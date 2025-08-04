import bhnerf
import bhnerf.constants as consts
import numpy as np
import os
from datetime import datetime
from astropy import units
import ehtim as eh
from bhnerf.optimization import LogFn
import sys

"""
Generate synthetic observations of a hot-spot
"""
fov_M = 16.0
spin = 0.2
inclination = np.deg2rad(60.0)      
nt = 64

array = 'ngEHT'             
flux_scale = 0.1                    # scale image-plane fluxes to `reasonable` values in Jy
tstart = 2.0 * units.hour           # observation start time
tstop = tstart + 40.0 * units.min   # observation stop time

# Compute geodesics (see Tutorial1)
geos = bhnerf.kgeo.image_plane_geos(
    spin, inclination, 
    num_alpha=64, num_beta=64, 
    alpha_range=[-fov_M/2, fov_M/2],
    beta_range=[-fov_M/2, fov_M/2]
)
Omega = np.sign(spin + np.finfo(float).eps) * np.sqrt(geos.M) / (geos.r**(3/2) + geos.spin * np.sqrt(geos.M))
t_injection = -float(geos.r_o)

# Generate hotspot measurements (see Tutorial2) 
emission_0 = flux_scale * bhnerf.emission.generate_hotspot_xr(
    resolution=(64, 64, 64), 
    rot_axis=[0.0, 0.0, 1.0], 
    rot_angle=0.0,
    orbit_radius=5.5,
    std=0.7,
    r_isco=bhnerf.constants.isco_pro(spin),
    fov=(fov_M, 'GM/c^2')
)
obs_params = {
    'mjd': 57851,                       # night of april 6-7, 2017
    'timetype': 'GMST',
    'nt': nt,                           # number of time samples 
    'tstart': tstart.to('hr').value,    # start of observations
    'tstop': tstop.to('hr').value,      # end of observation 
    'tint': 30.0,                       # integration time,
    'array': eh.array.load_txt('../eht_arrays/{}.txt'.format(array))
}
obs_empty = bhnerf.observation.empty_eht_obs(**obs_params)
fov_rad = (fov_M * consts.GM_c2(consts.sgra_mass) / consts.sgra_distance.to('m')) * units.rad
psize = fov_rad.value / geos.alpha.size 
obs_args = {'psize': psize, 'ra': obs_empty.ra, 'dec': obs_empty.dec, 'rf': obs_empty.rf, 'mjd': obs_empty.mjd}
t_frames = np.linspace(tstart, tstop, nt)
image_plane = bhnerf.emission.image_plane_dynamics(emission_0, geos, Omega, t_frames, t_injection)
movie = eh.movie.Movie(image_plane, times=t_frames.value, **obs_args)
obs = bhnerf.observation.observe_same(movie, obs_empty, ttype='direct', seed=None)
"""
Optimize network paremters to recover the 3D emission (as a continuous function) from observations 
Note that logging is done using tensorboardX. To view the tensorboard (from the main directory):
    `tensorboard --logdir runs`
"""
batchsize = 6
z_width = 4                # maximum disk width [M]
rmax = fov_M / 2           # maximum recovery radius
rmin = float(geos.r.min()) # minimum recovery radius
SEED = int(sys.argv[1])
hparams = {'num_iters': 5000, 'lr_init': 1e-4, 'lr_final': 1e-6, 'seed': SEED}

# Checkpointing
current_time = datetime.now().strftime('%Y-%m-%d.%H:%M:%S')
runname = 'tutorial4/recovery.vis.{}'.format(current_time)

# Observation parameters 
chisqdata = eh.imaging.imager_utils.chisqdata_vis
train_step = bhnerf.optimization.TrainStep.eht(t_frames, obs, movie.fovx(), movie.xdim, chisqdata)

# Optimization
predictor = bhnerf.network.NeRF_Predictor(rmax, rmin, rmax, z_width)
raytracing_args = bhnerf.network.raytracing_args(geos, Omega, t_injection, t_frames[0])
optimizer = bhnerf.optimization.Optimizer(hparams, predictor, raytracing_args, checkpoint_dir='../checkpoints/ensemble/seed{}_{}'.format(SEED, runname))
optimizer.run(batchsize, train_step, raytracing_args)