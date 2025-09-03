from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable
from pathlib import Path
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable
import bhnerf
import bhnerf.network
import importlib
importlib.reload(bhnerf.network)
import bhnerf.network
from bhnerf import constants as consts
from ehtim.image import make_square
from ehtim.imaging import imager_utils as iu
from astropy import units

def build_A_per_frame(obs, t_frames, image_fov_rad, image_size, dtype='vis', pol='I'):
    """
    Args:
        obs: Obsdata object for the black hole
        t_frames: (nt,) astropy quantity array of times for each frame
        image_fov_rad: image field of view in radians
        image_size: image size in pixels
        dtype: chisqdata type
        pol: polarization type.
    
    Returns
        target: (nt, Nvis_t, Npol)
        sigma: (nt, Nvis_t, Npol)
        A: (nt, Nvis_t, image_size**2)
    """
    dt_sec = (t_frames[-1] - t_frames[0]).to('s').value / (len(t_frames) + 1)
    obs_list = obs.split_obs(t_gather=dt_sec)
    prior = make_square(obs, image_size, image_fov_rad)
    chisq_fn = getattr(iu, f'chisqdata_{dtype}')

    targ, sig, A_all = [], [], []
    for ob in obs_list:
        t, s, A = chisq_fn(ob, prior, mask=[], pol=pol)
        targ.append(np.asarray(t))
        sig.append(np.asarray(s))
        A_all.append(np.asarray(A))
    return (np.stack(targ), np.stack(sig), np.stack(A_all))

def make_forward_model(predictor_apply, predictor_params, t_frame, ray_args, A):
    
    t_units = t_frame.unit
    t_frame    = t_frame
    Omega      = ray_args['Omega']
    g          = ray_args['g']
    dtau       = ray_args['dtau']
    Sigma      = ray_args['Sigma']
    t_geos     = ray_args['t_geos']
    J          = ray_args['J']
    t_start    = ray_args['t_start_obs']
    t_inj      = ray_args['t_injection']
    t_units    = t_units 
    
    @jax.jit
    def forward_model(coords):
        x, y, z = jnp.moveaxis(coords, -1, 0)
        coords_list = [x, y, z]

        I_img = bhnerf.network.image_plane_prediction(
            predictor_params, predictor_apply,
            t_frames          = t_frame,
            coords            = coords_list,
            Omega             = Omega,
            J                 = J,
            g                 = g,
            dtau              = dtau,
            Sigma             = Sigma,
            t_start_obs       = t_start,
            t_geos            = t_geos,
            t_injection       = t_inj,
            t_units           = t_units,
        )
        imvec = I_img.reshape(-1)
        vis = (A @ imvec)
        return vis
    return forward_model

def trilinear_no_var(coords: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Trilinear interpolation into *grid* at **normalised** 3-D *coords*.

    Parameters
    ----------
    coords: [..., 3] float32 in [0,1] cube
    grid: [gx, gy, gz, C]

    Returns
    -------
    values : [..., C]
    """
    gx, gy, gz, _ = grid.shape
    x = coords[..., 0] * (gx - 1)
    y = coords[..., 1] * (gy - 1)
    z = coords[..., 2] * (gz - 1)

    i0, j0, k0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32), jnp.floor(z).astype(jnp.int32)
    i1, j1, k1 = jnp.clip(i0 + 1, 0, gx - 1), jnp.clip(j0 + 1, 0, gy - 1), jnp.clip(k0 + 1, 0, gz - 1)

    wx, wy, wz = x - i0, y - j0, z - k0

    def gather(ii, jj, kk):
        return grid[ii, jj, kk]

    # 8 vertices
    c000 = gather(i0, j0, k0)
    c100 = gather(i1, j0, k0)
    c010 = gather(i0, j1, k0)
    c110 = gather(i1, j1, k0)
    c001 = gather(i0, j0, k1)
    c101 = gather(i1, j0, k1)
    c011 = gather(i0, j1, k1)
    c111 = gather(i1, j1, k1)

    c00 = c000 * (1 - wx)[..., None] + c100 * wx[..., None]
    c01 = c001 * (1 - wx)[..., None] + c101 * wx[..., None]
    c10 = c010 * (1 - wx)[..., None] + c110 * wx[..., None]
    c11 = c011 * (1 - wx)[..., None] + c111 * wx[..., None]

    c0 = c00 * (1 - wy)[..., None] + c10 * wy[..., None]
    c1 = c01 * (1 - wy)[..., None] + c11 * wy[..., None]

    return c0 * (1 - wz)[..., None] + c1 * wz[..., None]

def interpolate(coords:jnp.ndarray, grid: jnp.ndarray):
    return jax.scipy.ndimage.map_coordinates(grid, coords, order=1, mode='nearest')

def trilinear_scalar(coords_unit: jnp.ndarray, grid3d: jnp.ndarray) -> jnp.ndarray:
    """coords_unit: (R,R,R,3) in [0,1); grid3d: (R,R,R) -> (R,R,R)"""
    grid4 = grid3d[..., None]
    out4  = trilinear(coords_unit, grid4)
    return out4[..., 0]

def mapcoords_scalar(coords_unit: jnp.ndarray, grid3d: jnp.ndarray) -> jnp.ndarray:
    """coords_unit: (R,R,R,3) in [0,1); grid3d: (R,R,R) -> (R,R,R)"""
    R = grid3d.shape[0]
    xi = coords_unit[..., 0] * (R - 1)
    yi = coords_unit[..., 1] * (R - 1)
    zi = coords_unit[..., 2] * (R - 1)
    idx = jnp.stack([xi, yi, zi], axis=0)
    return jax.scipy.ndimage.map_coordinates(grid3d, idx, order=1, mode='nearest')

def _coords_unit_identity(R: int) -> jnp.ndarray:
    eps = 1e-7
    xs = jnp.linspace(0.0, 1.0 - eps, R)
    ys = jnp.linspace(0.0, 1.0 - eps, R)
    zs = jnp.linspace(0.0, 1.0 - eps, R)
    return jnp.stack(jnp.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)

def interpolation_check_scalar(volume3d: jnp.ndarray, fov: float, title: str = ""):
    """volume3d must be (R,R,R). No channels added; output remains (R,R,R)."""
    import matplotlib.pyplot as plt
    if volume3d.ndim != 3:
        raise ValueError("Pass a scalar volume shaped (R, R, R).")

    R = volume3d.shape[0]
    coords = _coords_unit_identity(R)

    out_tri = trilinear_scalar(coords, volume3d)
    out_map = mapcoords_scalar(coords, volume3d)

    e_tri = float(jnp.max(jnp.abs(out_tri - volume3d)))
    e_map = float(jnp.max(jnp.abs(out_map - volume3d)))
    e_tm  = float(jnp.max(jnp.abs(out_tri - out_map)))

    print(f"[{title}] shape={tuple(volume3d.shape)}")
    print("  max |trilinear - original|   :", e_tri)
    print("  max |map_coords - original|  :", e_map)
    print("  max |trilinear - map_coords| :", e_tm)

    zc = R // 2
    orig = np.asarray(volume3d[:, :, zc])
    tri  = np.asarray(out_tri[:, :, zc])
    mco  = np.asarray(out_map[:, :, zc])

    fig, axs = plt.subplots(2, 3, figsize=(12, 8)); axs = axs.ravel()
    im0 = axs[0].imshow(orig, origin='lower'); axs[0].set_title('Original [z mid]')
    im1 = axs[1].imshow(tri,  origin='lower'); axs[1].set_title('Trilinear [z mid]')
    im2 = axs[2].imshow(mco,  origin='lower'); axs[2].set_title('MapCoords [z mid]')
    im3 = axs[3].imshow(tri - orig, origin='lower'); axs[3].set_title('Tri - Orig (diff)')
    im4 = axs[4].imshow(mco - orig, origin='lower'); axs[4].set_title('Map - Orig (diff)')
    im5 = axs[5].imshow(tri - mco,  origin='lower');  axs[5].set_title('Tri - Map (diff)')
    for ax in axs: ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im0, ax=[axs[0], axs[1], axs[2]], shrink=0.75)
    fig.colorbar(im3, ax=[axs[3], axs[4], axs[5]], shrink=0.75)
    plt.tight_layout(); plt.show()

    import bhnerf.visualization as vis
    vis.ipyvolume_3d(np.asarray(volume3d), fov, level=[0.0, 0.2, 0.7])
    vis.ipyvolume_3d(np.asarray(out_tri),  fov, level=[0.0, 0.2, 0.7])
    vis.ipyvolume_3d(np.asarray(out_map),  fov, level=[0.0, 0.2, 0.7])

    return np.asarray(out_tri), np.asarray(out_map)

# replace with jax.scipy.map_coordinates
# be careful with this function, do sanity checks at diff grid resolutions
def trilinear(coords: jnp.ndarray, grid: jnp.ndarray, variance: bool = False) -> jnp.ndarray:
    """
    Trilinear interpolation into grid at normalized 3-D *coords*.

    Parameters
    ----------
    coords : [..., 3] float32 in [0, 1]^3
    grid   : [gx, gy, gz, C] values at voxel corners
             - For values (e.g., emission): store values in C channels.
             - For variances (e.g., σ_x^2, σ_y^2, σ_z^2): store *variances*
               in C channels. This function will combine corners with squared
               barycentric weights when `variance=True`.

    Returns
    -------
    out : [..., C]
      Interpolated values if variance=False;
      Interpolated variances if variance=True (already in σ^2 units).
    """
    def _index_and_weights(u, size):
        # keep strictly inside so i1 exists
        u = jnp.clip(u, 0.0, 1.0 - 1e-7)
        x = u * (size - 1)
        i0 = jnp.floor(x).astype(jnp.int32)
        i1 = i0 + 1
        w1 = x - i0
        w0 = 1.0 - w1
        return i0, i1, w0, w1

    gx, gy, gz, _ = grid.shape

    i0,i1,wx0,wx1 = _index_and_weights(coords[...,0], gx)
    j0,j1,wy0,wy1 = _index_and_weights(coords[...,1], gy)
    k0,k1,wz0,wz1 = _index_and_weights(coords[...,2], gz)

    # 8 barycentric weights
    w000 = wx0*wy0*wz0 
    w100 = wx1*wy0*wz0
    w010 = wx0*wy1*wz0
    w110 = wx1*wy1*wz0
    w001 = wx0*wy0*wz1
    w101 = wx1*wy0*wz1
    w011 = wx0*wy1*wz1
    w111 = wx1*wy1*wz1

    weights = jnp.stack([w000,w100,w010,w110,w001,w101,w011,w111], axis=-1)
    
    # Var = SUM((weight_i)^2 Var_i)
    if variance: # interpolate variance w squared weights--technically statistically correct
        weights = weights ** 2

    # gather 8 corners
    def gather(ii, jj, kk):
        return grid[ii, jj, kk]

    c000 = gather(i0,j0,k0)
    c100 = gather(i1,j0,k0)
    c010 = gather(i0,j1,k0)
    c110 = gather(i1,j1,k0)
    c001 = gather(i0,j0,k1)
    c101 = gather(i1,j0,k1)
    c011 = gather(i0,j1,k1)
    c111 = gather(i1,j1,k1)

    corners = jnp.stack([c000,c100,c010,c110,c001,c101,c011,c111], axis=-2)
    out = jnp.sum(corners * weights[..., None], axis=-2)
    return out

def flatten(tree):
    return jnp.concatenate([t.ravel() for t in jax.tree_util.tree_leaves(tree)])

class DeformationGrid(nn.Module):
    resolution: tuple[int, ...]

    @nn.compact
    def __call__(self, coords):
        theta = self.param('theta', nn.initializers.zeros, self.resolution + (coords.shape[-1],))
        return trilinear(coords, theta)

def make_coords_unit_weighted(coords, g, dtau, keep=0.99, margin=1.01, eps=1e-6):
    weights = g * dtau
    reduce_axes = tuple(range(coords.ndim - 1))
    thr = jnp.quantile(weights, 1.0 - keep)
    mask = weights >= thr
    masked = jnp.where(mask[..., None], coords, jnp.nan)

    xyz_min = jnp.nanmin(masked, axis=reduce_axes)
    xyz_max = jnp.nanmax(masked, axis=reduce_axes)
    center = 0.5 * (xyz_max + xyz_min)
    half = jnp.maximum(0.5 * (xyz_max - xyz_min), eps) * margin

    u = (coords - center) / half
    u = (u + 1.0) * 0.5
    u = jnp.clip(u, eps, 1.0 - eps)
    return u, center, half

def make_coords_unit(coords, margin=1.01, eps=1e-6):
    reduce_axes = tuple(range(coords.ndim - 1))
    xyz_min = jnp.min(coords, axis=reduce_axes)
    xyz_max = jnp.max(coords, axis=reduce_axes)
    center = 0.5 * (xyz_max + xyz_min)
    half = 0.5 * (xyz_max - xyz_min)
    half = jnp.maximum(half, eps) * margin

    u = (coords - center) / half
    u = (u + 1.0) * 0.5
    u = jnp.clip(u, eps, 1.0 - eps)

    return u, center, half

# brandon notes
# for first frame, compute the movie, compute the measurements at teach t, compute the loss at each timestep, and then propagate back to t=0
# we have a gradient that is a sum of all gradients from each time step

# for ex 1st frame, can make a movie using the coordinate changes, but to get back to first frame, do inverse of operation it takes to get from 2nd to 1st frame and repeat bayesrays?

class BayesRaysUncertaintyMapper():
    def __init__(self, predictor_apply: Callable, predictor_params, raytracing_args: dict, t_frames: jnp.ndarray, 
                 frames_to_include: list[int], A, sigma, fov, grid_res: tuple=(64, 64, 64), lam: float=1e-4/(64**3)):
        self.P = 3 * grid_res[0] * grid_res[1] * grid_res[2]
        self.grid_res = grid_res
        self.lam = lam

        self.pred_apply = predictor_apply
        self.pred_params = jax.tree_util.tree_map(jnp.array, predictor_params)
        self.def_grid = DeformationGrid(grid_res)

        self.def_params = self.def_grid.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))['params']
        self.rt_args = raytracing_args
        self.t_frames = t_frames
        
        self.A = A
        self.sigma = sigma
        
        self.coords = jnp.stack(raytracing_args["coords"], axis=-1)

        # NOTE: attempt 1 with coordinates. originally the uncertainty blob was in the corners of the 2D map and the 3d reconstruction was just a 
        # flat blob.
        self.coords_unit = (self.coords/(fov/2.0) + 1.0) * 0.5
        self.coords_unit = jnp.clip(self.coords_unit, 0.0, 1.0 - 1e-7)

        # NOTE: attempt 2. this assumes all coords are within z_width. If they aren't they get clipped to [0,1] range.
        # I suspect this is why the uncertainty looked like a peanut shape using these unit_coords

        #extents = jnp.array([fov/2, fov/2, z_width], dtype=jnp.float32)
        #self.coords_unit = (self.coords / extents + 1.0) * 0.5
        #self.coords_unit = jnp.clip(self.coords_unit, 0.0, 1.0-1e-7)

        # NOTE: attempt 3. here, we fit our 'coord box' around the actual ray cloud, avoiding clipping
        # this does not work and returns a flat / empty uncertainty map
        #xyz_half = jnp.max(jnp.abs(self.coords), axis=(0,1,2,3))
        #margin = 1.02
        #extents = xyz_half * margin
        #self.coords_unit = (self.coords / extents + 1.0) * 0.5
        #self.coords_unit = jnp.clip(self.coords_unit, 0.0, 1.0 - 1e-7)

        #NOTE: attempt 3.5
        #self.coords_unit, center, half = make_coords_unit(self.coords, margin=1.01, eps=1e-6)
        
        # NOTE: attempt 4 with adjusting the coordinates. still doesnt work
        #self.coords_unit, center, half = make_coords_unit_weighted(self.coords, self.rt_args['g'], self.rt_args['dtau'], keep=0.99, margin=1.01, eps=1e-6)
        
        xyz_half = jnp.max(jnp.abs(self.coords), axis=(0,1,2,3))
        self.voxel_world = (2.0 * xyz_half) / (jnp.array(self.grid_res) - 1.0)
        #clipped = ((self.coords_unit<=1e-6)|(self.coords_unit>=1-1e-6)).mean(axis=(0,1,2,3))
        #print("box center:", center, "box half:", half, "clipped frac per axis:", np.array(clipped))
        
        print("Coords (bayesrays):", self.coords.shape)
        print("Coords min and max:", self.coords.max(), self.coords.min())
        print("Coords unit min, max:", self.coords_unit.max(axis=(0,1,2,3)), self.coords_unit.min(axis=(0,1,2,3)))
        print("Coords unit shape:", self.coords_unit.shape, "Coords shape:", self.coords.shape)

        self.fm_per_frame = []
        t_units = t_frames.unit
        t_list = [t_frames[i] for i in frames_to_include]
        A_list  = [jnp.asarray(A[i]) for i in frames_to_include]

        for t_f, A_f in zip(t_list, A_list):
            Omega  = raytracing_args['Omega']
            g      = raytracing_args['g']
            dtau   = raytracing_args['dtau']
            Sigma  = raytracing_args['Sigma']
            t_geos = raytracing_args['t_geos']
            J      = raytracing_args['J']
            t_start= raytracing_args['t_start_obs']
            t_inj  = raytracing_args['t_injection']

            @jax.jit
            def fm(coords_f, *, t_f=t_f, A_f=A_f):
                coords_moved_f = jnp.moveaxis(coords_f, -1, 0)
                img = bhnerf.network.image_plane_prediction(
                    predictor_params, predictor_apply,
                    t_frames    = t_f,
                    coords      = [coords_moved_f[0], coords_moved_f[1], coords_moved_f[2]],
                    Omega       = Omega,
                    J           = J,
                    g           = g,
                    dtau        = dtau,
                    Sigma       = Sigma,
                    t_start_obs = t_start,
                    t_geos      = t_geos,
                    t_injection = t_inj,
                    t_units     = t_units,
                    stop_grad   = True
                )
                imvec = img.reshape(-1)
                return A_f @ imvec
            self.fm_per_frame.append(fm)
        
        self.nvis = int(len(self.fm_per_frame) * sigma.shape[1])
    
    def _render_one(self, def_params, k):
        offsets = self.def_grid.apply({"params": def_params}, self.coords_unit)
        coords_deformed = self.coords + offsets * self.voxel_world
        vis = self.forward_model(coords_deformed)
        return vis[k]
    
    def _fisher_diag_row(self, k, sigma_k):
        def re_fn(def_p): return jnp.real(self._render_one(def_p, k))
        def im_fn(def_p): return jnp.imag(self._render_one(def_p, k))

        g_re = flatten(jax.grad(re_fn)(self.def_params))
        g_im = flatten(jax.grad(im_fn)(self.def_params))

        factor = 1.0 / (sigma_k / jnp.sqrt(2.0))
        gi2 = g_re**2 + g_im**2

        row = factor**2 * (g_re**2 + g_im**2)
        active = (jnp.max(gi2) > 1e-20).astype(jnp.int32)

        return row, active
    
    def compute_hessian_diag(self, sigma: jnp.ndarray, batch_size: int = 256):
        """
        sigma : (N_vis,)  noise per visibility (same epoch)
        """
        H = jnp.zeros((self.P,), dtype=jnp.float32)
        R_eff = 0

        _fisher_batch = jax.jit(jax.vmap(self._fisher_diag_row, in_axes=(0,0)))
        for start in tqdm(range(0, self.nvis, batch_size), desc='iteration'):
            end = min(start + batch_size, self.nvis)
            idx = jnp.arange(start, end)
            rows, active = _fisher_batch(idx, sigma[idx])
            H += jnp.sum(rows, axis=0)
            R_eff += int(jnp.sum(active))
        
        R_eff = max(R_eff, 1)
        return H, R_eff

    def get_covariance(self, H:jnp.array):
        print(f"computing covariance matrix with lambda: {self.lam}")
        return 1/(H/self.nvis + 2 * self.lam)
    
    def covariance_to_sigma(self, V):
        """
        converts convariance diagonal to per voxel standard deviation
        sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2)
        """
        return np.array(jnp.sqrt(jnp.sum(V.reshape(*self.grid_res, 3), axis=-1)))

    def upsample(self, V, resolution=None, squared_weights=False):
        """
        Trilinearly upsample covariance diagonal, V to resolution R
        
        Args:
            V: covariance diagonal
            R: upsample resolution
        Returns:
            covariance grid, shape (3, R, R, R)
        """
        nx, ny, nz = self.grid_res
        grid = V.reshape(nx, ny, nz, 3)
        var = jnp.sum(grid, axis=-1)
        if resolution is not None and (resolution != nx):
            xs = jnp.linspace(0, 1, resolution)
            coords = jnp.stack(jnp.meshgrid(xs, xs, xs, indexing='ij'), axis=-1)
            var = trilinear(coords, grid, variance=squared_weights).sum(axis=-1)
        return jnp.asarray(var) 

    def prep_uncertainty_3d(self, hessian, covariance: jnp.ndarray, fov,
                            resolution=64, mask=False, squared_weights=False, output=False):
        def sigma_volume(V, upres=64):
            """upsample and normalize the uncertainty map"""
            nx, ny, nz = self.grid_res
            grid = V.reshape((nx, ny, nz, 3))

         
            if upres > nx:
                print('upsampling')
                xs = jnp.linspace(0, 1, upres)
                ys = jnp.linspace(0, 1, upres)
                zs = jnp.linspace(0, 1, upres)

                coords = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
                grid = trilinear(coords, grid, variance=squared_weights)
            return np.array(jnp.sqrt(jnp.sum(grid, axis=-1)))

        sigma_vol = sigma_volume(covariance, resolution)
        
        xmax, xmin = hessian.max(), hessian.min()
        min_uncertainty, max_uncertainty = -3, 6

        uncertainty = np.log10(sigma_vol + 1e-12)
        uncertainty = np.clip(uncertainty, min_uncertainty, max_uncertainty)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)

        if mask:
            emission_estimate = bhnerf.network.sample_3d_grid(self.pred_apply, self.pred_params, fov=fov, resolution=resolution)
            eps = emission_estimate.max() * 0.001
            mask = (emission_estimate > eps)
            uncertainty = np.where(mask, uncertainty, 0.0)

        return uncertainty

    def render_uncertainty_3d_ipv(self, uncertainty, fov, view_kw=dict(azimuth=30, elevation=25, distance=2.8), 
        level_norm=(0.0, 0.3, 0.6, 0.9), opacity=(0.00, 0.10, 0.45, 0.85), output=False, cmap="plasma"):
        import ipyvolume as ipv
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        ipv.figure()
        ipv.view(**view_kw)
        v = ipv.volshow(uncertainty, extent=[(-fov/2, fov/2)]*3, memorder="F", controls=True)
        cmap = cm.get_cmap(cmap)
        rgba = np.array([cmap(l) for l in level_norm], dtype="float32")
        rgba[:, 3] = opacity
        v.tf = ipv.TransferFunction(rgba=rgba, level=level_norm, opacity=opacity)
        ipv.show()

        fig, ax = plt.subplots(figsize=(4, .35))
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        cb.set_label("normalized sigma")
        plt.show()
        
        if output:
            pass
            # TODO

    def propagate_positional_variance(self, sigma0_var: jnp.ndarray, t_frames, fov_M: float, R: int, spin: float, M: float = 1.0, 
        rot_axis=(0.0, 0.0, 1.0), chunk_pts: int = 32768, include_Omega_grad: bool = False, eps_reg: float = 1e-6) -> jnp.ndarray:
        """
        Rigid default (matches differentiating velocity_warp_coords w.r.t coords):
            Var_t(x) = 3 * σ0^2(w(x;t))

        If include_Omega_grad=True:
            Var_t(x) = σ0^2(w(x;t)) * ||J_w(x,t)^{-1}||_F^2
        """

        side = jnp.linspace(-fov_M/2, fov_M/2, R, dtype=jnp.float32)
        X, Y, Z = jnp.meshgrid(side, side, side, indexing='ij')
        pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        I3 = jnp.eye(3, dtype=jnp.float32)

        if hasattr(t_frames, "__len__") and len(t_frames) > 0 and hasattr(t_frames[0], "unit"):
            base_u = t_frames[0].unit
            tq = jnp.asarray([float((t - t_frames[0]).to_value(base_u)) for t in t_frames], dtype=jnp.float32)
            GMc3 = consts.GM_c3(consts.sgra_mass).to(base_u).value
            tM_all = tq / float(GMc3)
        else:
            tM_all = jnp.asarray(t_frames, dtype=jnp.float32)

        sgn = jnp.sign(spin + 1e-12)
        Ms = jnp.sqrt(M)
        def omega_of_r(r):
            return sgn * Ms / (r**1.5 + spin * Ms)

        def w_point(x: jnp.ndarray, tM: jnp.ndarray) -> jnp.ndarray:
            coords = x.reshape(3, 1, 1, 1)
            r = jnp.linalg.norm(x) + 1e-12
            Om = omega_of_r(r)
            wc = bhnerf.emission.velocity_warp_coords(
                coords=coords,
                Omega=Om,   
                t_frames=tM,
                t_start_obs=0.0,
                t_geos=0.0,
                t_injection=0.0,
                rot_axis=rot_axis,
                M=M,
                t_units=None,
                use_jax=True,
            )
            return wc.reshape(3,)

        grid_sigma = sigma0_var.astype(jnp.float32)
        def sample_sigma0(y_points: jnp.ndarray) -> jnp.ndarray:
            u = (y_points / (fov_M/2.0) + 1.0) * 0.5
            u = jnp.clip(u, 0.0, 1.0)
            return trilinear(u, grid_sigma[..., None])[..., 0]

        def _batch_map(fn, arr, *extra):
            outs = []
            for i in range(0, arr.shape[0], chunk_pts):
                sl = slice(i, min(i + chunk_pts, arr.shape[0]))
                outs.append(jax.vmap(fn)(arr[sl], *extra))
            return jnp.concatenate(outs, axis=0)

        v_warp = lambda xs, tM: _batch_map(lambda p: w_point(p, tM), xs)

        if include_Omega_grad:
            jac_w = jax.jit(jax.jacfwd(w_point, argnums=0))
            def stretch_at_point(x, tM):
                Jw = jac_w(x, tM)
                V  = jnp.linalg.solve(Jw + eps_reg*I3, I3) 
                return jnp.sum(V**2)
            v_stretch = lambda xs, tM: _batch_map(lambda p: stretch_at_point(p, tM), xs)

        vols = []
        for tM in tM_all:
            y_pts = v_warp(pts, tM)
            s0 = sample_sigma0(y_pts)
            if include_Omega_grad:
                stretch = v_stretch(pts, tM)
                var_t = (s0 * stretch).reshape(R, R, R)
            else:
                var_t = (3.0 * s0).reshape(R, R, R)
            vols.append(var_t)

        return jnp.stack(vols, axis=0)

    def build_emission_movie(self, t_frames, fov_M, R, spin, M=1.0):
        Omega3D = omega_grid_kepler(fov_M=fov_M, R=R, spin=spin, M=M)
        vols = []
        for t in t_frames:
            vol = bhnerf.network.sample_3d_grid(
                self.pred_apply,
                self.pred_params,
                t_frame=t, 
                t_start_obs=t_frames[0], 
                Omega=Omega3D,
                fov=fov_M,
                coords=None,
                resolution=R
            )
            vols.append(np.asarray(vol))
        return np.stack(vols, axis=0)
    
    def delta_theta(self, t_f, f_ref=0):
        GMc3 = bhnerf.constants.GM_c3(bhnerf.constants.sgra_mass).to(t_f.unit).value
        t_ref = self.t_frames[f_ref]
        dt = (t_f - t_ref).to_value(self.t_frames.unit)
        return dt / GMc3 * self.rt_args['Omega']

    def _make_fisher_batch_for_frame(self, f: int):
        """Returns a jitted, vmapped fisher diag function for a single frame f.
        Args:
        - f (int): frame index

        Returns:
        - (JitWrapped): The fisher gradient function for frame f, vmapped over rays.
        """
        
        fm_f = self.fm_per_frame[f]
        voxel_world = self.voxel_world

        #dtheta = self.delta_theta(self.t_frames[f], f_ref=0)
        #Rinv_f = bhnerf.utils.rotation_matrix([0, 0, -1], -dtheta, use_jax=True)

        def _render_one_ray_f(def_params, ray_idx):
            offsets = self.def_grid.apply({"params": def_params}, self.coords_unit)
            #ffsets_T = jnp.moveaxis(offsets_ref, -1, 0)
            #offsets_preT = jnp.sum(Rinv_f * offsets_T, axis=1)
            #offsets_pre = jnp.moveaxis(offsets_preT, 0, -1)

            coords_def = self.coords + offsets * voxel_world
            vis_f = fm_f(coords_def)
            return vis_f[ray_idx]

        def _fisher_diag_ray_f(ray_idx, sigma_k):
            def re_fn(dp): return jnp.real(_render_one_ray_f(dp, ray_idx))
            def im_fn(dp): return jnp.imag(_render_one_ray_f(dp, ray_idx))
            g_re = flatten(jax.grad(re_fn)(self.def_params))
            g_im = flatten(jax.grad(im_fn)(self.def_params))
            w = 1.0 / (sigma_k / jnp.sqrt(2.0))
            row  = (w**2) * (g_re**2 + g_im**2)
            active = (jnp.max(g_re**2 + g_im**2) > 1e-20).astype(jnp.int32)
            return row, active

        return jax.jit(jax.vmap(_fisher_diag_ray_f, in_axes=(0, 0)))

    def compute_hessian_diag_all(self, sigma: jnp.ndarray, batch_size: int = 256):
        """
        Compute the diagonal of the Hessian matrix (MAP objective) across all frames

        Args:
        - Sigma (jnp.ndarray): shape (N_frames, Nvis), noise per visibility for each frame
        - batch_size (int): the number of rays to vmap over at once

        Returns:
        - H (jnp.ndarray): Shape (P,) the diagonal of the hessian, where P is the number of parameters the deformation grid
        - R_eff (int): the number of active rays (have non-zero gradient which intersect w/the bh volume)
        """
        nvis_f = int(self.nvis/len(self.fm_per_frame))
        print("nvis per frame:", nvis_f, "total:", self.nvis)
        H = jnp.zeros((self.P,), dtype=jnp.float32)
        R_eff = 0

        for f in tqdm(range(len(self.fm_per_frame)), desc='frame'):
            _fisher_batch = self._make_fisher_batch_for_frame(f)
            for start in tqdm(range(0, nvis_f, batch_size), desc='iteration', leave=False):
                end = min(start + batch_size, self.nvis)
                ray_idx = jnp.arange(start, end)

                rows, active = _fisher_batch(ray_idx, sigma[f, ray_idx])
                
                H += jnp.sum(rows, axis=0)
                R_eff += int(jnp.sum(active))
        
        R_eff = max(R_eff, 1)
        return H, R_eff

def prepare_unc_for_render(sigma_vol, mask, min_uncertainty=-3.0, max_uncertainty=6.0):
    """
    Log scales the uncertainty, clips it between min and max uncertainty, and masks it where the network has 0 emission.

    Args:
        sigma_vol: per voxel uncertainty (3, res, res, res)
    Returns:
        the clipped, prepared, and masked uncertainty 
    """
    nt = sigma_vol.shape[0]
    outs = []
    
    for i in range(nt):
        uncertainty = np.log10(sigma_vol[i] + 1e-12)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)
        uncertainty = np.where(mask[i], uncertainty, 0.0)
        outs.append(uncertainty.astype(np.float32))
    return np.stack(outs, axis=0)

def build_dynamic_mask(movie, relative_threshold=0.1):
    per_frame_max = movie.max(axis=(1, 2, 3), keepdims=True)
    return (movie > (per_frame_max*relative_threshold))

def omega_grid_kepler(fov_M: float, R: int, spin: float, M: float = 1.0) -> np.ndarray:
    """Ω(r) on a regular (R,R,R) cube in geometric units."""
    side = np.linspace(-fov_M/2, fov_M/2, R, dtype=np.float32)
    X, Y, Z = np.meshgrid(side, side, side, indexing='ij')
    r = np.sqrt(X*X + Y*Y + Z*Z) + 1e-12
    sgn = np.sign(spin + 1e-12)
    Ms  = np.sqrt(M).astype(np.float32)
    return (sgn * Ms / (r**1.5 + spin * Ms)).astype(np.float32)  