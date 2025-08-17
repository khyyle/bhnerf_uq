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
from ehtim.image import make_square
from ehtim.imaging import imager_utils as iu
from astropy import units


def build_A_per_frame(obs, t_frames, image_fov_rad, image_size, dtype='vis', pol='I'):
    """
    Returns
        target: (nt, Nvis_t, Npol)
        sigma: (nt, Nvis_t, Npol)
        A: (nt, Nvis_t, image_size**2)
    """
    # 1. split the big Obsdata object into nt shorter chunks
    dt_sec = (t_frames[-1] - t_frames[0]).to('s').value / (len(t_frames) + 1)
    obs_list = obs.split_obs(t_gather=dt_sec)

    # 2. square prior image defines pixel grid
    prior = make_square(obs, image_size, image_fov_rad)

    chisq_fn = getattr(iu, f'chisqdata_{dtype}')  # e.g. chisqdata_vis

    targ, sig, A_all = [], [], []
    for ob in obs_list:
        # one call per (snapshot × Stokes)
        t, s, A = chisq_fn(ob, prior, mask=[], pol=pol)
        targ.append(np.asarray(t))
        sig.append(np.asarray(s))
        A_all.append(np.asarray(A))         # (Nvis_t, Npix)

    return (np.stack(targ), np.stack(sig), np.stack(A_all))

def make_forward_model(predictor_apply, predictor_params, t_frames, ray_args, A):
    t_frames   = t_frames
    Omega      = ray_args['Omega'][None,...]
    g          = ray_args['g'][None,...]
    dtau       = ray_args['dtau'][None,...]
    Sigma      = ray_args['Sigma'][None,...]
    t_geos     = ray_args['t_geos'][None, ...]
    J          = ray_args['J']
    t_start    = ray_args['t_start_obs']
    t_inj      = ray_args['t_injection']
    t_units    = None 
    
    print("\n----------single frame params----------")
    print("Omega:", Omega.shape)
    print("g:", g.shape)
    print("dtau: ", dtau.shape)
    print("Sigma: ", Sigma.shape)
    print("t_geos: ", t_geos.shape)
    print("Coords (forward model):", ray_args['coords'].shape)
    
    @jax.jit
    def forward_model(coords):
        x, y, z = jnp.moveaxis(coords, -1, 0)
        coords_list = [x, y, z]

        I_img = bhnerf.network.image_plane_prediction(
            predictor_params, predictor_apply,
            t_frames          = t_frames,
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
        )[0]
        vis = A @ I_img.ravel()
        return vis.astype(jnp.complex64)
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

def trilinear(coords: jnp.ndarray,
              grid: jnp.ndarray,
              *,
              variance: bool = False) -> jnp.ndarray:
    """
    Trilinear interpolation into *grid* at **normalized** 3-D *coords*.

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

    weights = jnp.stack([w000,w100,w010,w110,w001,w101,w011,w111], axis=-1)  # [..., 8]
    if variance:
        # Var = SUM((weight_i)^2 Var_i)
        weights = weights ** 2
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-12)

    # gather 8 corners
    def gather(ii, jj, kk):  # [..., C]
        return grid[ii, jj, kk]

    c000 = gather(i0,j0,k0)
    c100 = gather(i1,j0,k0)
    c010 = gather(i0,j1,k0)
    c110 = gather(i1,j1,k0)
    c001 = gather(i0,j0,k1)
    c101 = gather(i1,j0,k1)
    c011 = gather(i0,j1,k1)
    c111 = gather(i1,j1,k1)

    corners = jnp.stack([c000,c100,c010,c110,c001,c101,c011,c111], axis=-2)  # [..., 8, C]

    # weighted sum over the 8 corners
    out = jnp.sum(corners * weights[..., None], axis=-2)  # [..., C]
    return out



def flatten(tree):
    return jnp.concatenate([t.ravel() for t in jax.tree_util.tree_leaves(tree)])

class DeformationGrid(nn.Module):
    resolution: tuple[int, ...]

    @nn.compact
    def __call__(self, coords):
        theta = self.param(
            'theta', nn.initializers.zeros, self.resolution + (coords.shape[-1],)
        )
        return trilinear(coords, theta)

class BayesRaysUncertaintyMapper():
    def __init__(self, predictor_apply: Callable, predictor_params, forward_model: Callable, raytracing_args: dict, t_frames: jnp.ndarray, A, fov, grid_res: tuple=(64, 64, 64), lam: float=1e-4/(64**3)):
        self.P = 3 * grid_res[0] * grid_res[1] * grid_res[2]
        self.grid_res = grid_res
        self.lam = lam

        self.pred_apply = predictor_apply
        self.pred_params = jax.tree_util.tree_map(jnp.array, predictor_params)
        self.def_grid = DeformationGrid(grid_res)

        self.def_params = self.def_grid.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))['params']
        self.forward_model = forward_model
        self.rt_args = raytracing_args
        self.t_frames = t_frames
        self.A = jnp.array(A)
        
        self.coords = jnp.stack(raytracing_args["coords"], axis=-1)
        world = jnp.stack(self.rt_args["coords"], axis=-1)  # shape [...,3]
        coords_unit = (world/(fov/2.0) + 1.0) * 0.5
        coords_unit = jnp.clip(coords_unit, 0.0, 1.0 - 1e-7)
        self.coords_unit = coords_unit
        print("Coords (bayesrays):", self.coords.shape)
        print("Coords min and max:", raytracing_args['coords'].max(), raytracing_args['coords'].min())
        print("Coords unit:", self.coords_unit.max(), self.coords_unit.min())
        
   
    def _render_one(self, def_params, k):
        t_frames   = self.t_frames
        Omega      = self.rt_args['Omega'][None,...]
        g          = self.rt_args['g'][None,...]
        dtau       = self.rt_args['dtau'][None,...]
        Sigma      = self.rt_args['Sigma'][None,...]
        t_geos     = self.rt_args['t_geos'][None, ...]
        J          = self.rt_args['J']
        t_start    = self.rt_args['t_start_obs']
        t_inj      = self.rt_args['t_injection']
        t_units    = None 

        offsets = self.def_grid.apply({"params": def_params}, self.coords)
        coords_deformed = self.coords + offsets
        x, y, z = jnp.moveaxis(coords_deformed, -1, 0)
        coords_list_deformed = [x, y, z]


        I_img = bhnerf.network.image_plane_prediction(
            self.pred_params, self.pred_apply,
            t_frames          = t_frames,
            coords            = coords_list_deformed,
            Omega             = Omega,
            J                 = J,
            g                 = g,
            dtau              = dtau,
            Sigma             = Sigma,
            t_start_obs       = t_start,
            t_geos            = t_geos,
            t_injection       = t_inj,
            t_units           = t_units,
        )[0]
        return jnp.dot(self.A[k], I_img.ravel())
    
    def forward_with_deform(self, def_params):
        """
        add offset and get updated visibility matrix
        """
        offsets = self.def_grid.apply({'params': def_params}, self.coords)
        """img_pred = self.pred_apply({'params': self.pred_params},
            self.t_frames,
            self.rt_args['t_injection'].unit if isinstance(self.rt_args['t_injection'], units.Quantity) else None,
            coords + offsets,
            self.rt_args['Omega'],
            self.rt_args['t_start_obs'],
            self.rt_args['t_geos'],
            self.rt_args['t_injection']
        )"""
        return self.forward_model(self.coords + offsets)
    
    def _fisher_diag_row(self, k, sigma_k):
        def re_fn(def_p): return jnp.real(self._render_one(def_p, k))
        def im_fn(def_p): return jnp.imag(self._render_one(def_p, k))

        g_re = flatten(jax.grad(re_fn)(self.def_params))
        g_im = flatten(jax.grad(im_fn)(self.def_params))

        factor = 1.0 / (sigma_k / jnp.sqrt(2.0))
        return factor**2 * (g_re**2 + g_im**2)
    
    def compute_hessian_diag(self, sigma: jnp.ndarray, batch_size: int = 256):
        """
        sigma : (N_vis,)  noise per visibility (same epoch)
        """
        nvis = int(sigma.shape[0])
        H = jnp.zeros((self.P,), dtype=jnp.float32)

        _fisher_batch = jax.jit(jax.vmap(self._fisher_diag_row, in_axes=(0,0)))
        for start in tqdm(range(0, nvis, batch_size), desc='iteration'):
            end = min(start + batch_size, nvis)
            idx = jnp.arange(start, end)
            H = H + jnp.sum(_fisher_batch(idx, sigma[idx]), axis=0)
        return H    
    
    def get_covariance(self, H:jnp.array, nvis: int):
        print(f"computing covariance matrix with lambda: {self.lam}")
        return 1/(H/nvis + 2 * self.lam)
    
    def covariance_to_sigma(self, V):
        """
        converts convariance diagonal to per voxel standard deviation
        sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2)
        """
        return np.array(jnp.sqrt(jnp.sum(V.reshape(*self.grid_res, 3), axis=-1)))

    def upsample(self, V, resolution=None):
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
            var = trilinear(coords, grid).sum(axis=-1)
        return jnp.asarray(var) 

    def render_uncertainty_3d(self, hessian, covariance: jnp.ndarray, fov: float, level_norm=(0.0, 0.3, 0.6, 0.9), opacity=(0.00, 0.10, 0.45, 0.85), 
                              cmap: str="plasma", view_kw = dict(azimuth=30, elevation=25, distance=2.8), resolution=64, output=False):
        import ipyvolume as ipv
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        def sigma_volume(V, upres=64):
            """upsample and normalize the uncertainty map"""
            nx, ny, nz = self.grid_res
            grid = V.reshape((nx, ny, nz, 3))

            # build normalized coords in unit cube
            xs = jnp.linspace(0, 1, upres)
            ys = jnp.linspace(0, 1, upres)
            zs = jnp.linspace(0, 1, upres)

            # upsample
            coords = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
            interp = trilinear(coords, grid, variance=False)
            var = np.array(jnp.sqrt(jnp.sum(interp, axis=-1)))

            return var

        sigma_vol = sigma_volume(covariance, resolution)
        
        xmax, xmin = hessian.max(), hessian.min()
        min_uncertainty, max_uncertainty = -3, 6

        uncertainty = np.log10(sigma_vol + 1e-12)
        uncertainty = np.clip(uncertainty, min_uncertainty, max_uncertainty)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)

        emission_estimate = bhnerf.network.sample_3d_grid(self.pred_apply, self.pred_params, fov=fov, resolution=resolution)
        eps = emission_estimate.max() * 0.01
        mask = (emission_estimate > eps)
        uncertainty = np.where(mask, uncertainty, 0.0)

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
        cb.set_label("normalized σ")
        plt.show()
        
        if output:
            pass
            #TODO: saving  

    def propagate_positional_variance(self, sigma0_var: jnp.ndarray, t_frames, fov_M: float, R: int, spin: float, M: float = 1.0, 
        rot_axis=(0.0, 0.0, 1.0), chunk_pts: int = 32768, include_Omega_grad: bool = False, eps_reg: float = 1e-6) -> jnp.ndarray:
        """
        Rigid default (matches differentiating velocity_warp_coords w.r.t coords):
            Var_t(x) = 3 * σ0^2(w(x;t))

        If include_Omega_grad=True:
            Var_t(x) = σ0^2(w(x;t)) * ||J_w(x,t)^{-1}||_F^2
        """
        import jax, jax.numpy as jnp
        from bhnerf import constants as consts

        # grid points in world coords, flattened to [N,3]
        side = jnp.linspace(-fov_M/2, fov_M/2, R, dtype=jnp.float32)
        X, Y, Z = jnp.meshgrid(side, side, side, indexing='ij')
        pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # [N,3]
        I3 = jnp.eye(3, dtype=jnp.float32)

        # times -> dimensionless t_M (avoid unit bugs)
        if hasattr(t_frames, "__len__") and len(t_frames) > 0 and hasattr(t_frames[0], "unit"):
            # Convert to same unit as first entry, then to GM/c^3
            base_u = t_frames[0].unit
            tq = jnp.asarray([float((t - t_frames[0]).to_value(base_u)) for t in t_frames], dtype=jnp.float32)
            GMc3 = consts.GM_c3(consts.sgra_mass).to(base_u).value
            tM_all = tq / float(GMc3)
        else:
            tM_all = jnp.asarray(t_frames, dtype=jnp.float32)

        # Keplerian Omega(r) in geometric units
        sgn = jnp.sign(spin + 1e-12)
        Ms = jnp.sqrt(M)
        def omega_of_r(r):
            return sgn * Ms / (r**1.5 + spin * Ms)

        # reuse your warp (backward map) per point (inverse of u)
        def w_point(x: jnp.ndarray, tM: jnp.ndarray) -> jnp.ndarray:
            coords = x.reshape(3, 1, 1, 1)   # (3,1,1,1)
            r = jnp.linalg.norm(x) + 1e-12
            Om = omega_of_r(r) # scalar
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
            )  # -> (1,1,1,3)
            return wc.reshape(3,)

        # sample sigma_0^2 at y = w(x;t)
        grid_σ = sigma0_var.astype(jnp.float32)  # (R,R,R)
        def sample_sigma0(y_points: jnp.ndarray) -> jnp.ndarray:
            u = (y_points / (fov_M/2.0) + 1.0) * 0.5  # map to [0,1]^3
            u = jnp.clip(u, 0.0, 1.0)
            return trilinear(u, grid_σ[..., None])[..., 0]

        # chunked helpers ---------------
        def _batch_map(fn, arr, *extra):
            outs = []
            for i in range(0, arr.shape[0], chunk_pts):
                sl = slice(i, min(i + chunk_pts, arr.shape[0]))
                outs.append(jax.vmap(fn)(arr[sl], *extra))
            return jnp.concatenate(outs, axis=0)

        v_warp = lambda xs, tM: _batch_map(lambda p: w_point(p, tM), xs)

        # full stretch with slow light-gradients via J_w inverse
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
            s0 = sample_sigma0(y_pts) # [N]
            if include_Omega_grad:
                stretch = v_stretch(pts, tM) # [N]
                var_t = (s0 * stretch).reshape(R, R, R)
            else:
                # rigid: J is a rotation → ||J||_F^2 = 3
                var_t = (3.0 * s0).reshape(R, R, R)
            vols.append(var_t)

        return jnp.stack(vols, axis=0)  # (nt, R, R, R)        


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
    return (sgn * Ms / (r**1.5 + spin * Ms)).astype(np.float32)  # (R,R,R)