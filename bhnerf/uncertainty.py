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
    --------
    target : (nt, Nvis_t, Npol)        complex / float
    sigma  : (nt, Nvis_t, Npol)        float
    A      : (nt, Nvis_t, image_size**2)  complex
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
    
    
    t_frames   = t_frames#ray_args['t_geos'].shape[0]          # or your t_frames array
    Omega      = ray_args['Omega'][None,...]
    g          = ray_args['g'][None,...]
    dtau       = ray_args['dtau'][None,...]
    Sigma      = ray_args['Sigma'][None,...]
    t_geos     = ray_args['t_geos'][None, ...]
    J          = ray_args['J']
    t_start    = ray_args['t_start_obs']
    t_inj      = ray_args['t_injection']
    t_units    = None 

    print("Omega:", Omega.shape)
    print("g:", g.shape)
    print("dtau: ", dtau.shape)
    print("Sigma: ", Sigma.shape)
    print("t_geos: ", t_geos.shape)

    
    @jax.jit
    def forward_model(coords):
        """
        coords_3d ── same shape as `coords`, but **after** adding the learnt
                     deformation field.  Returns C      (N_vis,) complex64
        """
        # 1. predict image-plane specific intensity I(α,β)
        x, y, z = jnp.moveaxis(coords, -1, 0)
        coords_list = [x, y, z]

        I_img = bhnerf.network.image_plane_prediction(
            predictor_params, predictor_apply,
            t_frames          = t_frames,      # or list/array
            coords            = coords_list,
            Omega             = Omega,
            J                 = J,
            g                 = g,
            dtau              = dtau,
            Sigma             = Sigma,
            t_start_obs       = t_start,
            t_geos            = t_geos,
            t_injection       = t_inj,
            t_units           = t_units,                   # constant
        )[0]                       # keep a single time-slice if nt==1

        

        # 2. flatten to vector and apply the DFT → complex visibilities
        #    (A_dft shape  [N_vis , Ny*Nx])
        vis = A @ I_img.ravel()
        return vis.astype(jnp.complex64)
    return forward_model

def bilinear(coords, theta):
    """gpt generated function to fill in for now"""
    gx, gy, _ = theta.shape
    # scale to vertex index space
    x = coords[..., 0] * (gx - 1)
    y = coords[..., 1] * (gy - 1)

    i0 = jnp.floor(x).astype(jnp.int32)
    j0 = jnp.floor(y).astype(jnp.int32)
    i1 = jnp.clip(i0 + 1, 0, gx - 1)
    j1 = jnp.clip(j0 + 1, 0, gy - 1)

    wx = x - i0
    wy = y - j0

    # gather four corners
    t00 = theta[i0, j0] # lower-left
    t10 = theta[i1, j0] # lower-right
    t01 = theta[i0, j1] # upper-left
    t11 = theta[i1, j1] # upper-right

    return ((1 - wx) * (1 - wy))[..., None] * t00 + \
           (wx * (1 - wy))[..., None] * t10 + \
           ((1 - wx) * wy)[..., None] * t01 + \
           (wx * wy)[..., None] * t11

def trilinear(coords: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Trilinear interpolation into *grid* at **normalised** 3-D *coords*.

    Parameters
    ----------
    coords : [..., 3] float32 in [0,1] cube
    grid   : [gx, gy, gz, C]

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
    def __init__(self, predictor_apply: Callable, predictor_params, forward_model: Callable, raytracing_args: dict, t_frames: jnp.ndarray, grid_res: tuple=(64, 64, 64), lam: float=1e-4/(64**3)):
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
        self.coords = jnp.stack(raytracing_args["coords"], axis=-1)
        

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

    @partial(jax.jit, static_argnums=(0,))
    def _ray_grad(self, idx, sigma) -> tuple[jnp.ndarray, jnp.ndarray]:
        def re_fn(def_params):
            return jnp.real(self.forward_with_deform(def_params)[idx])
        
        def im_fn(def_params):
            return jnp.imag(self.forward_with_deform(def_params )[idx])
        
        grad_re = flatten(jax.grad(re_fn)(self.def_params))
        grad_im = flatten(jax.grad(im_fn)(self.def_params))
        return grad_re/(sigma/jnp.sqrt(2)), grad_im/(sigma/jnp.sqrt(2))

    def compute_hessian(self, sigma: jnp.ndarray, batch_size: int = 512):
        H = jnp.zeros((self.P,))
        nvis = sigma.shape[0]
        print("building hessian diagonal")
        print("-"*20)
        print(f"NVIS = {nvis}")
        print(f"{'uniform noise' if len(jnp.unique(sigma))==1 else 'heteroscedastic noise'}, with median: {jnp.median(sigma)}")

        batch_grad = jax.vmap(lambda idx, sig: self._ray_grad(idx, sig), in_axes=(0,0))

        for start in tqdm(range(0, nvis, batch_size), desc='iteration'):
            end = min(nvis, start+batch_size)
            idx = jnp.arange(start, end)
            sigma_batch = sigma[idx]

            g_re, g_im = batch_grad(idx, sigma_batch)
            print(g_re, g_im)
            
            H += jnp.sum(g_re**2 + g_im**2, axis=0)
        
        print(f"Hessian computed; median value {jnp.median(H)}")
        return partial(self._get_covariance, nvis), H
    
    def _get_covariance(self, H:jnp.array, nvis: int):
        print(f"computing covariance matrix with $\lambda$: {self.lam}")
        return 1/(H/nvis + 2 * self.lam)
        


def compute_hessian_and_uncertainty(chunk_size: int, P, method: str):
    '''
    Args:
        chunk_size (int): how many rows of the hessian to compute at a time. This is deterministic
        not approximated
        lam (float): laplace regularizer
        method (str): 
            'approx': bayes rays' implementation, never build full hessian
            'direct': build full hessian, invert, keep only diagonal of covariance matrix
            '': build full hessian, invert, keep full covariance matrix
    Returns:
        (jnp.ndarray shape (P, P), jnp.ndarray shape(nvis, nvis))
    '''
    
    if method == "approx":
        H = fisher_sum_diag(chunk_size, P, nvis)
        H = H/nvis
        lam = 1e-4/(grid_res[0]**2)
        #lam = 0.5*0.02*jnp.median(H)
        print("lambda:", lam)
        #print(jnp.linalg.cond(H))
        H = H + 2*lam
        V = 1/H
    else:
        assert n_crop**2 >= P, "grid too fine for chosen crop! Choose a grid resolution such that the uncertainty at each point is constrained by the number of visibilites"
        H = fisher_sum(chunk_size, P, nvis)
        H = H/nvis + 2.0*lam*jnp.eye(P)
        V = jnp.linalg.inv(H)
        if method == "direct":
            V = jnp.diag(V)
    return H, V




    def flat_idx(i, j, channel): # channel=0(x) or 1(y)
        return 2 * (i*gy + j) + channel

    @jax.jit
    def build_W_and_idx(u):
        """
        Build the bilinear weight matrix and neighboring weight indicies for effective batching so we dont OOM
        Note that the batching is deterministic. 
        """
        ux, uy = u
        x = ux * (gx-1)
        y = uy * (gy-1)
        i0, j0 = jnp.floor(x).astype(int), jnp.floor(y).astype(int)
        i1, j1 = jnp.minimum(i0+1, gx-1), jnp.minimum(j0+1, gy-1)
        wx, wy = x - i0, y - j0

        # four vertex weights
        w  = jnp.array([
            (1-wx)*(1-wy),
            wx *(1-wy),
            (1-wx)*wy,
            wx *wy,
        ])
        idx = jnp.array([
            flat_idx(i0,j0,0), flat_idx(i0,j0,1),
            flat_idx(i1,j0,0), flat_idx(i1,j0,1),
            flat_idx(i0,j1,0), flat_idx(i0,j1,1),
            flat_idx(i1,j1,0), flat_idx(i1,j1,1),
        ]) # length-8

        #sigma_sub = V[idx[:,None], idx[None,:]]  # (8,8) sub matrix of neighboring bilinear interpolation weights
        W = jnp.stack([ # 2×8
            jnp.reshape(jnp.repeat(w,2)[0::2], (4,)), # x-weights
            jnp.reshape(jnp.repeat(w,2)[1::2], (4,)), # y-weights
        ], axis=0).repeat(2, axis=1)[:,:8]
        
        return idx, W
        '''C = W @ sigma_sub @ W.T
        return jnp.sqrt(jnp.trace(C))'''

    def make_unc_fn(Sigma):
        """
        Function to handle both a diagonal and full hessian computation.
        Later, we compare the diagonal with the full hessian.
        """
        is_diag = Sigma.ndim==1

        @jax.jit
        def pixel_sigma(u):
            idx, W = build_W_and_idx(u)
            if is_diag:
                sigma_sub = jnp.diag(Sigma[idx])
            else:
                sigma_sub = Sigma[idx[:,None], idx[None,:]]
            return jnp.sqrt(jnp.trace(W @ sigma_sub @ W.T))
        return pixel_sigma

    # vectorise over pixels in manageable chunks
    def sigma_map_from_coords(coords, Sigma, chunk=8192):
        """
        compute the full uncertainty map from bayes rays, evaluating at 
        each point in coords
        """
        c_flat = coords.reshape(-1,2)
        unc_fn = make_unc_fn(Sigma)
        out = []
        for k in range(0, c_flat.shape[0], chunk):
            out.append(jax.vmap(unc_fn)(c_flat[k:k+chunk]))
        return jnp.concatenate(out).reshape(coords.shape[:2])

    def vis_to_image(vis_vec, n_crop, full_N=xdim):
        """to visualize the magnitude of noise added (note it was added in fourier space)"""
        uv = np.zeros((full_N, full_N), dtype=np.complex64)
        cx = (full_N - n_crop) // 2
        cy = (full_N - n_crop) // 2
        uv[cy:cy+n_crop, cx:cx+n_crop] = vis_vec.reshape(n_crop, n_crop)
        return np.fft.ifft2(np.fft.ifftshift(uv)).real

    def plot_uncertainty_figs(sigma_map, log_scaled: bool, save_fig: bool, parent_dir: str):
        noise_str: str
        if not noise_type:
            noise_str = 'none'
        elif noise_type == 'variable':
            noise_str = noise_type + "_1e-2-9e-2_radial"
        elif noise_type == 'uniform':
            noise_str = noise_type + str(sigma_vis[0])
        
        fig, ax = plt.subplots(1, 4, figsize=(20,10))
        if noise_type:
            im_3 = ax[3].imshow(vis_obs.real, cmap=cmap, origin='upper')
            ax[3].set_title('Training image')
        else:
            im_3 = ax[3].imshow(vis_to_image(vis_obs, n_crop), cmap=cmap)
            ax[3].set_title('Training image')

        if log_scaled:
            im_0 = ax[0].imshow(jnp.log10(sigma_map + 1e-12), cmap='plasma', origin="upper")
            ax[0].set_title(f"Uncertainty map (log scaled)\nn_crop={n_crop}, grid res={grid_res}\nnoise ={noise_str}, diagonal={'None' if not diagonal else diagonal}")
        else: 
            p98 = jnp.percentile(sigma_map, 98)
            p2 = jnp.percentile(sigma_map, 2)

            im_0 = ax[0].imshow(sigma_map.T, vmin=p2, vmax=p98, cmap='plasma', origin="upper")
            ax[0].set_title(f"Uncertainty map (5th-90th percentile)\npercentile_n_crop={n_crop}, grid res={grid_res}\nnoise ={noise_str}, diagonal={'None' if not diagonal else diagonal}")

        im_1 = ax[1].imshow(jnp.log10(full_image), cmap=cmap, origin='upper')
        ax[1].set_title('Image inpaint prediction (log scaled)')
        im_2 = ax[2].imshow(full_image, cmap=cmap, origin='upper')
        ax[2].set_title('Image inpaint prediction')


        for a, im in zip(ax, (im_0, im_1, im_2, im_3)):
            divider = make_axes_locatable(a)
            cax = divider.append_axes('right', size='3.5%', pad=0.2)
            fig.colorbar(im, cax=cax)

        if save_fig:
            plt.savefig(f'{parent_dir}/log_scaled_clip_n_crop{n_crop}_grid_res{grid_res[0]}x{grid_res[1]}_noise_{noise_str}_diagonal{diagonal}_posenc{posenc_deg}.png')
        plt.tight_layout()
        fig.show()
        return sigma_map

    # bilinear upsample
    grid_coords = np.stack(np.meshgrid(np.linspace(0, 1, gx), np.linspace(0, 1, gy), indexing='xy'), axis=-1)
    sigma_map = sigma_map_from_coords(grid_coords, V).reshape(gx, gy)
    sigma_map = bilinear(coords[..., ::-1], sigma_map[..., None]).squeeze(-1)

    plot_uncertainty_figs(sigma_map, True, True, parent_dir='black_hole_unc_results')
    print()