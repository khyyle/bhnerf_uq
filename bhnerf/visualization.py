import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from bhnerf.utils import normalize
import jax
from jax import numpy as jnp
import functools

def animate_chi2_3d(movie, chi2, figsize=(9,4), legend_loc='lower right', cmap='RdBu_r', fps=10, output=None, writer='ffmpeg'):

    def animate_both(i):
        return animate_frame(i), animate_plot(i)
        
    # Image animation function (called sequentially)
    def animate_frame(i):
        axes[0].set_title(r'Emission estimate: $\theta_o={:1.1f}$'.format(chi2.index[i]))
        im.set_array(movie[i].clip(max=1))
        return im
    
    def animate_plot(i):
        line.set_xdata(chi2.index[i])
        return line,

    fig, axes = plt.subplots(1,2,figsize=figsize)
    num_frames = len(movie)
    
    plot_chi2(chi2_inc, inc_true, False, axes[0])
    line = axes[0].axvline(0, color='blue', linestyle='--', label='hypothesis')
    axes[0].legend(loc=legend_loc)
    axes[0].set_xlim(chi2.index[0], chi2.index[-1])
    axes[1].set_title('Emission estimate')
    axes[1].set_axis_off()
    im =  axes[1].imshow(np.zeros_like(movie[0]))
   
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate_both, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        anim.save(output, writer=writer, fps=fps)
    return anim
    
def plot_stokes_lc(lightcurves, stokes, t_frames=None, axes=None, label=None, color=None, fmt='.', add_mean=False, plot_qu=False):
    
    num_stokes = len(stokes)
    if lightcurves.shape[1] != num_stokes:
        raise AttributeError('lightcurve data doesnt match stokes number: {}'.format(num_stokes))

    t_frames = range(lightcurves.shape[0]) if t_frames is None else t_frames
    
    if not ('Q' in stokes and 'U' in stokes): plot_qu = False
    
    if axes is None:
        num_axes = num_stokes
        if plot_qu: num_axes += 1
        fig, axes = plt.subplots(1, num_axes, figsize=(3*num_axes,3))
    else: 
        num_axes = len(axes)
        if num_axes == num_stokes: plot_qu = False
        
    for i in range(num_stokes):
        axes[i].set_title('{} lightcurve'.format(stokes[i]))
        axes[i].errorbar(t_frames, lightcurves[:, i], color=color, fmt=fmt, label=label)
        
        if add_mean:
            axes[i].axhline(lightcurves[:,i].mean(), linestyle='--', color='r')
            
    if plot_qu:
        axes[-1].set_title('Q-U loop')
        axes[-1].scatter(lightcurves[0:,stokes.index('Q')], lightcurves[0:,stokes.index('U')], s=3, label=label, color=color)
    plt.tight_layout()
    return axes
    
def plot_evpa_ticks(Q, U, alpha, beta, ax=None, scale=None, color=None, pivot='mid', headaxislength=0, headlength=0, width=0.005):
    aolp = (np.arctan2(U, Q) / 2) 
    dolp = np.sqrt(Q**2 + U**2)
    if ax is None: fig, ax = plt.subplots(1, 1)
    ax.quiver(alpha, beta, dolp*np.sin(aolp), -dolp*np.cos(aolp), pivot='mid', 
              headaxislength=0, headlength=0, width=0.005, scale=scale, color=color)
    
def slider_frame_comparison(frames1, frames2, axis=0, scale='amp'):
    """
    Slider comparison of two 3D xr.DataArray along a chosen dimension.
    Parameters
    ----------
    frames1: xr.DataArray
        A 3D array with 'axis' dimension to compare along
    frames2:  xr.DataArray
        A 3D array with 'axis' dimension to compare along
    scale: 'amp' or 'log', default='amp'
        Compare absolute values or log of the fractional deviation.
    """
    from ipywidgets import interact
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plt.tight_layout()
    mean_images = [frames1.mean(axis=axis), frames2.mean(axis=axis),
                   (np.abs(frames1 - frames2)).mean(axis=axis)]
    cbars = []
    titles = [None]*3
    if scale == 'amp':
        titles[2] = 'Absolute difference'
    elif scale == 'log':
        titles[2] = 'Log relative difference'

    for ax, image in zip(axes, mean_images):
        im = ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbars.append(fig.colorbar(im, cax=cax))

    def imshow_frame(frame):
        image1 = np.take(frames1, frame, axis=axis)
        image2 = np.take(frames2, frame, axis=axis)

        if scale == 'amp':
            image3 = np.abs(np.take(frames1, frame, axis=axis) - np.take(frames2, frame, axis=axis))
        elif scale == 'log':
            image3 = np.log(np.abs(np.take(frames1, frame, axis=axis) / np.take(frames2, frame, axis=axis)))

        for ax, img, title, cbar in zip(axes, [image1, image2, image3], titles, cbars):
            ax.imshow(img, origin='lower')
            ax.set_title(title)
            cbar.mappable.set_clim([img.min(), img.max()])

    num_frames = min(frames1.shape[axis], frames2.shape[axis])
    plt.tight_layout()
    interact(imshow_frame, frame=(0, num_frames-1));
    
def plot_geodesic_3D(data_array, geos, method='interact', max_r=10, figsize=(5,5), init_alpha=0, 
                     init_beta=0, vmin=None, vmax=None, cbar_shrink=0.65, fps=10, horizon=True, wire_sphere_r=None):
    
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from mpl_toolkits.mplot3d import Axes3D
    import ipywidgets as widgets
    
    if ('alpha' in geos.dims) and ('beta' in geos.dims):
        def update(ialpha, ibeta, vmin, vmax):
            trajectory = geos.isel(alpha=ialpha, beta=ibeta)
            values = data_array.isel(alpha=ialpha, beta=ibeta)
            trajectory = trajectory.where(trajectory.r < 2*max_r)
            x, y, z = trajectory.x.data, trajectory.y.data, trajectory.z.data
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(values)
            if vmin is None:
                vmin = values.min()
            if vmax is None:
                vmax = values.max()
            lc.set_clim([vmin, vmax])
            ax.set_title('alpha={}, beta={}'.format(values.alpha, values.beta))
            return lc,
    elif ('pix' in geos.dims):
        def update(pix, vmin, vmax):
            trajectory = geos.isel(pix=pix)
            values = data_array.isel(pix=pix)
            trajectory = trajectory.where(trajectory.r < 2*max_r)
            x, y, z = trajectory.x.data, trajectory.y.data, trajectory.z.data
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(values)
            if vmin is None:
                vmin = values.min()
            if vmax is None:
                vmax = values.max()
            lc.set_clim([vmin, vmax])
            ax.set_title('alpha={}, beta={}'.format(values.alpha, values.beta))
            return lc,
    else: 
        raise AttributeError
        
    if method not in ['interact', 'static', 'animate']:
        raise AttributeError('undefined method: {}'.format(method))
    
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.set_xlim([-max_r, max_r])
    ax.set_ylim([-max_r, max_r])
    ax.set_zlim([-max_r, max_r])
    
    # Plot the black hole event horizon (r_plus) 
    if horizon:
        r_plus = float(1 + np.sqrt(1 - geos.spin**2))
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(r_plus*x, r_plus*y, r_plus*z, linewidth=0.0, color='black')

    # Plot the ISCO as a wire-frame
    if wire_sphere_r is not None:
        ax.plot_wireframe(wire_sphere_r*x, wire_sphere_r*y, wire_sphere_r*z, rcount=10, ccount=10, linewidth=0.3)
    
    if ('alpha' in geos.dims) and ('beta' in geos.dims):
        trajectory = geos.isel(alpha=init_alpha, beta=init_beta)
        values = data_array.isel(alpha=init_alpha, beta=init_beta)
    elif ('pix' in geos.dims):
        trajectory = geos.isel(pix=0)
        values = data_array.isel(pix=0)
        
    trajectory = trajectory.where(trajectory.r < 2*max_r)
    x, y, z = trajectory.x.data, trajectory.y.data, trajectory.z.data

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, cmap='viridis')
    lc.set_array(values)
    lc.set_clim([vmin, vmax])
    lc.set_linewidth(2)

    line = ax.add_collection3d(lc)
    cb = fig.colorbar(line, ax=ax, shrink=cbar_shrink, location='left')

    output = fig
    if method == 'interact':
        if ('alpha' in geos.dims) and ('beta' in geos.dims):
            widgets.interact(update, ialpha=(0, geos.alpha.size-1), ibeta=(0, geos.beta.size-1),
                             vmin=widgets.fixed(vmin), vmax=widgets.fixed(vmax))
        elif ('pix' in geos.dims):
            widgets.interact(update, pix=(0, geos.pix.size-1), vmin=widgets.fixed(vmin), vmax=widgets.fixed(vmax))
    if method == 'animate':
        raise NotImplementedError
        # output = animation.FuncAnimation(fig, lambda pix: update(pix, vmin, vmax), 
        #                                 frames=geos.pix.size-1, interval=1e3 / fps)
    return output
       
def animate_synced(movie, measurements, axes, t_dim='t', vmin=None, vmax=None, cmap='RdBu_r', add_ticks=True,
                   add_colorbar=True, title=None, fps=10, output=None, bitrate=1e6):

    def animate_both(i):
        return animate_frame(i), animate_plot(i)
        
    # Image animation function (called sequentially)
    def animate_frame(i):
        im.set_array(movie.isel({t_dim: i}))
        return im
    
    def animate_plot(i):
        line.set_xdata(measurements.isel({t_dim: i}))  # update the data.
        return line,
    
    fig = plt.gcf()
    num_frames, nx, ny = movie.sizes.values()
    image_dims = ['x', 'y']
    extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
              movie[image_dims[1]].min(), movie[image_dims[1]].max()]

    if add_ticks == False:
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    axes[0].set_title(title)

    im =  axes[0].imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap, aspect="equal")

    if add_colorbar:
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
    
    vmin = vmin if vmin else movie.min().data
    vmax = vmax if vmax else movie.max().data
    im.set_clim(vmin, vmax)

    y = np.linspace(movie.y[0], movie.y[-1], measurements.pix.size)
    line, = axes[1].plot(measurements.isel(t=0), y)
    axes[1].set_title(title)
    
    asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
    axes[1].set_aspect(asp)
    
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate_both, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(output, writer=writer)
    return anim

def animate_movies_synced(movie_list, axes, t_dim='t', vmin=None, vmax=None, cmaps='afmhot', add_ticks=False,
                   add_colorbars=True, titles=None, fps=10, output=None, flipy=False, bitrate=1e6):
    """
    Synchronous animation of multiple 3D xr.DataArray along a chosen dimension.

    Parameters
    ----------
    movie_list: list of xr.DataArrays
        A list of movies to animated synchroniously.
    axes: list of matplotlib axis,
        List of matplotlib axis object for the visualization. Should have same length as movie_list.
    t_dim: str, default='t'
        The dimension along which to animate frames
    vmin, vmax : float, optional
        vmin and vmax define the data range that the colormap covers.
        By default, the colormap covers the complete value range of the supplied data.
    cmaps : list of str or matplotlib.colors.Colormap, optional
        If this is a scalar then it is extended for all movies.
        The Colormap instance or registered colormap name used to map scalar data to colors.
        Defaults to :rc:`image.cmap`.
    add_ticks: bool, default=True
        If true then ticks will be visualized.
    add_colorbars: list of bool, default=True
        If this is a scalar then it is extended for all movies. If true then a colorbar will be visualized.
    titles: list of strings, optional
        List of titles for each animation. Should have same length as movie_list.
    fps: float, default=10,
        Frames per seconds.
    output: string,
        Path to save the animated gif. Should end with .gif.
    flipy: bool, default=False,
        Flip y-axis to match ehtim plotting function

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation object.
    """
    # Image animation function (called sequentially)
    def animate_frame(i):
        for movie, im in zip(movie_list, images):
            im.set_array(movie.isel({t_dim: i}))
        return images
    
    def animate_frame(i):
        updated = []
        for movie, im in zip(movie_list, images):
            frame = movie.isel({t_dim: i}).values   # numpy array
            im.set_array(frame)
            im.set_clim(float(frame.min()), float(frame.max()))
            updated.append(im)
        return updated

    fig = plt.gcf()
    num_frames, nx, ny = movie_list[0].sizes.values()

    image_dims = list(movie_list[0].sizes.keys())
    image_dims.remove('t')
    extent = [movie_list[0][image_dims[0]].min(), movie_list[0][image_dims[0]].max(),
              movie_list[0][image_dims[1]].min(), movie_list[0][image_dims[1]].max()]

    # initialization function: plot the background of each frame
    images = []
    titles = [movie.name for movie in movie_list] if titles is None else titles
    cmaps = [cmaps]*len(movie_list) if isinstance(cmaps, str) else cmaps
    vmin_list = [movie.min() for movie in movie_list] if vmin is None else vmin
    vmax_list = [movie.max() for movie in movie_list] if vmax is None else vmax

    for movie, ax, title, cmap, vmin, vmax in zip(movie_list, axes, titles, cmaps, vmin_list, vmax_list):
        if add_ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)
        #im.set_clim(vmin, vmax)
        images.append(im)
        if flipy:
            ax.invert_yaxis()

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(output, writer=writer)
    return anim

@xr.register_dataarray_accessor("visualize")
class _VisualizationAccessor(object):
    """
    Register a custom accessor VisualizationAccessor on xarray.DataArray object.
    This adds methods for visualization of Gaussian Random Fields (3D DataArrays) along a single axis.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def slider(self, t_dim='t', ax=None, cmap=None):
        """
        Interactive slider visualization of a 3D xr.DataArray along specified dimension.

        Parameters
        ----------
        t_dim: str, default='t'
            The dimension along which to animate frames
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        cmap : str or matplotlib.colors.Colormap, optional
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        """
        from ipywidgets import interact
        
        movie = self._obj.squeeze()
        if movie.ndim != 3:
            raise AttributeError('Movie dimensions ({}) different than 3'.format(movie.ndim))

        num_frames = movie[t_dim].size
        image_dims = list(movie.dims)
        image_dims.remove(t_dim)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
                  movie[image_dims[1]].min(), movie[image_dims[1]].max()]

        im = ax.imshow(movie.isel({t_dim: 0}), extent=extent, origin='lower', cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        def imshow_frame(frame):
            img = movie.isel({t_dim: frame})
            im.set_array(movie.isel({t_dim: frame}))
            cbar.mappable.set_clim([img.min(), img.max()])

        interact(imshow_frame, frame=(0, num_frames-1));

    def animate(self, t_dim='t', ax=None, vmin=None, vmax=None, cmap='RdBu_r', add_ticks=True, add_colorbar=True,
                fps=10, output=None, bitrate=1e6):
        """
        Animate a 3D xr.DataArray along a chosen dimension.

        Parameters
        ----------
        t_dim: str, default='t'
            The dimension along which to animate frames
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers.
            By default, the colormap covers the complete value range of the supplied data.
        cmap : str or matplotlib.colors.Colormap, default='RdBu_r'
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        add_ticks: bool, default=True
            If true then ticks will be visualized.
        add_colorbar: bool, default=True
            If true then a colorbar will be visualized
        fps: float, default=10,
            Frames per seconds.
        output: string,
            Path to save the animated gif. Should end with .gif.

        Returns
        -------
        anim: matplotlib.animation.FuncAnimation
            Animation object.
        """
        movie = self._obj.squeeze()
        if movie.ndim != 3:
            raise AttributeError('Movie dimensions ({}) different than 3'.format(movie.ndim))

        num_frames = movie[t_dim].size
        image_dims = list(movie.dims)
        image_dims.remove(t_dim)
        nx, ny = [movie.sizes[dim] for dim in image_dims]

        # Image animation function (called sequentially)
        def animate_frame(i):
            im.set_array(movie.isel({t_dim: i}))
            return [im]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
                  movie[image_dims[1]].min(), movie[image_dims[1]].max()]

        # Initialization function: plot the background of each frame
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.2)
            fig.colorbar(im, cax=cax)
        if add_ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])

        def animate_frame(i):
            frame = movie.isel({t_dim: i})
            im.set_array(frame)
            im.set_clim(frame.min(), frame.max())   #  <-- per-frame colour limits
            return [im]

        vmin = movie.min() if vmin is None else vmin
        vmax = movie.max() if vmax is None else vmax
        im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1e3 / fps)

        if output is not None:
            writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
            anim.save(output, writer=writer)
        return anim

class VolumeVisualizer(object):
    def __init__(self, width, height, samples):
        """
        A Volume visualization class
        
        Parameters
        ----------
        width: int
            camera horizontal resolution.
        height: int
            camera vertical resolution.
        samples: int
            Number of integration points along a ray.
        """
        self.width = width
        self.height = height
        self.samples = samples 
        self._pts = None
        
    def set_view(self, cam_r, domain_r, azimuth, zenith, up=np.array([0., 0., 1.])):
        """
        Set camera view geometry
        
        Parameters
        ----------
        cam_r: float,
            Distance from the origin
        domain_r: float, 
            Maximum radius of the spherical domain
        azimuth: float, 
            Azimuth angle in radians
        zenith: float, 
            Zenith angle in radians
        up: array, default=[0,0,1]
            The up direction determines roll of the camera
        """
        camorigin = cam_r * np.array([np.cos(azimuth)*np.sin(zenith), 
                                       np.sin(azimuth)*np.sin(zenith), 
                                       np.cos(zenith)])
        self._viewmatrix = self.viewmatrix(camorigin, up, camorigin)
        fov = 1.06 * np.arctan(np.sqrt(3) * domain_r / cam_r)
        focal = .5 * self.width / jnp.tan(fov)
        rays_o, rays_d = self.generate_rays(
            self._viewmatrix, self.width, self.height, focal)
        
        near = cam_r - np.sqrt(3) * domain_r
        far  = cam_r + np.sqrt(3) * domain_r
    
        self._pts = self.sample_along_rays(rays_o, rays_d, near, far, self.samples)
        self.x, self.y, self.z = self._pts[...,0], self._pts[...,1], self._pts[...,2]
        self.d = jnp.linalg.norm(jnp.concatenate([jnp.diff(self._pts, axis=2), 
                                                  jnp.zeros_like(self._pts[...,-1:,:])], 
                                                  axis=2), axis=-1)
    
    def render(self, emission, facewidth, jit=False, bh_radius=0.0, linewidth=0.1, bh_albedo=[0,0,0], cmap='hot'):
        """
        Render an image of the 3D emission
        
        Parameters
        ----------
        emission: 3D array 
            3D array with emission values
        jit: bool, default=False,
            Just in time compilation. Set true for rendering multiple frames.
            First rendering will take more time due to compilation.
        bh_radius: float, default=0.0
            Radius at which to draw a black hole (for visualization). 
            If bh_radius=0 then no black hole is drawn.
        facewidth: float, default=10.0 
            width of the enclosing cube face
        linewidth: float, default=0.1
            width of the cube lines
        bh_albedo: list, default=[0,0,0]
            Albedo (rgb) of the black hole. default is completly black.
        cmap: str, default='hot'
            Colormap for visualization
            
        Returns
        -------
        rendering: array,
            Rendered image
        """
        if self._pts is None: 
            raise AttributeError('must set view before rendering')
    
        
        cm = plt.get_cmap(cmap) 
        emission_cm = cm(emission)
        emission_cm = jnp.clip(emission_cm - 0.05, 0.0, 1.0)
        emission_cm = jnp.concatenate([emission_cm[..., :3], emission[..., None] / jnp.amax(emission)], axis=-1)

        if jit:
            emission_cube = draw_cube_jit(emission_cm, self._pts, facewidth, linewidth)
            if bh_radius > 0:
                emission_cube = draw_bh_jit(emission_cube, self._pts, bh_radius, bh_albedo)
        else:
            emission_cube = draw_cube(emission_cm, self._pts, facewidth, linewidth)
            if bh_radius > 0:
                emission_cube = draw_bh(emission_cube, self._pts, bh_radius, bh_albedo)
        rendering = alpha_composite(emission_cube, self.d, self._pts, bh_radius, facewidth / 2. - linewidth)
        return rendering
    
    def viewmatrix(self, lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def generate_rays(self, camtoworlds, width, height, focal):
        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(width, dtype=np.float32),  # X-Axis (columns)
            np.arange(height, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - width * 0.5 + 0.5) / focal,
             -(y - height * 0.5 + 0.5) / focal, -np.ones_like(x)],
            axis=-1)
        directions = ((camera_dirs[..., None, :] *
                       camtoworlds[None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(camtoworlds[None, None, :3, -1],
                                  directions.shape)

        return origins, directions

    def sample_along_rays(self, rays_o, rays_d, near, far, num_samples):
        t_vals = jnp.linspace(near, far, num_samples)
        pts = rays_o[..., None, :] + t_vals[None, None, :, None] * rays_d[..., None, :]
        return pts
    
    @property
    def coords(self):
        coords = None if self._pts is None else jnp.moveaxis(self._pts, -1, 0)
        return coords

def alpha_composite(emission, dists, pts, bh_rad, inside_halfwidth=7.5):
    emission = np.clip(emission, 0., 1.)
    color = emission[..., :-1] * dists[0, ..., None]
    alpha = emission[..., -1:] 
    
    # mask for points inside wireframe
    inside = np.where(np.less(np.amax(np.abs(pts), axis=-1), inside_halfwidth), 
                      np.ones_like(pts[..., 0]),
                      np.zeros_like(pts[..., 0]))

    # masks for points outside black hole
    bh = np.where(np.greater(np.linalg.norm(pts, axis=-1), bh_rad),
                  np.ones_like(pts[..., 0]),
                  np.zeros_like(pts[..., 0]))

    combined_mask = np.logical_and(inside, bh)


    rendering = np.zeros_like(color[:, :, 0, :])
    acc = np.zeros_like(color[:, :, 0, 0])
    outside_acc = np.zeros_like(color[:, :, 0, 0])
    for i in range(alpha.shape[-2]):
        ind = alpha.shape[-2] - i - 1

        # if pixels inside cube and outside black hole, don't alpha composite
        rendering = rendering + combined_mask[..., ind, None] * color[..., ind, :]

        # else, alpha composite      
        outside_alpha = alpha[..., ind, :] * (1. - combined_mask[..., ind, None])
        rendering = rendering * (1. - outside_alpha) + color[..., ind, :] * outside_alpha 

        acc = alpha[..., ind, 0] + (1. - alpha[..., ind, 0]) * acc
        outside_acc = outside_alpha[..., 0] + (1. - outside_alpha[..., 0]) * outside_acc

    rendering += np.array([1., 1., 1.])[None, None, :] * (1. - acc[..., None])
    return rendering

@jax.jit
def draw_cube_jit(emission_cm, pts, facewidth, linewidth):
    linecolor = jnp.array([0.0, 0.0, 0.0, 1e6])
    vertices = jnp.array([[-facewidth/2., -facewidth/2., -facewidth/2.],
                        [facewidth/2., -facewidth/2., -facewidth/2.],
                        [-facewidth/2., facewidth/2., -facewidth/2.],
                        [facewidth/2., facewidth/2., -facewidth/2.],
                        [-facewidth/2., -facewidth/2., facewidth/2.],
                        [facewidth/2., -facewidth/2., facewidth/2.],
                        [-facewidth/2., facewidth/2., facewidth/2.],
                        [facewidth/2., facewidth/2., facewidth/2.]])
    dirs = jnp.array([[-1., 0., 0.],
                      [1., 0., 0.],
                      [0., -1., 0.],
                      [0., 1., 0.],
                      [0., 0., -1.],
                      [0., 0., 1.]])
    
    # [nverts, ndirs, npts, 3]
#     line_seg_pts = vertices[:, None, None, :] + jnp.linspace(0.0, facewidth, 64)[None, None, :, None] * dirs[None, :, None, :]
#     print('test:', line_seg_pts.shape, pts.shape)

    for i in range(vertices.shape[0]):

        for j in range(dirs.shape[0]):
            # Draw line segments from each vertex
            line_seg_pts = vertices[i, None, :] + jnp.linspace(0.0, facewidth, 64)[:, None] * dirs[j, None, :]

            for k in range(line_seg_pts.shape[0]):
                dists = jnp.linalg.norm(pts - jnp.broadcast_to(line_seg_pts[k, None, None, None, :], pts.shape), axis=-1)
                update = linecolor[None, None, None, :] * jnp.exp(-1. * dists / linewidth ** 2)[..., None]
                emission_cm += update

    out = jnp.where(jnp.greater(jnp.broadcast_to(jnp.amax(jnp.abs(pts), axis=-1, keepdims=True), emission_cm.shape), 
                                facewidth/2. + linewidth), jnp.zeros_like(emission_cm), emission_cm)
    return out

def draw_cube(emission_cm, pts, facewidth, linewidth):
    linecolor = jnp.array([0.0, 0.0, 0.0, 1e6])
    vertices = jnp.array([[-facewidth/2., -facewidth/2., -facewidth/2.],
                        [facewidth/2., -facewidth/2., -facewidth/2.],
                        [-facewidth/2., facewidth/2., -facewidth/2.],
                        [facewidth/2., facewidth/2., -facewidth/2.],
                        [-facewidth/2., -facewidth/2., facewidth/2.],
                        [facewidth/2., -facewidth/2., facewidth/2.],
                        [-facewidth/2., facewidth/2., facewidth/2.],
                        [facewidth/2., facewidth/2., facewidth/2.]])
    dirs = jnp.array([[-1., 0., 0.],
                      [1., 0., 0.],
                      [0., -1., 0.],
                      [0., 1., 0.],
                      [0., 0., -1.],
                      [0., 0., 1.]])
    
    # [nverts, ndirs, npts, 3]
#     line_seg_pts = vertices[:, None, None, :] + jnp.linspace(0.0, facewidth, 64)[None, None, :, None] * dirs[None, :, None, :]
#     print('test:', line_seg_pts.shape, pts.shape)

    for i in range(vertices.shape[0]):

        for j in range(dirs.shape[0]):
            # Draw line segments from each vertex
            line_seg_pts = vertices[i, None, :] + jnp.linspace(0.0, facewidth, 64)[:, None] * dirs[j, None, :]

            for k in range(line_seg_pts.shape[0]):
                dists = jnp.linalg.norm(pts - jnp.broadcast_to(line_seg_pts[k, None, None, None, :], pts.shape), axis=-1)
                update = linecolor[None, None, None, :] * jnp.exp(-1. * dists / linewidth ** 2)[..., None]
                emission_cm += update

    out = jnp.where(jnp.greater(jnp.broadcast_to(jnp.amax(jnp.abs(pts), axis=-1, keepdims=True), emission_cm.shape), 
                                facewidth/2. + linewidth), jnp.zeros_like(emission_cm), emission_cm)
    return out

@jax.jit
def draw_bh_jit(emission, pts, bh_radius, bh_albedo):
    bh_albedo = jnp.array(bh_albedo)[None, None, None, :]
    lightdir = jnp.array([-1., -1., 1.])
    lightdir /= jnp.linalg.norm(lightdir, axis=-1, keepdims=True)
    bh_color = jnp.sum(lightdir * pts, axis=-1)[..., None] * bh_albedo
    emission = jnp.where(jnp.less(jnp.linalg.norm(pts, axis=-1, keepdims=True), bh_radius),
                    jnp.concatenate([bh_color, jnp.ones_like(emission[..., 3:])], axis=-1), emission)
    return emission

def draw_bh(emission, pts, bh_radius, bh_albedo):
    bh_albedo = jnp.array(bh_albedo)[None, None, None, :]
    lightdir = jnp.array([-1., -1., 1.])
    lightdir /= jnp.linalg.norm(lightdir, axis=-1, keepdims=True)
    bh_color = jnp.sum(lightdir * pts, axis=-1)[..., None] * bh_albedo
    emission = jnp.where(jnp.less(jnp.linalg.norm(pts, axis=-1, keepdims=True), bh_radius),
                    jnp.concatenate([bh_color, jnp.ones_like(emission[..., 3:])], axis=-1), emission)
    return emission


def ipyvolume_3d(volume, fov, azimuth=0, elevation=-60, distance=2.5, level=[0, 0.2, 0.7], opacity=[0, 0.2, 0.3], controls=False):
    
    import ipyvolume as ipv
    if volume.ndim == 3:
        ipv.figure()
        ipv.view(azimuth, elevation, distance=distance)
        ipv.volshow(volume, extent=[(-fov/2, fov/2)]*3, memorder='F', level=level, opacity=opacity, controls=True)
        ipv.show()
        
    elif volume.ndim == 4:
        from ipywidgets import interact
        import ipywidgets as widgets
        @interact(t=widgets.IntSlider(min=0, max=volume.shape[0]-1, step=1, value=0))
        def plot_vol(t):
            ipv.figure()
            ipv.view(azimuth, elevation, distance=distance)
            ipv.volshow(volume[t], extent=[(-fov/2, fov/2)]*3, memorder='F', level=level, opacity=opacity, controls=controls)
            ipv.show()
            
            
    else:
        raise AttributeError('volume.ndim = {} not supported'.format(volume.ndim))

def show_uncert_volume(vol, fov,
                       cmap_name="plasma",
                       level_norm=[0.0, 0.3, 0.6, 0.9],
                       opacity   =[0.00, 0.10, 0.45, 0.85],
                       view_kw   =dict(azimuth=30, elevation=25, distance=2.8)) -> None:
    import ipyvolume as ipv
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    """
    Args:
        vol: 3D numpy array (any dtype) – *already normalised* to [0,1]
        fov: physical half‑width of the cube in your GM/c² units
        cmap_name: any Matplotlib colormap name
        level_norm: sigma‑knots in [0,1]  (same length as `opacity`)
        opacity: opacity at those knots
        view_kw: forwarded to ipv.view()
    """
    if vol.ndim == 3:
        ipv.figure()
        ipv.view(**view_kw)

        v = ipv.volshow(vol, extent=[(-fov/2, fov/2)]*3, memorder="F")
        #print("Rendering with cmap:", cmap_name)
        cmap   = cm.get_cmap(cmap_name)
        rgba   = [cmap(l) for l in level_norm]
        rgba   = np.array(rgba, dtype='float32')
        rgba[:, 3] = opacity

        tf = ipv.TransferFunction(rgba=rgba,
                                level=level_norm,
                                opacity=opacity)
        v.tf = tf
        ipv.show()
    elif vol.ndim == 4:
        from ipywidgets import interact
        import ipywidgets as widgets
        @interact(t=widgets.IntSlider(min=0, max=vol.shape[0]-1, step=1, value=0))
        def plot_vol(t):
            ipv.figure()
            ipv.view(**view_kw)
            ipv.volshow(vol[t], extent=[(-fov/2, fov/2)]*3, memorder='F', level=level_norm, opacity=opacity, controls=True)
            ipv.show()

    cmap = plt.get_cmap(cmap_name)
    fig_cb, ax_cb = plt.subplots(figsize=(4, .35))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb   = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax_cb, orientation='horizontal')
    cb.set_label("normalized σ")
    plt.show()

def render_volumes_plotly(volumes_norm, fov, percentiles: list[tuple[float, float]],
                          titles=None,
                          colorscale="Plasma",
                          surface_count=10,
                          opacityscale=[[0.00, 0.00],[0.05, 0.05],[0.20, 0.15],[0.40, 0.30],[0.60, 0.50],[0.80, 0.80],[1.00, 1.00]],
                          global_opacity=None, interactive_sync=False):
    """
    Args
    ----

    volumes_norm: list 
        list of volumes to render
    fov: float
        field of view
    titles: list[str]
        list of title to set for each volume in volumes_norm
    surface_count: int
        the number of isosurfaces to draw
    percentiles: tuple(float, float)
        isomin and isomax percentiles
    opacityscale: list[list(float, float)] 
        opacity scale to render on. Can also be set to None
    global_opacity: float
        global multiplier for opacity
    interactive_sync:
        if multiple volumes, syncs camera movements across the two

    """
    import plotly.subplots
    assert len(volumes_norm) >= 1, "Need at least one volume"
    R_x, R_y, R_z = volumes_norm[0].shape
    for v in volumes_norm[1:]:
        assert v.shape == (R_x, R_y, R_z), "All volumes must have same shape"

    xs = np.linspace(-fov/2, fov/2, R_x)
    ys = np.linspace(-fov/2, fov/2, R_y)
    zs = np.linspace(-fov/2, fov/2, R_z)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")

    n = len(volumes_norm)
    if titles is None: titles = [f"Vol {i+1}" for i in range(n)]
    assert len(titles) == n

    ctor = go.FigureWidget if interactive_sync else go.Figure
    fig = ctor(plotly.subplots.make_subplots(
        rows=1, cols=n,
        specs=[[{'type': 'scene'} for _ in range(n)]],
        subplot_titles=titles
    ))

    cam = dict(up=dict(x=0, y=0, z=1),
               center=dict(x=0, y=0, z=0),
               eye=dict(x=1.35, y=1.35, z=1.35))

    for i, vol in enumerate(volumes_norm):
        vals = np.asarray(vol, dtype=float).ravel(order='C')

        lo, hi = np.percentile(vals, percentiles[i])
        vol_kwargs = dict(
            x=Xg.ravel(order='C'),
            y=Yg.ravel(order='C'),
            z=Zg.ravel(order='C'),
            value=vals,
            colorscale=colorscale,
            surface_count=surface_count,
            isomin=float(lo),
            isomax=float(hi),
            caps=dict(x_show=False, y_show=False, z_show=False)
        )
        if global_opacity is not None:
            vol_kwargs["opacity"] = float(global_opacity)
        if opacityscale is not None:
            vol_kwargs["opacityscale"] = opacityscale

        fig.add_trace(go.Volume(**vol_kwargs), row=1, col=i+1)

        scene_key = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{
            scene_key: dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="cube",
                camera=cam
            )
        })
    for i in range(n):
        scene_key = 'scene' if i == 0 else f'scene{i+1}'
        domain_x0, domain_x1 = fig.layout[scene_key].domain.x
        cb_x = domain_x1 + 0.02
        tr = fig.data[i]
        tr.colorbar = dict(x=cb_x, len=0.85, tickformat=".3g", outlinewidth=0)

    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    if interactive_sync:
        # FIXME: doesnt error, but kills kernel when i try to display with IPython.display's display() and gives backend error
        scenes = ['scene'] + [f'scene{i+1}' for i in range(1, n)]
        _sync_flag = {'busy': False}
        def make_cb(src_scene):
            def _cb(scene, camera):
                if _sync_flag['busy']: return
                _sync_flag['busy'] = True
                for s in scenes:
                    if s != src_scene:
                        getattr(fig.layout, s).camera = camera
                _sync_flag['busy'] = False
            return _cb
        for s in scenes:
            getattr(fig.layout, s).on_change(make_cb(s), 'camera')
    return fig

def render_uncert_volume_plotly(std_norm, fov,
                                level_norm=(0.00, 0.20, 0.40, 0.60, 0.80, 1.00),
                                opacity=(0.00, 0.02, 0.05, 0.10, 0.18, 0.28),
                                cmap="Plasma", surface_count=28, isomin=None, isomax=None, global_opacity=None):

    Xsz, Ysz, Zsz = std_norm.shape
    xs = np.linspace(-fov/2, fov/2, Xsz)
    ys = np.linspace(-fov/2, fov/2, Ysz)
    zs = np.linspace(-fov/2, fov/2, Zsz)

    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')

    vals = std_norm.ravel(order='C')
    if isomin is None: isomin = float(vals.min())
    if isomax is None: isomax = float(vals.max())

    vol_kwargs = dict(
        x=Xg.ravel(order='C'),
        y=Yg.ravel(order='C'),
        z=Zg.ravel(order='C'),
        value=vals,
        colorscale=cmap,
        surface_count=surface_count,
        isomin=isomin,
        isomax=isomax,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
    if global_opacity is not None:
        vol_kwargs["opacity"] = float(global_opacity)

    if level_norm is not None and opacity is not None: 
        opacityscale = [[float(l), float(a)] for l, a in zip(level_norm, opacity)]
        vol_kwargs["opacityscale"] = opacityscale

    fig = go.Figure(go.Volume(**vol_kwargs))
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.show()

    print("opacityscale used:", fig.data[0].opacityscale)
    print("isomin/isomax:", fig.data[0].isomin, fig.data[0].isomax)
    return fig

def render_3d_movie(volume, t_frames, visualizer, rmax, cam_r=37.0, bh_radius=2.0, linewidth=0.1, fps=20, azimuth=0, zenith=np.pi/3, cmap='inferno', normalize=True, output=''):
    """
    Args:
        volumes: np.ndarray, shape (Nens, nt, Nz, Ny, Nx)
        t_frames: 1D array of length nt
        visualizer: VolumeVisualizer(width, height, samples)
        cam_r, rmax, bh_radius, linewidth: camera/render params
        fps: frames per second for output animation
    
    Returns: 
        FuncAnimation: 3D time dependent movie
    """
    from tqdm.auto import tqdm
    visualizer.set_view(
        cam_r    = cam_r,
        domain_r = rmax,
        azimuth  = azimuth,
        zenith   = zenith,
    )

    imgs = []
    for i in tqdm(range(len(t_frames)), desc=f"rendering frame"):
        if normalize:
            vol = volume[i] / (volume[i].max() + 1e-12)
        else:
            vol = volume[i]
        img = visualizer.render(
            vol,
            facewidth = 2*rmax,
            jit       = True,
            bh_radius = bh_radius,
            linewidth = linewidth,
            cmap      = cmap,
        ).clip(0,1)
        imgs.append(img)

    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(imgs[0], origin='lower', vmin=0, vmax=1)
    norm = plt.Normalize(vmin=0, vmax=1)
    sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

    def _update(frame_i):
        im.set_data(imgs[frame_i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=len(imgs), interval=1000/fps
    )

    if output:
        anim.save(output, writer='ffmpeg', fps=fps)

    return anim

def render_3d_quiver(sigma_x, sigma_y, sigma_z,
                     x=None, y=None, z=None,
                     origin=(0.0, 0.0, 0.0),
                     spacing=(1.0, 1.0, 1.0),
                     stride=(2, 2, 2),
                     mask=None,
                     mag_quantile=None,
                     max_arrows=5000,
                     normalize=False,
                     scale=1.0,
                     anchor='tail',
                     colorscale='Viridis',
                     opacity=0.9,
                     sizemode='absolute',
                     sizeref=None,
                     showscale=True,
                     title="3D Uncertainty Quiver (cones)",
                     color_quantile=None,
                     colorbar_pos=(1.08, 0.5, 0.9)):  
    
    """
    Render a 3d quiver plot of the uncertainty vectors on an image space grid

    Parameters
    ----------
    sigma_x, sigma_y, sigma_z : np.ndarray
        3D arrays (Nx, Ny, Nz) with per-voxel standard deviations along x,y,z.
    x, y, z : None | 1D arrays | 3D arrays
        If 1D, treated as coordinate axes; if 3D, must match sigma_* shapes and give positions.
        If None, coordinates are built from origin + index * spacing.
    origin: tuple[float, float, float]
        (x,y,z) describing the center of the viewed area
    spacing: tuple[float, float, float]
        physical spacing between grid points if building from indices. If xyz coords passed this is ignored.
    stride: tuple[float, float, float]
        dx, dy, dz spacing for each cone
    mask : bool/float np.ndarray
        Optional same-shaped array. If bool, True keeps; if float, values > 0 keep.
    mag_quantile : float in (0,1)
        Drop vectors above this magnitude quantile (useful to damp a few giant cones).
    max_arrows: int
        maximum number of cones to draw on the plot
    normalize : bool
        If True, draw only direction (unit cones) and use 'scale' to set uniform length.
    anchor: str
        ('tail', 'tip', 'cm') where to place the cone relative to (x,y,z)
    colorscale: str
        colormap to apply to the plot
    opacity: float
        relative opacity value to set for each cone
    sizemode: str
        ('absolute', 'scaled') thickness of the cones.
    sizeref: 
        reference thickness when sizemode='absolute'. Makes cones fatter.
    showscale: bool
        render a colorbar
    title: str
        Title of the plot
    """
    sx = np.asarray(sigma_x)
    sy = np.asarray(sigma_y)
    sz = np.asarray(sigma_z)
    assert sx.shape == sy.shape == sz.shape, "sigma_x, sigma_y, sigma_z must have identical shapes"
    Nx, Ny, Nz = sx.shape

    if x is None or y is None or z is None:
        dx, dy, dz = spacing
        ox, oy, oz = origin
        xi = np.arange(Nx) * dx + ox
        yi = np.arange(Ny) * dy + oy
        zi = np.arange(Nz) * dz + oz
        X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')
    else:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        if x_arr.ndim == y_arr.ndim == z_arr.ndim == 1:
            X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
        else:
            X, Y, Z = np.asarray(x), np.asarray(y), np.asarray(z)
            assert X.shape == sx.shape == Y.shape == Z.shape, "When passing 3D x/y/z, they must match sigma shapes"

    sx_s = sx[::stride[0], ::stride[1], ::stride[2]]
    sy_s = sy[::stride[0], ::stride[1], ::stride[2]]
    sz_s = sz[::stride[0], ::stride[1], ::stride[2]]
    Xs   = X [::stride[0], ::stride[1], ::stride[2]]
    Ys   = Y [::stride[0], ::stride[1], ::stride[2]]
    Zs   = Z [::stride[0], ::stride[1], ::stride[2]]

    Xf, Yf, Zf = Xs.ravel(), Ys.ravel(), Zs.ravel()
    Uf, Vf, Wf = sx_s.ravel(), sy_s.ravel(), sz_s.ravel()

    keep = np.ones_like(Uf, dtype=bool)
    if mask is not None:
        m = np.asarray(mask)[::stride[0], ::stride[1], ::stride[2]].ravel()
        keep &= (m.astype(bool) if m.dtype == bool else (m > 0))

    mag = np.sqrt(Uf**2 + Vf**2 + Wf**2)
    keep &= np.isfinite(mag) & (mag > 0)

    if mag_quantile is not None and 0.0 < mag_quantile < 1.0:
        cutoff = np.quantile(mag[keep], mag_quantile)
        keep &= (mag <= cutoff)

    Xf, Yf, Zf, Uf, Vf, Wf, mag = Xf[keep], Yf[keep], Zf[keep], Uf[keep], Vf[keep], Wf[keep], mag[keep]

    if Xf.size > max_arrows:
        rng = np.random.default_rng(0)
        idx = rng.choice(Xf.size, size=max_arrows, replace=False)
        Xf, Yf, Zf, Uf, Vf, Wf, mag = Xf[idx], Yf[idx], Zf[idx], Uf[idx], Vf[idx], Wf[idx], mag[idx]

    if normalize:
        eps = 1e-12
        Uf, Vf, Wf = Uf / (mag + eps), Vf / (mag + eps), Wf / (mag + eps)
        mag = np.ones_like(mag)
    Uf, Vf, Wf = Uf * scale, Vf * scale, Wf * scale

    final_mag = np.sqrt(Uf**2 + Vf**2 + Wf**2)
    if final_mag.size == 0:
        cmin, cmax = 0.0, 1.0
    else:
        if color_quantile and 0 <= color_quantile[0] < color_quantile[1] <= 100:
            cmin, cmax = np.nanpercentile(final_mag, list(color_quantile))
        else:
            cmin, cmax = float(np.nanmin(final_mag)), float(np.nanmax(final_mag))
        if not np.isfinite(cmin): cmin = 0.0
        if not np.isfinite(cmax) or cmax <= cmin: cmax = cmin + 1e-12

    if sizeref is None:
        xr = (Xf.max() - Xf.min()) or 1.0
        yr = (Yf.max() - Yf.min()) or 1.0
        zr = (Zf.max() - Zf.min()) or 1.0
        sizeref = 0.02 * max(xr, yr, zr)

    cone = go.Cone(
        x=Xf, y=Yf, z=Zf,
        u=Uf, v=Vf, w=Wf,
        anchor=anchor,
        colorscale=colorscale,
        showscale=showscale,
        opacity=opacity,
        sizemode=sizemode,
        sizeref=sizeref,
        cmin=float(cmin),
        cmax=float(cmax),
        cauto=False,
        colorbar=dict(x=1.08,y=0.5,len=0.9,xpad=10)
    )

    fig = go.Figure(data=[cone])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def overlay_obs_and_warp_from_geos(
    fig,
    geos,
    x, y, z,
    pixel=None,
    spin=+0.2,
    warp_axis=np.array([0.0,0.0,1.0]),
    obs_color="#DC143C",
    warp_color="#4169E1",
    obs_label="obs",
    warp_label=None,
):
    def _to_mesh(a, b, c):
        A, B, C = np.asarray(a), np.asarray(b), np.asarray(c)
        if A.ndim == B.ndim == C.ndim == 1:
            return np.meshgrid(A, B, C, indexing='ij')
        return A, B, C

    X, Y, Z = _to_mesh(x, y, z)
    xmin, xmax = float(np.nanmin(X)), float(np.nanmax(X))
    ymin, ymax = float(np.nanmin(Y)), float(np.nanmax(Y))
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2], dtype=float)
    Lmax   = float(max(xmax-xmin, ymax-ymin, zmax-zmin) or 1.0)

    obs_dir, p_obs = obs_dir_from_geos(geos, pixel=pixel)

    def _add_arrow(p0, vhat, length, color, name, legend=False, legendgroup=None, text=None, text_shift=0.035):
        p1 = p0 + length * vhat
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode='lines', line=dict(width=6, color=color), name=name, showlegend=legend,
            legendgroup=legendgroup 
        ))
        fig.add_trace(go.Cone(
            x=[p1[0]], y=[p1[1]], z=[p1[2]],
            u=[vhat[0]], v=[vhat[1]], w=[vhat[2]],
            anchor='tail', sizemode='absolute',
            showscale=False,
            colorscale=[[0, color], [1, color]],
            name=name + " head", 
            showlegend=False,
            legendgroup=legendgroup
        ))
        if text:
            fig.add_trace(go.Scatter3d(
                x=[p1[0] + text_shift*Lmax], y=[p1[1]], z=[p1[2]],
                mode='text', text=[text], showlegend=False
            ))

    obs_len = 0.7 * Lmax
    p0_obs = center - 0.35 * Lmax * obs_dir
    _add_arrow(p0_obs, obs_dir, obs_len, obs_color, name=obs_label or "Observation direction", legend=True,legendgroup="obs")

    a = np.asarray(warp_axis, dtype=float)
    a /= (np.linalg.norm(a) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - a*np.dot(tmp, a)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(a, e1)

    R = 0.35 * Lmax
    theta0 = np.deg2rad(25.0)
    dtheta = np.deg2rad(85.0) * (1.0 if spin >= 0 else -1.0)
    K = 64
    thetas = theta0 + np.linspace(0, dtheta, K)

    arc = (center.reshape(3,1)
           + R*(np.outer(e1, np.cos(thetas)) + np.outer(e2, np.sin(thetas))))
    fig.add_trace(go.Scatter3d(
        x=arc[0], y=arc[1], z=arc[2],
        mode='lines', line=dict(width=6, color=warp_color),
        name=warp_label or "Rotation direction", showlegend=True, legendgroup='warp'
    ))
    tan = -np.sin(thetas[-1])*e1 + np.cos(thetas[-1])*e2
    if spin < 0: tan *= -1.0
    _add_arrow(arc[:, -1], tan/np.linalg.norm(tan), 0.3*R, warp_color,
               name=warp_label or "Rotation direction", legend=False, legendgroup="warp")

    fig.update_layout(
        legend=dict(
            x=0.02, y=0.02, xanchor='left', yanchor='bottom',
            bgcolor='rgba(255,255,255,0.65)', bordercolor='rgba(0,0,0,0.25)', borderwidth=1
        ),
        scene=dict(aspectmode='data')
    )
    return fig

def obs_dir_from_geos(geos, pixel=None):
    if hasattr(geos, "dims") and ("alpha" in geos.dims) and ("beta" in geos.dims):
        na = int(np.asarray(geos.alpha).size)
        nb = int(np.asarray(geos.beta).size)
        ia, ib = (na//2, nb//2) if pixel is None else (int(pixel[0]), int(pixel[1]))
        x = np.asarray(geos.x.isel(alpha=ia, beta=ib))
        y = np.asarray(geos.y.isel(alpha=ia, beta=ib))
        z = np.asarray(geos.z.isel(alpha=ia, beta=ib))
    else:
        Xg, Yg, Zg = map(np.asarray, (geos.x, geos.y, geos.z))
        na, nb = int(np.asarray(geos.alpha).size), int(np.asarray(geos.beta).size)
        shp = Xg.shape
        ax_a = next(i for i,s in enumerate(shp) if s == na)
        ax_b = next(i for i,s in enumerate(shp) if (i != ax_a and s == nb))
        ia, ib = (na//2, nb//2) if pixel is None else (int(pixel[0]), int(pixel[1]))
        def _take(A):
            A = np.take(A, ia, axis=ax_a); A = np.take(A, ib, axis=ax_b); return A
        x, y, z = _take(Xg), _take(Yg), _take(Zg)

    r = np.sqrt(x*x + y*y + z*z)
    i_obs = int(np.nanargmax(r))
    candidates = []
    if i_obs > 0:             candidates.append(i_obs - 1)
    if i_obs < r.size - 1:    candidates.append(i_obs + 1)
    i_next = min(candidates, key=lambda i: r[i])

    v = np.array([x[i_next]-x[i_obs], y[i_next]-y[i_obs], z[i_next]-z[i_obs]], float)
    v /= (np.linalg.norm(v) + 1e-12)
    p_obs = np.array([x[i_obs], y[i_obs], z[i_obs]], float)
    return v, p_obs

def overlay_obs_and_warp_outside(fig, geos, x, y, z, spin=+0.2,
                                 pixel=None, warp_axis=np.array([0,0,1.0]),
                                 obs_color="#DC143C", warp_color="#4169E1",
                                 obs_label="Observation direction", warp_label='warp direction',
                                 margin=0.12,
                                 arrow_len=0.80
                                 ):
    def _mesh(a,b,c):
        A,B,C = map(np.asarray, (a,b,c))
        return np.meshgrid(A,B,C, indexing='ij') if (A.ndim==B.ndim==C.ndim==1) else (A,B,C)
    X,Y,Z = _mesh(x,y,z)
    xmin,xmax = float(np.nanmin(X)), float(np.nanmax(X))
    ymin,ymax = float(np.nanmin(Y)), float(np.nanmax(Y))
    zmin,zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2], float)
    span  = np.array([xmax-xmin, ymax-ymin, zmax-zmin], float)
    Lmax = float(np.max(span) or 1.0)

    d_obs, p_obs = obs_dir_from_geos(geos, pixel=pixel)

    start = center + (0.5 + margin)*Lmax * d_obs
    length = arrow_len * Lmax
    end   = start + length * d_obs

    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode='lines', line=dict(width=8, color=obs_color),
        name=obs_label, showlegend=True
    ))
    fig.add_trace(go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[d_obs[0]], v=[d_obs[1]], w=[d_obs[2]],
        anchor='tail', sizemode='absolute', sizeref=0.06*length,
        showscale=False, colorscale=[[0,obs_color],[1,obs_color]],
        name=obs_label+" head", showlegend=False
    ))

    a = np.asarray(warp_axis, float); a /= (np.linalg.norm(a)+1e-12)
    tmp = np.array([1,0,0], float) if abs(a[0])<0.9 else np.array([0,1,0], float)
    e1 = tmp - a*np.dot(tmp,a) 
    e1 /= (np.linalg.norm(e1)+1e-12)
    e2 = np.cross(a, e1)
    R = (0.45 + margin)*Lmax
    theta0 = np.deg2rad(25.0)
    dtheta = np.deg2rad(85.0) * (1.0 if spin>=0 else -1.0)
    th = theta0 + np.linspace(0, dtheta, 64)
    arc = (center.reshape(3,1)
           + R*(np.outer(e1, np.cos(th)) + np.outer(e2, np.sin(th))))
    fig.add_trace(go.Scatter3d(
        x=arc[0], y=arc[1], z=arc[2],
        mode='lines', line=dict(width=8, color=warp_color),
        name="Warp (rotation)", showlegend=True
    ))
    tan = -np.sin(th[-1])*e1 + np.cos(th[-1])*e2
    if spin<0: tan *= -1.0
    tan /= (np.linalg.norm(tan)+1e-12)
    head_start = arc[:, -1]
    head_len = 0.25*Lmax
    fig.add_trace(go.Cone(
        x=[head_start[0]], y=[head_start[1]], z=[head_start[2]],
        u=[tan[0]], v=[tan[1]], w=[tan[2]],
        anchor='tail', sizemode='absolute', sizeref=0.06*head_len,
        showscale=False, colorscale=[[0,warp_color],[1,warp_color]],
        name=(warp_label or ("warp +" if spin>=0 else "warp -")), showlegend=False
    ))

    fig.update_layout(
        legend=dict(
            x=0.02, y=0.02, xanchor='left', yanchor='bottom',
            bgcolor='rgba(255,255,255,0.65)', bordercolor='rgba(0,0,0,0.25)', borderwidth=1
        ),
        scene=dict(aspectmode='data')
    )
    return fig


def plot_geodesics_plotly(
    geos,
    value=None,
    stride=(1,1),
    s_stride=1,
    max_r=None,
    mode="markers",
    marker_size=2.5,
    line_every=8,
    max_line_traces=256,
    colorscale="Viridis",
    cmin=None, cmax=None,
    colorbar_title=None,
    show_horizon=True,
    wire_sphere_r=None,
    fig=None,
):
    """
    Plot (subsampled) geodesics from an xarray Dataset `geos` using Plotly.
    Colors points (or rays) by `value` (same grid as geos), else by radius.

    Supports geos with dims (alpha,beta,s) or (pix,s).

    Parameters:
    -----------

    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    value: xr.DataArray
        xr.DataArray with same geodesic grid (e.g., doppler g). If None, colors by radius r.
    stride: tuple[int, int]
        subsample in (alpha,beta) to keep plotting fast. Default will draw all rays
    s_stride: int
        subsample along each geodesic
    max_r: int
        clip rays to r < max_r (in M); None = no clip
    mode: str
        "markers" | "markers+lines" | "lines" -- plotting type for the rays
    marker_size: int
        dot size when markers are drawn
    line_every: int
        draw a *few* line rays (every Nth in the subsampled grid)
    max_line_traces: int
        safety cap for number of line traces
    colorscale:
        the colorscale to plot against
    cmin: float
        minimum intensity
    cmax: float
        maximum intensity
    colorbar_title: str
        title of the colorbar
    show_horizon: bool
        render a wire sphere representing the event horizong of the black hole
    wire_sphere_r: float
        render the stable orbit region of the black hole as a grey, transparent sphere
    fig: go.Figure
        if passed in, adds geodesics to an existing figure. 
    """

    if hasattr(geos, "dims") and ("alpha" in geos.dims) and ("beta" in geos.dims):
        a_dim, b_dim, s_dim = "alpha", "beta", next(d for d in geos.dims if d not in ("alpha","beta"))
        A = geos[a_dim].size; B = geos[b_dim].size
        ia = slice(0, None, max(1,int(stride[0])))
        ib = slice(0, None, max(1,int(stride[1])))
        isamp = {a_dim: ia, b_dim: ib, s_dim: slice(0, None, max(1,int(s_stride)))}
        x = np.asarray(geos.x.isel(**isamp)); y = np.asarray(geos.y.isel(**isamp)); z = np.asarray(geos.z.isel(**isamp))
        if value is not None:
            val = np.asarray(value.isel(**isamp))
    elif hasattr(geos, "dims") and ("pix" in geos.dims):
        p_dim, s_dim = "pix", next(d for d in geos.dims if d != "pix")
        ip = slice(0, None, max(1,int(stride[0])))  # use stride[0] for pix
        isamp = {p_dim: ip, s_dim: slice(0, None, max(1,int(s_stride)))}
        x = np.asarray(geos.x.isel(**isamp)); y = np.asarray(geos.y.isel(**isamp)); z = np.asarray(geos.z.isel(**isamp))
        if value is not None:
            val = np.asarray(value.isel(**isamp))
    else:
        raise ValueError("Unsupported geos dimensions; expected (alpha,beta,*) or (pix,*)")

    r = np.sqrt(x*x + y*y + z*z)
    if max_r is not None:
        mask_r = (r <= max_r)
        x = np.where(mask_r, x, np.nan)
        y = np.where(mask_r, y, np.nan)
        z = np.where(mask_r, z, np.nan)
        if value is not None:
            val = np.where(mask_r, val, np.nan)

    if value is None:
        color = r
        if colorbar_title is None: colorbar_title = "r"
    else:
        color = val
        if colorbar_title is None: colorbar_title = getattr(value, "name", "value")

    cf = color[np.isfinite(color)]
    if cmin is None or cmax is None:
        if cf.size:
            c_lo, c_hi = np.nanpercentile(cf, [2, 98])
            if cmin is None: cmin = float(c_lo)
            if cmax is None: cmax = float(max(c_hi, c_lo + 1e-9))
        else:
            cmin, cmax = 0.0, 1.0

    if fig is None:
        fig = go.Figure()

    if "markers" in mode:
        fig.add_trace(go.Scatter3d(
            x=x.ravel(), y=y.ravel(), z=z.ravel(),
            mode="markers",
            marker=dict(
                size=marker_size,
                color=color.ravel(),
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                colorbar=dict(title=colorbar_title, len=0.85)
            ),
            name="Geodesics (points)",
            showlegend=False
        ))

    if "lines" in mode:
        if x.ndim == 3:
            Na, Nb, Ns = x.shape
            line_is = list(range(0, Na, max(1,int(line_every))))
            line_js = list(range(0, Nb, max(1,int(line_every))))
            traces = 0
            for i in line_is:
                for j in line_js:
                    if traces >= max_line_traces: break
                    xi, yi, zi = x[i,j,:], y[i,j,:], z[i,j,:]
                    ci = color[i,j,:]
                    c_ij = float(np.nanmedian(ci[np.isfinite(ci)])) if np.isfinite(ci).any() else cmin
                    fig.add_trace(go.Scatter3d(
                        x=xi, y=yi, z=zi,
                        mode="lines",
                        line=dict(width=3, color="#9aa0a6"), #colorscale=colorscale, cmin=cmin, cmax=cmax),
                        name="Geodesic",
                        showlegend=False,
                    ))
                    traces += 1
                if traces >= max_line_traces: break
        else:
            Np, Ns = x.shape
            for k in range(0, Np, max(1,int(line_every))):
                xi, yi, zi = x[k,:], y[k,:], z[k,:]
                ci = color[k,:]
                c_k = float(np.nanmedian(ci[np.isfinite(ci)])) if np.isfinite(ci).any() else cmin
                fig.add_trace(go.Scatter3d(
                    x=xi, y=yi, z=zi,
                    mode="lines",
                    line=dict(width=3, color=c_k, colorscale=colorscale, cmin=cmin, cmax=cmax),
                    name="Geodesic",
                    showlegend=False,
                ))

    if show_horizon:
        spin = float(np.asarray(geos.spin)) if hasattr(geos, "spin") else 0.0
        r_plus = float(1.0 + np.sqrt(max(0.0, 1.0 - spin**2)))
        u = np.linspace(0, 2*np.pi, 64)
        v = np.linspace(0, np.pi, 32)
        xs = r_plus*np.outer(np.cos(u), np.sin(v))
        ys = r_plus*np.outer(np.sin(u), np.sin(v))
        zs = r_plus*np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            showscale=False, opacity=0.9, name="Horizon",
            colorscale=[[0,"black"],[1,"black"]]
        ))

    if wire_sphere_r is not None:
        R = float(wire_sphere_r)
        u = np.linspace(0, 2*np.pi, 64)
        v = np.linspace(0, np.pi, 16)
        xs = R*np.outer(np.cos(u), np.sin(v))
        ys = R*np.outer(np.sin(u), np.sin(v))
        zs = R*np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            showscale=False, opacity=0.12, name="Wire sphere",
            surfacecolor=np.zeros_like(xs),
            colorscale=[[0,"#888"],[1,"#888"]]
        ))

    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0,r=0,t=40,b=0),
    )
    return fig

def _grid_mesh(x, y, z):
    """Accept 1D axes or 3D meshes; return 3D meshes X,Y,Z."""
    X = np.asarray(x); Y = np.asarray(y); Z = np.asarray(z)
    if X.ndim == Y.ndim == Z.ndim == 1:
        return np.meshgrid(X, Y, Z, indexing='ij')
    return X, Y, Z

def _voxel_step(a3):
    """Estimate typical step size (voxel pitch) from a 3D mesh or 1D axis."""
    a = np.asarray(a3)
    if a.ndim == 3: a = a[:, 0, 0]
    d = np.diff(a)
    d = d[np.isfinite(d) & (np.abs(d) > 0)]
    return float(np.median(np.abs(d))) if d.size else 1.0

def _pick_geodesic(geos, select="center", pixel=None, stride=(4,4)):
    """
    Choose which ray (pixel) to use.
    select:
      - "center": center pixel
      - "min_r": pixel whose geodesic attains the smallest radius (closest approach)
      - ("pix", k): use integer pixel index for geos with dims (pix, s)
      - ("ab", ia, ib): force (alpha,beta) indices
    stride: subsampling when searching over pixels (for speed)
    """
    if hasattr(geos, "dims") and ("alpha" in geos.dims) and ("beta" in geos.dims):
        a_dim, b_dim = "alpha", "beta"
        s_dim = next(d for d in geos.dims if d not in ("alpha","beta"))
        A, B = geos[a_dim].size, geos[b_dim].size

        if isinstance(select, tuple) and select and select[0] == "ab":
            ia, ib = int(select[1]), int(select[2])

        elif select == "center" and pixel is None:
            ia, ib = A//2, B//2

        elif select == "min_r":
            ia = slice(0, None, max(1, int(stride[0])))
            ib = slice(0, None, max(1, int(stride[1])))
            r = np.asarray(geos.r.isel({a_dim: ia, b_dim: ib})) if "r" in geos else None
            else_r = None
            if "r" not in geos:
                x = np.asarray(geos.x.isel({a_dim: ia, b_dim: ib}))
                y = np.asarray(geos.y.isel({a_dim: ia, b_dim: ib}))
                z = np.asarray(geos.z.isel({a_dim: ia, b_dim: ib}))
                r = np.sqrt(x*x + y*y + z*z)
            idx_flat = np.nanargmin(np.nanmin(r, axis=-1)).item()
            ia0 = np.arange(0, A, max(1,int(stride[0])))[idx_flat // r.shape[1]]
            ib0 = np.arange(0, B, max(1,int(stride[1])))[idx_flat %  r.shape[1]]
            ia, ib = int(ia0), int(ib0)

        else:
            ia, ib = (A//2, B//2) if pixel is None else (int(pixel[0]), int(pixel[1]))

        x = np.asarray(geos.x.isel({a_dim: ia, b_dim: ib}))
        y = np.asarray(geos.y.isel({a_dim: ia, b_dim: ib}))
        z = np.asarray(geos.z.isel({a_dim: ia, b_dim: ib}))
        return (("ab", ia, ib), np.stack([x, y, z], axis=1))  # points (S,3)

    elif hasattr(geos, "dims") and ("pix" in geos.dims):
        p_dim = "pix"; s_dim = next(d for d in geos.dims if d != "pix")
        P = geos[p_dim].size

        if isinstance(select, tuple) and select and select[0] == "pix":
            k = int(select[1])

        elif select == "center" and pixel is None:
            k = P//2

        elif select == "min_r":
            ip = slice(0, None, max(1, int(stride[0])))
            r = np.asarray(geos.r.isel({p_dim: ip})) if "r" in geos else None
            else_r = None
            if "r" not in geos:
                x = np.asarray(geos.x.isel({p_dim: ip}))
                y = np.asarray(geos.y.isel({p_dim: ip}))
                z = np.asarray(geos.z.isel({p_dim: ip}))
                r = np.sqrt(x*x + y*y + z*z)
            # r shape: (P', S)
            k_s = int(np.nanargmin(np.nanmin(r, axis=-1)))
            k = np.arange(0, P, max(1,int(stride[0])))[k_s]

        else:
            k = int(pixel if isinstance(pixel, int) else (P//2))

        x = np.asarray(geos.x.isel({p_dim: k}))
        y = np.asarray(geos.y.isel({p_dim: k}))
        z = np.asarray(geos.z.isel({p_dim: k}))
        return (("pix", k), np.stack([x, y, z], axis=1))

    else:
        raise ValueError("Unsupported geos dims; expected (alpha,beta,*) or (pix,*)")

def geodesic_tube_mask(x, y, z, geos, select="center", pixel=None,
                       stride=(4,4), s_stride=1, radius=None, chunk_voxels=200_000):
    """
    Boolean mask of voxels within 'radius' of the chosen *curved* geodesic.

    - x,y,z: 1D axes or 3D meshes defining your volume grid (same as you pass to render_3d_quiver)
    - geos: xr.Dataset with geodesics (dims: (alpha,beta,s) or (pix,s))
    - select: "center" | "min_r" | ("ab", ia, ib) | ("pix", k)
    - s_stride: subsample along the geodesic (e.g. 1 keeps all points, 2 keeps every other)
    - radius: tube radius in the same units as x,y,z. If None, defaults to ~2 voxels.
    - chunk_voxels: compute distances in chunks to cap memory.
    """
    sel, P = _pick_geodesic(geos, select=select, pixel=pixel, stride=stride)
    P = P[::max(1,int(s_stride))]
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.shape[0] < 2:
        raise ValueError("Not enough points on the selected geodesic after subsampling.")

    X, Y, Z = _grid_mesh(x, y, z)
    if radius is None:
        h = np.median([_voxel_step(X), _voxel_step(Y), _voxel_step(Z)])
        radius = 2.0 * float(h)

    R = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    N = R.shape[0]
    min_d2 = np.full(N, np.inf, dtype=float)

    seg_P0 = P[:-1]
    seg_P1 = P[1:]
    V = seg_P1 - seg_P0
    VV = np.sum(V*V, axis=1)

    keep = VV > 0
    seg_P0, V, VV = seg_P0[keep], V[keep], VV[keep]
    S = seg_P0.shape[0]

    bs = int(chunk_voxels)
    for s in range(0, N, bs):
        e = min(N, s+bs)
        Rb = R[s:e]
        # For each segment, compute distance from all Rb points to that segment
        # t* = clamp_{[0,1]} ((Rb - P0)·V / (V·V))
        # closest point C = P0 + t* V
        # d^2 = ||Rb - C||^2
        # We do this segment-by-segment to keep memory ~ B
        d2_b = np.full(e - s, np.inf, dtype=float)
        for k in range(S):
            P0 = seg_P0[k]
            v  = V[k]
            vv = VV[k]
            Rp = Rb - P0
            t  = (Rp @ v) / vv
            t  = np.clip(t, 0.0, 1.0)
            C  = P0 + t[:,None]*v
            d2 = np.sum((Rb - C)**2, axis=1)
            d2_b = np.minimum(d2_b, d2)
        min_d2[s:e] = d2_b

    mask = (min_d2 <= float(radius)**2).reshape(X.shape)
    return (mask, P, sel)
