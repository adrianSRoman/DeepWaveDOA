import math
import torch
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.spatial as spatial


def laplacian_exp2(R, normalized=True):
    r"""
    Sparse Graph Laplacian based on exponential-decay metric::

        L     = I - D^{-1/2} W D^{-1/2}  OR
        L_{n} = (2 / \mu_{\max}) L - I

    Parameters
    ----------
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points.
    normalized : bool
        Rescale Laplacian spectrum to [-1, 1].

    Returns
    -------
    L : :py:class:`~scipy.sparse.csr_matrix`
        (N_px, N_px) Laplacian operator.
        If `normalized = True`, return L_{n}.
        If `normalized = False`, return L.
    rho : float
        Scale parameter \rho corresponding to the average distance of a point
        on the graph to its nearest neighbor.
    """
    # Form convex hull to extract nearest neighbors. Each row in
    # cvx_hull.simplices is a triangle of connected points.
    cvx_hull = spatial.ConvexHull(R.T)
    cols = np.roll(cvx_hull.simplices, shift=1, axis=-1).reshape(-1)
    rows = cvx_hull.simplices.reshape(-1)

    # Form sparse affinity matrix from extracted pairs
    W = sp.coo_matrix((cols * 0 + 1, (rows, cols)),
                      shape=(cvx_hull.vertices.size, cvx_hull.vertices.size))
    # Symmetrize the matrix to obtain an undirected graph.
    extended_row = np.concatenate([W.row, W.col])
    extended_col = np.concatenate([W.col, W.row])
    W.row, W.col = extended_row, extended_col
    W.data = np.concatenate([W.data, W.data])
    W = W.tocsr().tocoo()  # Delete potential duplicate pairs

    # Weight matrix elements according to the exponential kernel
    distance = linalg.norm(cvx_hull.points[W.row, :] -
                           cvx_hull.points[W.col, :], axis=-1)
    rho = np.mean(distance)
    W.data = np.exp(- (distance / rho) ** 2)
    W = W.tocsr()

    # Form Graph Laplacian
    D = W.sum(axis=0)
    D_hinv = sp.diags((1 / np.sqrt(D)).tolist()[0], 0, format='csr')
    I_sp = sp.identity(W.shape[0], dtype=np.float, format='csr')
    L = I_sp - D_hinv.dot(W.dot(D_hinv))

    if normalized:
        D_max = splinalg.eigsh(L, k=1, return_eigenvectors=False)
        Ln = (2 / D_max[0]) * L - I_sp
        return Ln, rho
    else:
        return L, rho

def estimate_order(XYZ, rho, wl, eps):
    r"""
    Compute order of polynomial filter to approximate asymptotic
    point-spread function on \cS^{2}.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian instrument coordinates.
    rho : float
        Scale parameter \rho corresponding to the average distance of a point
        on the graph to its nearest neighbor.
        Output of :py:func:`~deepwave.tools.math.graph.laplacian_exp`.
    wl : float
        Wavelength of observations [m].
    eps : float
        Ratio in (0, 1).
        Ensures all PSF magnitudes lower than `max(PSF)*eps` past the main
        lobe are clipped at 0.

    Returns
    -------
    K : int
        Order of polynomial filter.
    """
    XYZ = XYZ / wl
    XYZ_centroid = np.mean(XYZ, axis=1, keepdims=True)
    XYZ_radius = np.mean(linalg.norm(XYZ - XYZ_centroid, axis=0))

    theta = np.linspace(0, np.pi, 1000)
    f = 20 * np.log10(np.abs(np.sinc(theta / np.pi)))
    eps_dB = 10 * np.log10(eps)
    theta_max = np.max(theta[f >= eps_dB])

    beam_width = theta_max / (2 * np.pi * XYZ_radius)
    K = np.sqrt(2 - 2 * np.cos(beam_width)) / rho
    K = int(np.ceil(K))
    return K

# TODO: parametrize this within parameters.py
R = np.load("/scratch/data/metu_arni_cartesian_grid/em32_cartesian_grid.npy") 
K = 18  # Order of polynomial filter.
N_antenna = 32
N_px = 242

pretr_params_freq0 = None
pretr_params_freq1 = None
pretr_params_freq2 = None
pretr_params_freq3 = None
pretr_params_freq4 = None
pretr_params_freq5 = None
pretr_params_freq6 = None
pretr_params_freq7 = None
pretr_params_freq8 = None

import scipy
import scipy.sparse as sp
import scipy.linalg as linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as splinalg
from scipy.sparse import coo_matrix
from pygsp.graphs import Graph

# Reference: https://github.com/deepsphere/deepsphere-pytorch/blob/master/deepsphere/utils/laplacian_funcs.py
def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.
    Args:
        csr_mat (csr_matrix): The sparse scipy matrix.
    Returns:
        sparse_tensor :obj:`torch.sparse.DoubleTensor`: The sparse torch matrix.
    """
    coo = coo_matrix(csr_mat)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    idx = torch.LongTensor(indices)
    vals = torch.DoubleTensor(values)
    shape = coo.shape
    sparse_tensor = torch.sparse.DoubleTensor(idx, vals, torch.Size(shape))
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor


def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    """
    def estimate_lmax(laplacian, tol=5e-3):
        """Estimate the largest eigenvalue of an operator.
        """
        lmax = sp.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2 * tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        I = sp.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian


def cvxhull_graph(R: np.ndarray, cheb_normalized: bool = True, compute_differential_operator: bool = True):

    r"""
    Build the convex hull graph of a point set in :math:`\mathbb{R}^3`.
    The graph edges have exponential-decay weighting.
    Definitions of the graph Laplacians:

    .. math::

        L     = I - D^{-1/2} W D^{-1/2},\qquad        L_{n} = (2 / \mu_{\max}) L - I

    Parameters
    ----------
    R : :py:class:`~numpy.ndarray`
        (N,3) Cartesian coordinates of point set with size N. All points must be **distinct**.
    cheb_normalized : bool
        Rescale Laplacian spectrum to [-1, 1].
    compute_differential_operator : bool
        Computes the graph gradient.

    Returns
    -------
    G : :py:class:`~pygsp.graphs.Graph`
        If ``cheb_normalized = True``, ``G.Ln`` is created (Chebyshev Laplacian :math:`L_{n}` above)
        If ``compute_differential_operator = True``, ``G.D`` is created and contains the gradient.
    rho : float
        Scale parameter :math:`\rho` corresponding to the average distance of a point
        on the graph to its nearest neighbors.

    Examples
    --------

    .. plot::

        import numpy as np
        from pycgsp.graph import cvxhull_graph
        from pygsp.plotting import plot_graph
        theta, phi = np.linspace(0,np.pi,6, endpoint=False)[1:], np.linspace(0,2*np.pi,9, endpoint=False)
        theta, phi = np.meshgrid(theta, phi)
        x,y,z = np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)
        R = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        G, _ = cvxhull_graph(R)
        plot_graph(G)

    Warnings
    --------
    In the newest version of PyGSP (> 0.5.1) the convention is changed: ``Graph.D`` is the divergence operator and
    ``Graph.D.transpose()`` the gradient (see routine `Graph.compute_differential_operator <https://pygsp.readthedocs.io/en/latest/reference/graphs.html#pygsp.graphs.Graph.compute_differential_operator>`_). The code should be adapted when this new version is released.

    """

    # Form convex hull to extract nearest neighbors. Each row in
    # cvx_hull.simplices is a triangle of connected points.
    cvx_hull = spatial.ConvexHull(R.T)
    cols = np.roll(cvx_hull.simplices, shift=1, axis=-1).reshape(-1)
    rows = cvx_hull.simplices.reshape(-1)

    # Form sparse affinity matrix from extracted pairs
    W = sp.coo_matrix((cols * 0 + 1, (rows, cols)),
                      shape=(cvx_hull.vertices.size, cvx_hull.vertices.size))
    # Symmetrize the matrix to obtain an undirected graph.
    extended_row = np.concatenate([W.row, W.col])
    extended_col = np.concatenate([W.col, W.row])
    W.row, W.col = extended_row, extended_col
    W.data = np.concatenate([W.data, W.data])
    W = W.tocsr().tocoo()  # Delete potential duplicate pairs

    # Weight matrix elements according to the exponential kernel
    distance = linalg.norm(cvx_hull.points[W.row, :] -
                           cvx_hull.points[W.col, :], axis=-1)
    rho = np.mean(distance)
    W.data = np.exp(- (distance / rho) ** 2)
    W = W.tocsc()

    G = _graph_laplacian(W, R, compute_differential_operator=compute_differential_operator,
                         cheb_normalized=cheb_normalized)
    return G, rho

def _graph_laplacian(W, R, compute_differential_operator=False, cheb_normalized=False):
    '''
    Form Graph Laplacian
    '''
    G = Graph(W, gtype='undirected', lap_type='normalized', coords=R)
    G.compute_laplacian(lap_type='normalized')  # Stored in G.L, sparse matrix, csc ordering
    if compute_differential_operator is True:
        G.compute_differential_operator()  # stored in G.D, also accessible via G.grad() or G.div() (for the adjoint).
    else:
        pass

    if cheb_normalized:
        D_max = splinalg.eigsh(G.L, k=1, return_eigenvectors=False)
        Ln = (2 / D_max[0]) * G.L - sp.identity(W.shape[0], dtype=np.float64, format='csc')
        G.Ln = Ln
    else:
        pass
    return G

def laplacian_exp(R, depth):
    """Get the icosahedron laplacian list for a certain depth.
    Args:
        R : :py:class:`~numpy.ndarray`
        (N,3) Cartesian coordinates of point set with size N. All points must be **distinct**.
    Returns:
        laplacian: `torch.Tensor` laplacian
        rho: float: laplacian order
    """
    laps = []
    for i in range(depth):
        G, rho = cvxhull_graph(R)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1], rho

class ReTanh(torch.nn.Module):
    '''
    Rectified Hyperbolic Tangent
    '''
    def __init__(self, alpha=1.000000):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        beta = self.alpha / torch.tanh(torch.ones(1, dtype=torch.float64, device='cuda'))
        return torch.fmax(torch.zeros(x.shape, dtype=torch.float64, device='cuda'), beta * torch.tanh(x))

def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere. (Npx, Npx)
        inputs (:obj:`torch.Tensor`): The current input data being forwarded. (Nsamps, Npx)
        weight (:obj:`torch.Tensor`): The weights of the current layer. (K,)
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution. (Nsamps, Npx)  
    """
    K = weight.shape[0]
    was_1d = (inputs.ndim == 1)
    if was_1d:
        inputs = inputs.unsqueeze(0)
    N_sample, Npx = inputs.shape[0], inputs.shape[1]
    inputs = inputs.to('cuda')
    x0 = inputs.T # (Nsamps, Npx).T -> (Npx, Nsamps)
    x0 = x0 #.to('cuda')
    inputs = x0.T.unsqueeze(0) #.to('cuda') # (1, Npx, Nsamps)
    laplacian = laplacian.to('cuda')
    x1 = torch.sparse.mm(laplacian, x0)
    # x1 = torch.sparse.mm(laplacian.to('cuda'), x0.to('cuda')).to('cuda') # (Npx, Npx) x (Npx, Nsamps) -> (Npx, Nsamps)
    inputs = torch.cat((inputs, x1.T.unsqueeze(0)), 0) # (1, Nsamps, Npx) + (1, Nsamps, Npx) = (2*Nsamps, Npx)
    for _ in range(1, K - 1):
        x2 = 2 * torch.sparse.mm(laplacian, x1).to('cuda') - x0 # (Npx, Npx) x (Npx, Nsamps) - (Npx, Nsamps) = (Npx, Nsamps)
        inputs = torch.cat((inputs, x2.T.unsqueeze(0)), 0)  # (ki, Nsamps, Npx) + (1, Nsamps, Npx) 
        x0, x1 = x1, x2 # (Npx, Nsamps), (Npx, Nsamps)
    inputs = inputs.permute(1, 2, 0).contiguous()  # (K, Nsamps, Npx)
    inputs = inputs.view([N_sample*Npx, K])  # (Nsamps*Npx, K)
    inputs = inputs.matmul(weight)  # (Nsamps*Npx, K) x (K,) -> (Nsamps*Npx,)
    inputs = inputs.view([N_sample, Npx]) # (Nsamps, Npx)                     
    return inputs

class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, weight=None, bias=False, conv=cheb_conv):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            weight (torch.Tensor): pre-trained or intial state weight matrix (K,)
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the graph convolution.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv
        if weight is None:
            shape = (kernel_size,)
            self.weight = torch.nn.Parameter(torch.DoubleTensor(*shape))
            std = math.sqrt(2 / (self.in_channels * self.kernel_size))
            self.weight.data.normal_(0, std)
        else:
            self.weight = torch.nn.Parameter(weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.DoubleTensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.bias_initialization()

    def bias_initialization(self):
        """Initialize bias.
        """
        if self.bias is not None:
            self.bias.data.fill_(0.00001)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs
    
class SphericalChebConv(torch.nn.Module):
    """Chebyshev Graph Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size, weight=None):
        """Initialization.
        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.DoubleTensor`): laplacian
            kernel_size (int): order of polynomial filter K. Defaults to 3.
            weight (:obj:`torch.sparse.DoubleTensor`): weight convolutional matrix (K,)
        """
        super().__init__()
        self.register_buffer("laplacian", lap)
        self.chebconv = ChebConv(in_channels, out_channels, kernel_size, weight)

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]
        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.chebconv(self.laplacian, x)
        return x

class BackProjLayer(torch.nn.Module):
    """Spherical Convolutional Neural Netork.
    """

    def __init__(self, Nch, Npx, tau=None, D=None):
        """Initialization.
        Args:
            Nch (int): number of channels in mic array
            Npx (int): number of pixels in Robinson projection
        """
        super().__init__()
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
            

    def reset_parameters(self):
        std = 1e-5
        # self.tau.data.normal_(0, std)
        # self.D.data.normal_(0, std)
        fan_in_tau = self.tau.size(0)
        std_tau = math.sqrt(2.0 / fan_in_tau)
        self.tau.data.uniform_(0, std_tau)

        fan_in_D = self.D.size(1)
        std_D = math.sqrt(2.0 / fan_in_D)
        self.D.data.uniform_(0, std_D)

    def forward(self, S):
        """Vectorized Forward Pass.
        Args:
            S (:obj:`torch.Tensor`): input to be forwarded. (N_samples, 4, 4)
        Returns:
            :obj:`torch.Tensor`: output: (N_samples, 242)
        """
        N_sample, N_ch = S.shape[:2]
        N_px = self.tau.shape[0]
        Ds, Vs = torch.linalg.eigh(S)  # (N_sample, N_ch, N_ch), (N_sample, N_ch, N_ch)
        idx = Ds > 0  # To avoid np.sqrt() issues.
        Ds = torch.where(idx, Ds, torch.zeros_like(Ds)) # apply mask to Ds
        Vs = Vs * torch.sqrt(Ds).unsqueeze(1) # element-wise multiplication between Vs and sqrt(Ds)
        y = torch.matmul(self.D.conj().T, Vs)
        y = torch.linalg.norm(y, dim=2) ** 2 # norm operation along the second dimension and square the result
        y -= self.tau
        return y

class DeepWave(torch.nn.Module):
    """DeepWave: real-time recurrent neural network for acoustic imaging.
    """

    def __init__(self, R, kernel_size, Nch, Npx, batch_size=1, depth=1, pretr_params=None):
        """Initialization.
        Args:
            R: Cartesian coordinates of point set (N,3)
            kernel_size (int): polynomial degree.
            Nch (int): number of channels in mic array
            Npx (int): number of pixels in Robinson projection
        """
        super().__init__()
        # TODO: add argument to determine whether we want to apply graph convolutions, conv2d, or none        
        self.laps, self.rho = laplacian_exp(R, depth)
        if pretr_params: # use pre-trained parameters: tau, mu and D
            self.y_backproj = BackProjLayer(Nch, Npx, tau=pretr_params['tau'], D=pretr_params['D'])
            self.sconvl = SphericalChebConv(Nch, Npx, self.laps[0], kernel_size, weight=pretr_params['mu'])
        else:
            self.y_backproj = BackProjLayer(Nch, Npx)
            self.sconvl = SphericalChebConv(Nch, Npx, self.laps[0], kernel_size)
        self.retanh = ReTanh(alpha=1.000000)

    def reset_parameters(self):
        std = 1e-4 
        self.I.data.random_(0, std)
        
    def forward(self, S, I_prev=None):
        """Forward Pass.
        Args:
            S (:obj:`torch.Tensor`): input (Nch, Nch)
        Returns:
            y_proj :obj: `torch.Tensor`: output (N_sample, Npx)
            x_conv* :obj: `torch.Tensor`: output (N_samle, Npx)
        """
        y_proj  = self.y_backproj(S)
        if I_prev is None:
            I_prev = torch.zeros(y_proj.shape[1], dtype=torch.float64)
        # x_conv4 = self.sconvl(I_prev) 
        # x_conv4 = x_conv4.add(y_proj)
        # x_conv4 = self.retanh(x_conv4)
        # x_conv3 = self.sconvl(x_conv4)
        # x_conv3 = x_conv3.add(y_proj)
        # x_conv3 = self.retanh(x_conv3)
        # x_conv2 = self.sconvl(x_conv3)
        # x_conv2 = x_conv2.add(y_proj)
        # x_conv2 = self.retanh(x_conv2)
        # x_conv1 = self.sconvl(x_conv2)
        # x_conv1 = x_conv1.add(y_proj)
        # x_conv1 = self.retanh(x_conv1)
        # x_conv0 = self.sconvl(I_prev)
        # x_conv0 = x_conv0.add(y_proj)
        # x_conv0 = self.retanh(x_conv0)
        out = y_proj

        return out
