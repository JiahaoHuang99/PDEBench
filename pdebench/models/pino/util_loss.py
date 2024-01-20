import torch


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def pointwise_rel_l2norm_loss(x, y):
    #   x, y [b, n, c]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-2) + eps
    diff = ((x-y)**2).mean(dim=-2)
    diff = diff / y_norm   # [b, c]
    diff = diff.sqrt().mean()
    return diff


def loss_fn_phys(a, u, batchsize, resolution, task='dr'):

    if task=='dr':
        _loss_fn = loss_dr_2d
        loss_fn = torch.nn.MSELoss()
        loss = _loss_fn(loss_fn, u, batchsize, resolution, space_range=2, time_range=5, alpha=0, beta=0, )
    else:
        raise NotImplementedError

    return loss


def loss_dr_2d(loss_fn, uv, batchsize, resolution, space_range=2, time_range=5, alpha=0, beta=0, ):
    device = uv.device
    b, n, t, _ = uv.shape
    assert b == batchsize
    assert n == resolution * resolution
    assert _ == 2

    u = uv[..., :1]
    v = uv[..., 1:]

    dx = space_range / resolution
    dy = space_range / resolution
    dt = time_range / t

    u = u.reshape(batchsize, resolution, resolution, t, 1)
    v = v.reshape(batchsize, resolution, resolution, t, 1)

    # pde term
    k = 5e-3
    Du = 1e-3
    Dv = 5e-3

    Ru = u - u ** 3 - k - v
    Rv = u - v

    Res_u = FDM_DR_2D(u=u, D=Du, dx=dx, dy=dy, dt=dt, ) - Ru
    Res_v = FDM_DR_2D(u=v, D=Dv, dx=dx, dy=dy, dt=dt, ) - Rv

    loss_phys = loss_fn(Res_u, torch.zeros_like(Res_u)) + loss_fn(Res_v, torch.zeros_like(Res_v))

    # initial conditions
    loss_i = torch.zeros(1, device=device)

    # boundary conditions
    loss_b = torch.zeros(1, device=device)

    loss = loss_phys + alpha * loss_b + beta * loss_i

    return loss

def FDM_DR_2D(u, D, dx, dy, dt, ):
    u_t = gradient_t(u, dt=dt)
    gradxx_u = gradient_xx_scalar(u, dx=dx)
    gradyy_u = gradient_yy_scalar(u, dy=dy)

    res = u_t - D * gradxx_u - D * gradyy_u

    return res


def gradient_t(v, dt):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, t, ...)
    :return: 2D array of shape (b, h, w, t, ...)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    assert h == w

    # Get the shape of the input array
    assert len(v.shape) >= 4
    # Compute the derivatives with respect to y (vertical, height)
    grad_t = torch.gradient(v, dim=3, edge_order=1, spacing=dt)[0]

    return grad_t


def gradient_xy_vector(v, dx, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 2)
    :return: 2D array of shape (b, h, w, ..., 2)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 2

    # Separate the components of v along the height (y-direction) and width (x-direction)
    vy, vx = v[..., 0], v[..., 1]

    # Compute the gradient of v_y with respect to the y-direction (height)
    grad_vy_y = torch.gradient(vy, dim=1, edge_order=1, spacing=dy)[0]  # d(vy)/dy

    # Compute the gradient of v_x with respect to the x-direction (width)
    grad_vx_x = torch.gradient(vx, dim=2, edge_order=1, spacing=dx)[0]  # d(vx)/dx

    # Combine the gradients into a single tensor of shape (H, W, 2)
    grad_v = torch.stack([grad_vy_y, grad_vx_x], dim=-1)

    return grad_v


def gradient_xy_scalar(v, dx, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 1)
    :return: 2D array of shape (b, h, w, ..., 1)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 1

    v = v[..., 0]

    # Compute the derivatives with respect to y (vertical, height, 1st dimension, y)
    grad_y = torch.gradient(v, dim=1, edge_order=1, spacing=dy)[0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_x = torch.gradient(v, dim=2, edge_order=1, spacing=dx)[0]

    # Combine the gradients into a single tensor with shape (h, w, ..., 2)
    grad_v = torch.stack((grad_y, grad_x), dim=-1)

    return grad_v


def gradient_xx_scalar(v, dx):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 1)
    :return: 2D array of shape (b, h, w, ..., 1)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 1

    v = v[..., 0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_x = torch.gradient(v, dim=2, edge_order=1, spacing=dx)[0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_grad_x = torch.gradient(grad_x, dim=2, edge_order=1, spacing=dx)[0]

    return grad_grad_x.unsqueeze(-1)


def gradient_yy_scalar(v, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 1)
    :return: 2D array of shape (b, h, w, ..., 1)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 1

    v = v[..., 0]

    # Compute the derivatives with respect to y (vertical, height, 1st dimension, y)
    grad_y = torch.gradient(v, dim=1, edge_order=1, spacing=dy)[0]

    # Compute the derivatives with respect to x (horizontal, width, 2nd dimension, x)
    grad_grad_y = torch.gradient(grad_y, dim=1, edge_order=1, spacing=dy)[0]

    return grad_grad_y.unsqueeze(-1)


def gradient_FDM_vector(v, dx, dy):
    '''
    Compute the gradient of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, ..., 2)
    :return: 2D array of shape (b, h, w, ..., 2)
    Set the boundary to zero.
    '''

    # Get the shape of the input array
    b, h, w = v.shape[0], v.shape[1], v.shape[2]
    _ = v.shape[-1]
    assert h == w
    assert _ == 2

    # Initialize the gradient array
    grad_v = torch.zeros_like(v, device=v.device)

    # Compute the derivatives with respect to y
    grad_v[:, 1:-1, :, 0] = (v[:, 2:, :, 0] - v[:, :-2, :, 0]) / (2 * dy)

    # Compute the derivatives with respect to x
    grad_v[:, :, 1:-1, 1] = (v[:, :, 2:, 1] - v[:, :, :-2, 1]) / (2 * dx)

    return grad_v


def laplacian(v, dx, dy):
    '''
    Compute the Laplacian of a 2D array using finite differences.
    :param v: 2D array of shape (b, h, w, 2)
    :return: 2D array of shape (b, h, w, 2)
    Compute the boundary gradient using the edge_order=1 option.
    '''

    # Compute the gradient of the input tensor v
    grad_v = gradient_xy_vector(v, dx, dy)
    # Math
    # grad_v[..., 0] = d(vy)/dy
    # grad_v[..., 1] = d(vx)/dx

    # Compute the unmixed second derivatives
    grad_vy_yy = torch.gradient(grad_v[..., 0], dim=1, spacing=dy)[0]  # d^2(vy)/dy^2
    grad_vy_xx = torch.gradient(grad_v[..., 1], dim=2, spacing=dx)[0]  # d^2(vx)/dx^2

    # Sum the second derivatives to obtain the Laplacian
    laplacian_v = grad_vy_yy + grad_vy_xx  # d^2(vy)/dy^2 + d^2(vx)/dx^2

    return laplacian_v


