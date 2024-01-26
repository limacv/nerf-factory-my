# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Ref-NeRF (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn.functional as thf


def reflect(viewdirs, normals):
    """Reflect view directions about normals.

    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.

    Args:
        viewdirs: [..., 3] array of view directions.
        normals: [..., 3] array of normal directions (assumed to be unit vectors).

    Returns:
        [..., 3] array of reflection directions.
    """
    return (
        2.0 * torch.sum(normals * viewdirs, dim=-1, keepdims=True) * normals - viewdirs
    )


def l2_normalize(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""

    return x / torch.sqrt(
        torch.fmax(torch.sum(x**2, dim=-1, keepdims=True), torch.full_like(x, eps))
    )


def compute_weighted_mae(weights, normals, normals_gt):
    """Compute weighted mean angular error, assuming normals are unit length."""
    one_eps = 1 - torch.finfo(torch.float32).eps
    return (
        (
            weights
            * torch.arccos(
                torch.clamp(torch.sum(normals * normals_gt, -1), -one_eps, one_eps)
            )
        ).sum()
        / torch.sum(weights)
        * 180.0
        / np.pi
    )


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    # return np.prod(a - np.arange(k)) / np.math.factorial(k)
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

    Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return (
        (-1) ** m
        * 2**l
        * np.math.factorial(l)
        / np.math.factorial(k)
        / np.math.factorial(l - k - m)
        * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l)
    )


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    # return (np.sqrt(
    #     (2.0 * l + 1.0) * np.math.factorial(l - m) /
    #     (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))
    return np.sqrt(
        (2.0 * l + 1.0)
        * np.math.factorial(l - m)
        / (4.0 * np.pi * np.math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    ml_array = np.array(ml_list).T
    return ml_array


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)
    
def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


class LearnableSphericalGaussianEncoding(torch.nn.Module):
    def __init__(self, degree, **kwargs) -> None:
        super().__init__()
        self.new_forward_mode = kwargs.get("new_forward_mode", True)
        self.num_g = degree
        self.fold_degree = kwargs.get("fold_degree", 1)
        self.training_offset_std = kwargs.get("training_offset_std", 0)
        assert self.num_g % self.fold_degree == 0, f"{self.__class__}::fold_degree should be dividable by degree"
        self.num_degree = self.num_g // self.fold_degree
        self.cone_trace = kwargs.get("cone_trace")
        
        if "init_translation" in kwargs:
            init_trans = kwargs["init_translation"]
            assert init_trans.shape == (self.num_g, 3)
        else:
            init_trans = torch.rand(self.num_g, 3) * 10 - 5
        
        if "init_scale" in kwargs:
            init_scale = kwargs["init_scale"]
            assert init_scale.shape == (self.num_g, 1) or init_scale.shape == (self.num_g, 3)
            init_scale = init_scale.expand_as(init_trans)
        else:
            init_scale = torch.ones_like(init_trans) * 5
        init_scale = self.scale2raw(init_scale)
        
        self.rotate = torch.nn.Parameter(torch.rand(self.num_g, 4) * 2 - 1, requires_grad=True)
        self._scale = torch.nn.Parameter(init_scale, requires_grad=True)
        self.translation = torch.nn.Parameter(init_trans, requires_grad=True)
    
    @staticmethod
    def scale2raw(x):
        return np.log(np.exp(1 / 5 / x) - 1)

    @property
    def invscale(self):
        return thf.softplus(self._scale) * 5 + 1e-8
    
    def forward(self, ray_d, ray_o, roughness_offset):
        if self.new_forward_mode:
            return self.new_forward(ray_d, ray_o, roughness_offset)
        else:
            return self.old_forward(ray_d, ray_o, roughness_offset)
    
    @staticmethod
    def qrot(point, q, dim=-1):
        q = thf.normalize(q, dim=-1, eps=1e-8)
        real_parts = torch.zeros(point.shape[:-1] + (1,)).to(point.device)
        point_as_quaternion = torch.cat((real_parts, point), -1)
        out = quaternion_raw_multiply(
            quaternion_raw_multiply(q, point_as_quaternion),
            quaternion_invert(q),
        )
        return out[..., 1:]

    def new_forward(self, ray_d,  ray_o, roughness_offset):
        orishape = ray_o.shape[:-1]
        ray_o = ray_o.reshape(-1, 3)
        ray_d = ray_d.reshape(-1, 3)
        roughness_offset = roughness_offset.reshape(-1, 1)

        translation = self.translation
        if self.training and self.training_offset_std > 0:
            translation = translation + torch.randn_like(translation) * self.training_offset_std
        
        ray_o_g = self.qrot(ray_o[:, None] - translation[None], self.rotate[None], dim=-1)
        ray_d_g = self.qrot(ray_d[:, None], self.rotate[None], dim=-1)

        t = - (ray_o_g * ray_d_g).sum(dim=-1, keepdim=True)  # N, Deg, 1, -t is the actual dot product
        scale_offset = roughness_offset[:, None]
        invscale = self.invscale * scale_offset

        ray_o_g = ray_o_g * invscale
        ray_d_g = ray_d_g * invscale

        ray_o_2 = (ray_o_g * ray_o_g).sum(dim=-1, keepdim=True)
        ray_d_2 = (ray_d_g * ray_d_g).sum(dim=-1, keepdim=True) + 1e-7
        ray_od = (ray_o_g * ray_d_g).sum(dim=-1, keepdim=True)
        dist = -ray_o_2 + ray_od.clamp_max(0) ** 2 / ray_d_2
        val = torch.exp(dist.clamp_max(0))
        val = val.reshape(val.size(0), self.num_degree, self.fold_degree).sum(dim=-1)
        return val.reshape(*orishape, -1)

    def old_forward(self, ray_d, ray_o, roughness_offset):
        orishape = ray_o.shape[:-1]
        ray_o = ray_o.reshape(-1, 3)
        ray_d = ray_d.reshape(-1, 3)
        roughness_offset = roughness_offset.reshape(-1, 1)

        translation = self.translation
        if self.training and self.training_offset_std > 0:
            translation = translation + torch.randn_like(translation) * self.training_offset_std

        ray_o_g = self.qrot(ray_o[: , None] - translation[None], self.rotate[None], dim=-1)
        ray_d_g = self.qrot(ray_d[:, None], self.rotate[None], dim=-1)
        t = - (ray_o_g * ray_d_g).sum(dim=-1, keepdim=True)  # N, Deg, 1, -t is the actual dot product
        d = torch.norm(ray_o_g, dim=-1, keepdim=True)
        if self.cone_trace:
            scale_offset = roughness_offset[:, None] * t.detach().clamp_min(1e-5)
        else:
            scale_offset = roughness_offset[:, None]

        invscale = self.invscale / (1 + self.invscale * scale_offset)

        with torch.no_grad():
            t_1 = torch.nan_to_num(t / d, 0, 0, 0).clamp(0, 1)
            visibility = t_1.detach().type_as(self.invscale)

        ray_o_g = ray_o_g * invscale
        ray_d_g = ray_d_g * invscale

        ray_o_2 = (ray_o_g * ray_o_g).sum(dim=-1, keepdim=True)
        ray_d_2 = (ray_d_g * ray_d_g).sum(dim=-1, keepdim=True) + 1e-7
        ray_od = (ray_o_g * ray_d_g).sum(dim=-1, keepdim=True)
        dist = -ray_o_2 + ray_od ** 2 / ray_d_2
        val = torch.exp(dist.clamp_max(0))
        # val = val * th.prod(invscale[..., :2], dim=-1, keepdim=True)
        val = val * visibility  # * self.amplitude[None]
        val = val.reshape(val.size(0), self.num_degree, self.fold_degree).sum(dim=-1)
        return val.reshape(*orishape, -1)
