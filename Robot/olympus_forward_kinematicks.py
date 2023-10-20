import torch
from torch import Tensor


class OlympusForwardKinematicks:
    def __init__(self) -> None:
        self._l_thigh = 0.18
        self._l_shank = 0.30
        self._height_paw = 0.05
        self._d_thighs_half = 0.09 / 2

    def get_torso_height_from_squat_angle(self, q: Tensor) -> Tensor:
        h1 = self._l_thigh * torch.cos(q)
        h2 = (
            torch.sqrt(self._l_shank**2 - (self._l_thigh * torch.sin(q) + self._d_thighs_half) ** 2)
            + self._height_paw
        )
        return h1 + h2
