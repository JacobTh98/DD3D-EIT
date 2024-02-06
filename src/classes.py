from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class TankProperties32x2:
    T_d: int = 194
    T_r: int = 97
    T_bx: tuple = (-T_d / 2, T_d / 2)
    T_by: tuple = (-T_d / 2, T_d / 2)
    T_bz: tuple = (0, 140)
    E_zr1: int = 50
    E_zr2: int = 100
    n_el: int = 64


@dataclass
class BallAnomaly:
    x: Union[int, float]
    y: Union[int, float]
    z: Union[int, float]
    r: Union[int, float]
    γ: Union[int, float]


@dataclass
class Boundary:
    x_0: Union[int, float] = 0
    y_0: Union[int, float] = 0
    z_0: Union[int, float] = 0
    x_length: Union[int, float] = 32
    y_length: Union[int, float] = 32
    z_length: Union[int, float] = 32
