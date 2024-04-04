from typing import Union, Tuple
from .classes import (
    TankProperties32x2,
    BallAnomaly,
    HitBox,
    PyEIT3DMesh,
    Ender5Stat,
    MeasurementInformation,
)
import numpy as np
import json
import time
from .ender5 import move_to_absolute_x_y_z, read_temperature
from .sciospec import sciospec_measurement
import os
from datetime import datetime
from sciopy import SystemMessageCallback_usb_hs
from sciopy.sciopy_dataclasses import ScioSpecMeasurementSetup


def compute_hitbox(
    tank: TankProperties32x2,
    ball: BallAnomaly,
    safety_tolerance: Union[int, float] = 5.0,
) -> HitBox:
    """
    Compute the hitbox if a ball object is placed inside the 32x2 tank.

    Parameters
    ----------
    tank : TankProperties32x2
        tank properties [mm]
    ball : BallAnomaly
        ball properties [mm]
    safety_tolerance : Union[int, float], optional
        border tolerance [mm], by default 5.0

    Returns
    -------
    HitBox
        x,y,z limits for measurements [mm]
    """
    hitbox = HitBox(
        r_min=0,
        r_max=tank.T_bx[1] - ball.d / 2 - safety_tolerance,
        x_min=tank.T_bx[0] + ball.d / 2 + safety_tolerance,
        x_max=tank.T_bx[1] - ball.d / 2 - safety_tolerance,
        y_min=tank.T_by[0] + ball.d / 2 + safety_tolerance,
        y_max=tank.T_by[1] - ball.d / 2 - safety_tolerance,
        z_min=tank.T_bz[0] + ball.d / 2 + safety_tolerance,
        z_max=tank.T_bz[1] - ball.d / 2 - safety_tolerance,
    )
    return hitbox


def create_meas_coordinates(
    hitbox: HitBox, x_pts: int, y_pts: int, z_pts: int
) -> np.ndarray:
    """
    Create the measurement trajectory/points with respect on the hitbox.

    Parameters
    ----------
    hitbox : HitBox
        x,y,z limits for measurements [mm]
    x_points : int
        number of measurement points on the x-axis
    y_points : int
        number of measurement points on the y-axis
    z_points : int
        number of measurement points on the z-axis

    Returns
    -------
    np.ndarray
        computed absolute measurement coordinates [mm]
    """
    x = np.linspace(hitbox.x_min, hitbox.x_max, x_pts)
    y = np.linspace(hitbox.y_min, hitbox.y_max, y_pts)
    z = np.linspace(hitbox.z_min, hitbox.z_max, z_pts)

    xx, yy, zz = np.meshgrid(x, y, z)

    distances = np.sqrt(xx**2 + yy**2)
    mask = distances <= hitbox.r_max

    x_flat = xx[mask].flatten()
    y_flat = yy[mask].flatten()
    z_flat = zz[mask].flatten()

    coordinates = np.vstack((x_flat, y_flat, z_flat)).T

    print(
        f"HitBox(x_pts,y_pts,z_pts) leads to {coordinates.shape[0]} available points."
    )
    print(f"So {coordinates.shape[0]} points will be measured.")
    return coordinates


def print_coordinates_props(coordinates: np.ndarray) -> None:
    """
    Print properties of coodinates

    Parameters
    ----------
    coordinates : np.ndarray
        computed absolute measurement coordinates [mm]
    """
    print("Properties of the computed coordinates")
    print("--------------------------------------")
    for ax in range(3):
        print(
            f"min:{np.min(coordinates[:,ax]):.2f}\tmax: {np.max(coordinates[:,ax]):.2f}"
        )
    print(f"\nshape {coordinates.shape}")


def create_measurement_directory(
    meas_dir: str = "measurements/",
) -> Tuple[str, str]:
    """
    Creates a measurement directory with the current timestamp as the name.

    Parameters
    ----------
    meas_dir : str, optional
        target directory, by default "measurements/"

    Returns
    -------
    Tuple[str, str]
        created save path, save directory name
    """
    today = datetime.now()
    f_name = today.strftime("%d_%m_%Y_%Hh_%Mm")
    s_path = f"{meas_dir}{f_name}"
    os.mkdir(s_path)
    print(f"Created new measurement directory at: {s_path}")
    os.mkdir(s_path + "/empty_tank")
    s_path += "/data"
    os.mkdir(s_path)
    s_path += "/"
    return s_path, f_name


def save_parameters_to_json_file(
    s_path: str,
    f_name: str,
    documentation: MeasurementInformation,
    ssms: ScioSpecMeasurementSetup,
    tank: TankProperties32x2,
    ball: BallAnomaly,
    hitbox: HitBox,
) -> None:
    """
    Save all measurement properties and dataclasses into a .json file.

    Parameters
    ----------
    s_path : str
        save path
    f_name : str
        name of the save directory
    ssms : ScioSpecMeasurementSetup
        sciospec measurement dataclass
    tank : TankProperties32x2
        tank properties [mm]
    ball : BallAnomaly
        ball properties [mm]
    hitbox : HitBox
        x,y,z limits for measurements [mm]
    """
    today = datetime.now()
    doc_dict = documentation.__dict__
    ssms_dict = ssms.__dict__
    tank_dict = tank.__dict__
    ball_dict = ball.__dict__
    hitbox_dict = hitbox.__dict__

    combined_json = {
        "SaveTime": today.strftime("%d.%m.%Y %H:%M"),
        "DirName": f_name,
        "Info": "All dimensions are given in [mm]",
        "Documentation": doc_dict,
        "ScioSpecMeasurementSetup": ssms_dict,
        "TankProperties32x2": tank_dict,
        "BallAnomaly": ball_dict,
        "HitBox": hitbox_dict,
    }

    combined_json_str = json.dumps(combined_json, indent=4)

    with open(s_path[:-5] + "info.json", "w") as file:
        file.write(combined_json_str)
    print(f"Saved properties to: {s_path}")


def create_mesh(
    tank: TankProperties32x2, h0: float = 0.1, perm_background: float = 1
) -> PyEIT3DMesh:
    """
    Creates an empty 3D-mesh.

    Parameters
    ----------
    tank : TankProperties32x2
        tank properties [mm]
    h0 : float, optional
        points per millimeter, by default 0.1
    perm_background : float, optional
        perm value, by default 1

    Returns
    -------
    PyEIT3DMesh
        3D point cloud dataclass
    """
    x_pts = y_pts = int(tank.T_d * h0)
    z_pts = int(tank.T_bz[1] * h0)
    # h0 ... points per mm
    x = np.linspace(tank.T_bx[0], tank.T_bx[1], x_pts)
    y = np.linspace(tank.T_by[0], tank.T_by[1], y_pts)
    z = np.linspace(tank.T_bz[0], tank.T_bz[1], z_pts)
    xx, yy, zz = np.meshgrid(x, y, z)

    tank_vol = np.sqrt(xx**2 + yy**2)
    mask = tank_vol <= tank.T_r

    x_nodes = xx[mask].flatten()
    y_nodes = yy[mask].flatten()
    z_nodes = zz[mask].flatten()
    perm = np.ones(len(x_nodes)) * perm_background
    return PyEIT3DMesh(x_nodes, y_nodes, z_nodes, perm)


def clear_perm(mesh: PyEIT3DMesh, perm_background: float = 1.0) -> PyEIT3DMesh:
    """
    Clear and reset all perm values to a given value, by default 1.

    Parameters
    ----------
    mesh : PyEIT3DMesh
        3D point cloud dataclass
    perm_background : float, optional
        initial perm value, by default 1.0

    Returns
    -------
    PyEIT3DMesh
        3D point cloud dataclass
    """
    mesh.perm_array = np.ones(len(mesh.perm_array)) * perm_background
    return mesh


def set_perm(
    mesh: PyEIT3DMesh,
    anomaly: BallAnomaly,
    perm_background: float = 1.0,
    clear_bg: bool = True,
) -> PyEIT3DMesh:
    """
    Set the perm values for point cloud representation.

    Parameters
    ----------
    mesh : PyEIT3DMesh
        3D point cloud dataclass
    anomaly : BallAnomaly
        description of the anomaly
    perm_background : float
        perm value, by default 1
    clear_bg : bool
        clear and reset background perm to perm_background, by default True

    Returns
    -------
    PyEIT3DMesh
        3D point cloud dataclass
    """
    if clear_bg:
        mesh = clear_perm(mesh=mesh, perm_background=perm_background)

    obj_vol = (
        np.sqrt(
            (-mesh.x_nodes - anomaly.x) ** 2
            + (mesh.y_nodes - anomaly.y) ** 2
            + (mesh.z_nodes - anomaly.z) ** 2
        )
        <= anomaly.d / 2
    )
    mesh.perm_array[obj_vol] = anomaly.perm
    mesh.material = anomaly.material
    return mesh


def rename_savedir(
    s_path: str, ball: BallAnomaly, ssms: ScioSpecMeasurementSetup
) -> None:
    """
    Rename the timestamp save directory to material, injection and size of the
    measurement procedure.

    Parameters
    ----------
    s_path : str
        save path
    ball : BallAnomaly
        description of the anomaly
    ssms : ScioSpecMeasurementSetup
        sciospec measurement dataclass
    """
    n_s_path = (
        s_path[:-24]
        + ball.material
        + "_skip_"
        + str(ssms.inj_skip)
        + "_d_"
        + str(ball.d)
    )
    print("renamed:", s_path, "to:")
    try:
        os.rename(s_path[:-5], n_s_path)
        print("\t", n_s_path)
    except BaseException:
        print("ERROR: Folder already exist or other problems.")


def empty_tank_measurement(
    COM_Ender,
    enderstat: Ender5Stat,
    COM_Sciospec,
    ssms: ScioSpecMeasurementSetup,
    s_path: str,
    ball: BallAnomaly,
    documentation: MeasurementInformation,
    sample_preamble: str,
    tank: TankProperties32x2(),
) -> None:
    """
    Creates the empty tank measurement. Please define before and after

    Parameters
    ----------
    COM_Ender : _type_
        serial connection to 3d printer
    enderstat : Ender5Stat
        ender 5 dataclass
    COM_Sciospec : _type_
        serial connection to sciospec eit device
    ssms : ScioSpecMeasurementSetup
        sciospec configuration dataclass
    s_path : str
        save path
    ball : BallAnomaly
        anomaly property dataclass
    documentation : MeasurementInformation
        documentation dataclass
    sample_preamble : str, ["before", "after"]
        name of the file before numbering it
    tank : TankProperties32x2
        tank properties dataclass
    """
    if enderstat.abs_z_pos + ball.d <= documentation.saline_height[0]:
        print("Move object out of the saline")
        return

    samples_counter = 0
    documentation.temperature = read_temperature(COM_Ender)
    current_time = datetime.now()
    documentation.timestamp = current_time.strftime("%d_%m_%Y_%Hh_%Mm")
    sciospec_data = sciospec_measurement(COM_Sciospec, ssms)

    for data in sciospec_data:
        current_time = datetime.now()
        documentation.timestamp = current_time.strftime("%d_%m_%Y_%Hh_%Mm")

        np.savez(
            s_path[:-5]
            + "empty_tank/"
            + sample_preamble
            + "_{0:06d}.npz".format(samples_counter),
            data=data,
            anomaly=ball,
            config=ssms,
            tank=tank,
            documentation=documentation,
        )
        samples_counter += 1
    SystemMessageCallback_usb_hs(COM_Sciospec, prnt_msg=False)
