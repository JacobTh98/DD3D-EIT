import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from .classes import TankProperties32x2, BallAnomaly, CSVConvertInfo  # PyEIT3DMesh
import csv
from sciopy.sciopy_dataclasses import ScioSpecMeasurementSetup, SingleFrame
import shutil
from itertools import chain
from tqdm import tqdm
from typing import Union

# from .functions import create_mesh, set_perm

import glob
from PIL import Image


def get_sample(l_path: str, idx: int) -> Union[np.lib.npyio.NpzFile, dict]:
    """
    Load a single sample out of a load path.

    Parameters
    ----------
    l_path : str
        load path
    idx : int
        sample index

    Returns
    -------
    Union[np.lib.npyio.NpzFile, dict]
        numpy measurement file, information dict
    """
    try:
        tmp = np.load(l_path + "data/sample_{0:06d}.npz".format(idx), allow_pickle=True)
        json_file = open(l_path + "info.json")
        info_dict = json.load(json_file)
        return tmp, info_dict
    except BaseException:
        print("Error during loading")


def temperature_history(
    l_path, plot: bool = True, save_plot: bool = False, nbins: int = 10
) -> np.ndarray:
    """
    Collects all temperature information of a measurement.
    You can plot and save the plottet result to the measurement directory.

    Parameters
    ----------
    l_path : _type_
        load path
    plot : bool, optional
        plot the temperature history, by default True
    save_plot : bool, optional
        save the plot to the l_path directory, by default False
    nbins : int, optional
        maximum number of x ticks, by default 10

    Returns
    -------
    np.ndarray
        temperature history
    """
    temp_hist = list()
    time_hist = list()
    for idx in range(len(os.listdir(l_path + "data/"))):
        tmp, _ = get_sample(l_path, idx)
        temp_hist.append(tmp["documentation"].tolist().temperature[0])
        time_hist.append(
            ":".join(tmp["documentation"].tolist().timestamp.split("_")[3:])
            .replace("h", "")
            .replace("m", "")
        )
    title = ".".join(tmp["documentation"].tolist().timestamp.split("_")[:3])
    temp_hist = np.array(temp_hist)
    if plot:
        # Auto Locator
        ax = plt.subplot(111)
        # plt.figure(figsize=(6, 4))
        t1 = "=".join(l_path.split("_")[1:3])
        t2 = "=".join(l_path.split("_")[3:5])[:-1]
        plt.title("measurement: " + t1 + ", " + t2 + "mm, " + title)
        plt.grid()
        ax.plot(time_hist, temp_hist)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_xlabel("Timestamp in hh:mm")
        ax.set_ylabel("Temperature in °C")
        plt.tight_layout()
        if save_plot:
            plt.savefig(l_path + "temperature_history.pdf")
            plt.savefig(l_path + "temperature_history.png", dpi=300)
        plt.show()
    return temp_hist


def get_mesh(tmp: np.lib.npyio.NpzFile):  # -> PyEIT3DMesh:
    """
    Load the mesh of a single .npz file.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        single measurement sample

    Returns
    -------
    PyEIT3DMesh
        mesh_obj dataclass
    """
    mesh_obj = tmp["mesh_obj"].tolist()
    return mesh_obj


def get_trajectory(
    l_path: str,
    plot_traj: bool = True,
    tank: TankProperties32x2 = TankProperties32x2(),
    elev: int = 10,
    azim: int = 30,
) -> np.ndarray:
    """
    Get all measured coordinates.
    It is default to plt a 3D visualiuation

    Parameters
    ----------
    l_path : str
        load path
    plot_traj : bool, optional
        3d plot of the coordinated, by default True
    tank : TankProperties32x2, optional
        tank properties, by default TankProperties32x2()
    elev : int, optional
        elevation angle of 3d plot, by default 10
    azim : int, optional
        azimut angle of 3d plot, by default 30

    Returns
    -------
    np.ndarray
        measured coordinates
    """
    dir_length = len(os.listdir(l_path + "data/"))
    traj_xyz = np.zeros((dir_length, 3))

    for idx in range(dir_length):
        tmp, _ = get_sample(l_path, idx)
        traj_xyz[idx, 0] = tmp["anomaly"].tolist().x
        traj_xyz[idx, 1] = tmp["anomaly"].tolist().y
        traj_xyz[idx, 2] = tmp["anomaly"].tolist().z

    if plot_traj:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        zyl_pnts = 50
        theta = np.linspace(0, 2 * np.pi, zyl_pnts)
        z = np.linspace(tank.T_bz[0], tank.T_bz[1], zyl_pnts)
        Z, Theta = np.meshgrid(z, theta)
        X = tank.T_r * np.cos(Theta)
        Y = tank.T_r * np.sin(Theta)
        ax.plot_surface(X, Y, Z, color="C7", alpha=0.2)
        p = ax.scatter(
            traj_xyz[:, 0],
            traj_xyz[:, 1],
            traj_xyz[:, 2],
            c=np.linspace(0, 1, traj_xyz.shape[0]),
            marker="o",
            s=25,
            alpha=1,
        )
        ax.set_xlim([tank.T_bx[0], tank.T_bx[1]])
        ax.set_ylim([tank.T_by[0], tank.T_by[1]])
        ax.set_zlim([tank.T_bz[0], tank.T_bz[1]])

        ax.set_xlabel("x pos [mm]")
        ax.set_ylabel("y pos [mm]")
        ax.set_zlabel("z pos [mm]")
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(
            p,
            shrink=0.7,
            orientation="horizontal",
            pad=0,
            label="Order from start to finish",
        )
        plt.tight_layout()
        plt.show()
    return np.unique(traj_xyz, axis=0)


def get_measured_potential(
    tmp: np.lib.npyio.NpzFile, shape_type="matrix"
) -> np.ndarray:
    """
    Read the measured complex potential data.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        measurement file
    shape_type : str, optional {'matrix', 'vector'}
        shape of the data, by default "matrix"

    Returns
    -------
    np.ndarray
        complex potential data
    """
    ssms = tmp["config"].tolist()
    ch_n = ssms.n_el // len(ssms.channel_group)
    pot_array = list()

    ch_group_srtng = np.zeros((len(ssms.channel_group), ch_n), dtype=complex)
    channel_switch = 0
    for frame in tmp["data"]:
        frame_tmp_dict = frame.__dict__
        group = frame_tmp_dict["channel_group"]
        for ch in range(ch_n):
            ch_group_srtng[group - 1, ch] = frame_tmp_dict[f"ch_{ch+1}"]
        channel_switch += 1
        if channel_switch == len(ssms.channel_group):
            ch_group_srtng = np.concatenate(ch_group_srtng)
            pot_array.append(ch_group_srtng)
            ch_group_srtng = np.zeros((len(ssms.channel_group), ch_n), dtype=complex)
            channel_switch = 0
    pot_array = np.array(pot_array)
    if shape_type == "matrix":
        return pot_array
    if shape_type == "vector":
        return np.concatenate(pot_array)


def get_inj_pattern(tmp: np.lib.npyio.NpzFile) -> int:
    """
    Reads the injection pattern from measured data.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        measurement file

    Returns
    -------
    int
        number of skipped electrodes
    """
    inj, gnd = tmp["data"][0].excitation_stgs
    skip = gnd - inj - 1
    print(f"{inj=}, {gnd=}, pattern: {skip=}")
    return skip


def get_channel_group(tmp: np.lib.npyio.NpzFile) -> int:
    """
    Reads the channel group from measured data.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        measurement file

    Returns
    -------
    int
        measurement channel group number
    """
    ch_grp = tmp["data"][0].channel_group
    print(f"Measured on channel group: {ch_grp}")
    return ch_grp


def get_BallAnomaly_properties(tmp: np.lib.npyio.NpzFile) -> BallAnomaly:
    """
    Get the properties of the placed anomaly.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        measurement file

    Returns
    -------
    BallAnomaly
        ball anomaly dataclass
    """
    return tmp["anomaly"].tolist()


def get_config(tmp: np.lib.npyio.NpzFile) -> ScioSpecMeasurementSetup:
    """
    Get the measurement configuration.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        measurement file

    Returns
    -------
    ScioSpecMeasurementSetup
        sciopy configuration dataclass
    """
    return tmp["config"].tolist()


def get_SingleFrame_exc_stage(dataframe: SingleFrame) -> np.ndarray:
    """
    Get the excitation electrodes from a single dataframe.

    A single tmp (measurement file) has 256 SingleFrames.

    Parameters
    ----------
    dataframe : SingleFrame
        single measurement iteration

    Returns
    -------
    np.ndarray
        injecting electrodes
    """
    return np.array(dataframe.excitation_stgs)


def get_excitation_stages(tmp: np.lib.npyio.NpzFile) -> np.array:
    """
    Get the excitation stages from a measurement.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        measurement file

    Returns
    -------
    np.array
        excitational stages
    """
    exc_stgs = list()
    for frame in tmp["data"]:
        exc_stgs.append(get_SingleFrame_exc_stage(frame))
    exc_stgs = np.array(exc_stgs)
    return exc_stgs


def prepare_csv_conv(l_path: str) -> CSVConvertInfo:
    s_path = l_path[:-1] + "_csv/"
    try:
        os.mkdir(s_path)
        print("Created save directory.")
    except BaseException:
        print("Directory already exists.")
    try:
        shutil.copyfile(l_path + "info.json", s_path + "info.json")
    except BaseException:
        print("No 'info.json' found.")
    try:
        shutil.copyfile(
            l_path + "temperature_history.pdf", s_path + "temperature_history.pdf"
        )
    except BaseException:
        print("No 'temperature_history.pdf' found.")

    with open(s_path + "data.csv", "w") as creating_new_csv_file:
        pass
    print("Empty .csv file created successfully")
    s_csv = s_path + "data.csv"
    n_samples = len(os.listdir(l_path + "data/"))
    return CSVConvertInfo(l_path, s_path, s_csv, n_samples)


def write_top_csv_row(
    conv_info: CSVConvertInfo, config: ScioSpecMeasurementSetup, anomaly: BallAnomaly
) -> None:
    """
    Creates the top row titles of the columns.

    Parameters
    ----------
    conv_info : CSVConvertInfo
        conversion info dataclass
    config : ScioSpecMeasurementSetup
        sciospec measurement config
    anomaly : BallAnomaly
        object properties
    """
    csv_top_row = [
        ["meas_num", "exc_stgs"],
        [f"obj_{anmly}_pos [mm]" for anmly in list(anomaly.__dict__.keys())[:-3]],
        ["obj d [mm]"],
        ["el_{0:02d}".format(el + 1) for el in range(config.n_el)],
    ]

    csv_top_row = list(chain(*csv_top_row))

    with open(conv_info.s_csv, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_top_row)
    print("Added top row to csv file.")


def parse_npzdata_in_csv(conv_info: CSVConvertInfo) -> None:
    """
    Convert all npz samples to csv.

    Parameters
    ----------
    conv_info : CSVConvertInfo
        conversion info dataclass
    """
    with open(conv_info.s_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        top_row = next(csv_reader, None)
        if top_row is not None:
            top_row_length = len(top_row)
        else:
            print("CSV file is empty, create top row: 'write_top_csv_row()'.")
            return
    try:
        tmp, _ = get_sample(conv_info.l_path, 0)
    except BaseException:
        print("Cant load sample 0.")

    rows_len = np.unique(get_excitation_stages(tmp), axis=0).shape[0]
    clmn_len = top_row_length
    print("Writing .csv...")
    for sample_idx in tqdm(range(len(os.listdir(conv_info.l_path + "data/")))):
        tmp, _ = get_sample(conv_info.l_path, sample_idx)
        config = get_config(tmp)
        extst = np.unique(get_excitation_stages(tmp), axis=0)
        anomaly = get_BallAnomaly_properties(tmp).__dict__
        pot = get_measured_potential(tmp)

        for i in range(rows_len):
            row_list = [None for _ in range(clmn_len)]
            row_list[0] = sample_idx
            ex1, ex2 = extst[i]
            row_list[1] = f"{ex1}, {ex2}"
            for anml_idx in range(4):
                row_list[anml_idx + 2] = list(anomaly.values())[anml_idx]
            for el_xx in range(config.n_el):
                row_list[el_xx + 6] = pot[i, el_xx]
            with open(conv_info.s_csv, "a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(row_list)
    print("Done.")


def get_permarray_FF(l_path: str, idx: int, h0: float = 1.0) -> np.ndarray:
    """
    Get perm array fast forward

    Parameters
    ----------
    l_path : str
        load path
    idx : int
        selected file index
    h0 : float, optional
        points per millimeter, by default 0.1

    Returns
    -------
    np.ndarray
        perm_array from PyEIT3DMesh dataclass
    """
    tmp, _ = get_sample(l_path, idx)
    tank = tmp["tank"].tolist()
    anomaly = get_BallAnomaly_properties(tmp)
    mesh_obj = create_mesh(tank, h0)
    mesh_obj = set_perm(mesh_obj, anomaly)
    return mesh_obj.perm_array


def get_pot_data_FF(l_path: str, idx: int) -> np.ndarray:
    """
    Get the absolute potential data fast forward.

    Parameters
    ----------
    l_path : str
        load path
    idx : int
        selected file index

    Returns
    -------
    np.ndarray
        absolute potential data
    """
    # get potential data fast forward
    tmp, _ = get_sample(l_path, idx)
    pot_data = get_measured_potential(tmp, "vector")
    return np.abs(pot_data)


def gif_inj_stages(
    tmp: np.lib.npyio.NpzFile,
    tank=TankProperties32x2(),
    elev: int = 10,
    azim: int = 10,
    n_el_per_channel: int = 16,
    duration: int = 100,
) -> None:
    """
    Generates a .gif for injection pattern visualization.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        single measurement saple
    tank : _type_, optional
        tank architecture dataclass, by default TankProperties32x2()
    elev : int, optional
        elevation angle, by default 10
    azim : int, optional
        azimut angle, by default 10
    n_el_per_channel : int, optional
        electrodes per channel, by default 16
    duration : int, optional
        gig duration, by default 100
    """

    def make_gif(gif_dir):
        frames = [Image.open(image) for image in np.sort(glob.glob(f"{gif_dir}/*.png"))]
        frame_one = frames[0]
        frame_one.save(
            "inj_pattern.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=duration,
            loop=0,
        )

    zyl_pnts = 50
    theta = np.linspace(0, 2 * np.pi, zyl_pnts)
    z = np.linspace(tank.T_bz[0], tank.T_bz[1], zyl_pnts)
    Z, Theta = np.meshgrid(z, theta)
    X = tank.T_r * np.cos(Theta)
    Y = tank.T_r * np.sin(Theta)
    phi_ypos = np.linspace(0, np.pi, n_el_per_channel, endpoint=False)
    phi_yneg = np.linspace(np.pi, 2 * np.pi, n_el_per_channel, endpoint=False)
    Z_r1 = tank.E_zr1
    Z_r2 = tank.E_zr2
    r = tank.T_r

    X_chgps = np.concatenate(
        [
            tank.T_r * np.cos(phi_ypos),
            tank.T_r * np.cos(phi_yneg),
            tank.T_r * np.cos(phi_ypos),
            tank.T_r * np.cos(phi_yneg),
        ]
    )
    Y_chgps = np.concatenate(
        [
            tank.T_r * np.sin(phi_ypos),
            tank.T_r * np.sin(phi_yneg),
            tank.T_r * np.sin(phi_ypos),
            tank.T_r * np.sin(phi_yneg),
        ]
    )
    Z_chgps = np.concatenate(
        [
            np.repeat(Z_r1, n_el_per_channel),
            np.repeat(Z_r1, n_el_per_channel),
            np.repeat(Z_r2, n_el_per_channel),
            np.repeat(Z_r2, n_el_per_channel),
        ]
    )
    chgps_c = np.concatenate(
        [[f"C{c}" for _ in range(n_el_per_channel)] for c in [2, 4, 0, 3]]
    )

    os.mkdir("tmp_gif_dir")

    inj_electrodes = np.unique(get_excitation_stages(tmp), axis=0) - 1
    img_num = 0
    for i1, i2 in inj_electrodes:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # phantom-tank border
        ax.plot_surface(X, Y, Z, color="C7", alpha=0.1)
        # plot channel groups
        ax.scatter(
            tank.T_r * np.cos(phi_ypos),
            tank.T_r * np.sin(phi_ypos),
            np.repeat(Z_r1, n_el_per_channel),
            color=chgps_c[:16],
            label="ch.gr. 1",
        )
        ax.scatter(
            tank.T_r * np.cos(phi_yneg),
            tank.T_r * np.sin(phi_yneg),
            np.repeat(Z_r1, n_el_per_channel),
            color=chgps_c[16:32],
            label="ch.gr. 3",
        )
        ax.scatter(
            tank.T_r * np.cos(phi_ypos),
            tank.T_r * np.sin(phi_ypos),
            np.repeat(Z_r2, n_el_per_channel),
            color=chgps_c[32:48],
            label="ch.gr. 2",
        )
        ax.scatter(
            tank.T_r * np.cos(phi_yneg),
            tank.T_r * np.sin(phi_yneg),
            np.repeat(Z_r2, n_el_per_channel),
            color=chgps_c[48:64],
            label="ch.gr. 4",
        )
        ax.plot(
            [X_chgps[i1], X_chgps[i2]],
            [Y_chgps[i1], Y_chgps[i2]],
            [Z_chgps[i1], Z_chgps[i2]],
            "--",
            c="black",
        )

        ax.set_xlim([tank.T_bx[0], tank.T_bx[1]])
        ax.set_ylim([tank.T_by[0], tank.T_by[1]])
        ax.set_zlim([tank.T_bz[0], tank.T_bz[1]])

        ax.set_xlabel("x pos [mm]")
        ax.set_ylabel("y pos [mm]")
        ax.set_zlabel("z pos [mm]")
        ax.view_init(elev=elev, azim=azim)
        plt.legend(loc="best", bbox_to_anchor=(0.57, 0.3, 0.5, 0.5))

        plt.tight_layout()
        plt.savefig("tmp_gif_dir/img_{0:04d}.png".format(img_num), dpi=250)
        img_num += 1
        plt.close()

    make_gif("tmp_gif_dir/")
    shutil.rmtree(r"tmp_gif_dir", ignore_errors=True)
