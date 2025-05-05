from modules.components import HexaEdgeFiducials, HexaFiducials, Fiducial, fit_hexagon_with_radius_constraint, plot_fitted_hexagon, AssemblyTrayFiducials, find_angle_to_rightmost_side_midpoint, SiliconFiducials, subtractValues, addValues
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def extractProtoModuleFiducials(file_name, sheet_name='WorkSheet_01', output_name='ProtoModuleFiducial.png', ToGantry=False):
    xls = pd.ExcelFile(file_name)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_name = 'Unnamed: 6'
    # extra TF and BF fiducials for alignment
    TFX = 0.0
    TFY = 0.0
    TF = Fiducial(TFX, TFY)
    BFX = float(df[col_name][19])
    BFY = float(df[col_name][20])
    BF = Fiducial(BFX, BFY)
    fids_TFBF = {
        'TF': TF,
        'BF': BF
    }
    if ToGantry:
        fids_TFBF['TF'] = TF.FlipY()
        fids_TFBF['BF'] = BF.FlipY()
    # pos 1
    map_fids_idx_pos = {
        "FD1": 24,
        "FD2": 30,
        "FD3": 36,
        "FD4": 42,
    }
    fids = {}
    for fid_name, idx in map_fids_idx_pos.items():
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fids[fid_name] = fid
    fiducials_pos1 = SiliconFiducials(
        fids, TF=TF, BF=BF)
    fiducials_pos1.visualize(output_name=output_name.replace(
        ".png", "_pos1.png"))
    if ToGantry:
        fiducials_pos1.ToGantry()
    return fiducials_pos1, fids_TFBF


def extractModuleFiducials(file_name, sheet_name='WorkSheet_01', output_name='ModuleFiducial.png', ToGantry=False):
    xls = pd.ExcelFile(file_name)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_name = 'Unnamed: 6'
    # extra TF and BF fiducials for alignment
    BFX = 0.0
    BFY = 0.0
    BF = Fiducial(BFX, BFY)
    TFX = float(df[col_name][18])
    TFY = float(df[col_name][19])
    TF = Fiducial(TFX, TFY)
    fids_TFBF = {
        'TF': TF,
        'BF': BF
    }
    if ToGantry:
        fids_TFBF['TF'] = TF.FlipY()
        fids_TFBF['BF'] = BF.FlipY()
    # pos 1
    map_fids_idx_pos = {
        "FD1": 37,
        "FD2": 44,
        "FD4": 51,
        "FD5": 58,
        "FD3": 25,
        "FD6": 31,
    }
    fids = {}
    for fid_name, idx in map_fids_idx_pos.items():
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fids[fid_name] = fid
    fiducials_pos1 = HexaFiducials(
        fids, TF=TF, BF=BF)
    fiducials_pos1.visualize(output_name=output_name.replace(
        ".png", "_pos1.png"))
    if ToGantry:
        fiducials_pos1.ToGantry()
    return fiducials_pos1, fids_TFBF


def extractHexaSideFiducials(file_name, sheet_name='WorkSheet_01', output_name='HexaFiducial.png', ToGantry=False, doSilicon=False):
    xls = pd.ExcelFile(file_name)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_name = 'Unnamed: 6'

    # extra TF and BF fiducials for alignment
    if not doSilicon:
        tf_idx = 141
        tf_idx = 129
        bf_idx = 157
        bf_idx = 145
    else:
        tf_idx = 15
        bf_idx = 31
    TFX = float(df[col_name][tf_idx])
    TFY = float(df[col_name][tf_idx + 1])
    TF = Fiducial(TFX, TFY)
    print("TF: ", TF)
    BFX = float(df[col_name][bf_idx])
    BFY = float(df[col_name][bf_idx + 1])
    BF = Fiducial(BFX, BFY)
    fids_TFBF = {
        'TF': TF,
        'BF': BF
    }
    if ToGantry:
        fids_TFBF['TF'] = TF.FlipY()
        fids_TFBF['BF'] = BF.FlipY()

    fiducials_pos1 = []
    if not doSilicon:
        pos1_indices = [8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68]
        pos1_indices = [14, 20, 26, 32, 38, 44, 50, 56, 62, 68]
        pos2_indices = [74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134]
        pos2_indices = [74, 80, 86, 92, 98, 104, 110, 116, 122]
    else:
        pos1_indices = [8, 38, 44, 50, 56, 62, 68, 74, 80,
                        86, 92, 98, 104, 110, 116, 122, 128, 134, 140]
        pos2_indices = [146, 152, 158, 164, 170, 176, 182, 188, 194, 200, 206]

    # pos 1
    for idx in pos1_indices:
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fiducials_pos1.append(fid)

    fiducials_pos1_hexaboard = HexaEdgeFiducials(
        fiducials_pos1, TF=TF, BF=BF)
    fiducials_pos1_hexaboard.visualize(output_name=output_name.replace(
        ".png", "_pos1.png"))
    if ToGantry:
        fiducials_pos1_hexaboard.ToGantry()

    # pos 2
    fiducials_pos2 = []
    for idx in pos2_indices:
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fiducials_pos2.append(fid)

    fiducials_pos2_hexaboard = HexaEdgeFiducials(
        fiducials_pos2, TF=TF, BF=BF)
    fiducials_pos2_hexaboard.visualize(output_name=output_name.replace(
        ".png", "_pos2.png"))
    if ToGantry:
        fiducials_pos2_hexaboard.ToGantry()

    return fiducials_pos1_hexaboard, fiducials_pos2_hexaboard, fids_TFBF


def extractHexaFiducials(file_name, sheet_name='WorkSheet_01', output_name='HexaFiducial.png', ToGantry=False):
    xls = pd.ExcelFile(file_name)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_name = 'Unnamed: 6'
    # extra TF and BF fiducials for alignment
    TFX = float(df[col_name][141])
    TFX = float(df[col_name][147])
    TFY = float(df[col_name][142])
    TFY = float(df[col_name][148])
    TF = Fiducial(TFX, TFY)
    BFX = float(df[col_name][157])
    BFX = float(df[col_name][163])
    BFY = float(df[col_name][158])
    BFY = float(df[col_name][164])
    BF = Fiducial(BFX, BFY)
    fids_TFBF = {
        'TF': TF,
        'BF': BF
    }
    if ToGantry:
        fids_TFBF['TF'] = TF.FlipY()
        fids_TFBF['BF'] = BF.FlipY()

    # pos 1
    fiducials_pos1 = []
    # for idx in [8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68]:
    for idx in [14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74]:
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fiducials_pos1.append(fid)
    fiducials_pos1_hexaboard = HexaEdgeFiducials(
        fiducials_pos1, TF=TF, BF=BF)

    # six fiducials
    map_fids_idx_pos1 = {
        "FD1": 165,
        "FD2": 172,
        "FD3": 179,
        "FD4": 186,
        "FD5": 193,
        "FD6": 200
    }
    map_fids_idx_pos1 = {
        "FD1": 171,
        "FD2": 178,
        "FD3": 185,
        "FD4": 192,
        "FD5": 199,
        "FD6": 206
    }
    fids = {}
    for fid_name, idx in map_fids_idx_pos1.items():
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fids[fid_name] = fid
    fiducials_pos1_hexaboard_6Fid = HexaFiducials(
        fids, TF=TF, BF=BF)
    fiducials_pos1_hexaboard_6Fid.visualize(output_name=output_name.replace(
        ".png", "_pos1_6Fid.png"))

    # pos 2
    fiducials_pos2 = []
    # for idx in [74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134]:
    for idx in [80, 86, 92, 98, 104, 110, 116, 122, 128, 134, 140]:
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fiducials_pos2.append(fid)
    fiducials_pos2_hexaboard = HexaEdgeFiducials(
        fiducials_pos2, TF=TF, BF=BF)
    fiducials_pos2_hexaboard.visualize(output_name=output_name.replace(
        ".png", "_pos2.png"))

    # six fiducials
    map_fids_idx_pos2 = {
        "FD1": 207,
        "FD2": 214,
        "FD3": 221,
        "FD4": 228,
        "FD5": 235,
        "FD6": 242
    }
    map_fids_idx_pos2 = {
        "FD1": 213,
        "FD2": 220,
        "FD3": 227,
        "FD4": 234,
        "FD5": 241,
        "FD6": 248
    }
    fids = {}
    for fid_name, idx in map_fids_idx_pos2.items():
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fids[fid_name] = fid
    fiducials_pos2_hexaboard_6Fid = HexaFiducials(
        fids, TF=TF, BF=BF)
    fiducials_pos2_hexaboard_6Fid.visualize(output_name=output_name.replace(
        ".png", "_pos2_6Fid.png"))
    if ToGantry:
        fiducials_pos1_hexaboard.ToGantry()
        fiducials_pos2_hexaboard.ToGantry()
        fiducials_pos1_hexaboard_6Fid.ToGantry()
        fiducials_pos2_hexaboard_6Fid.ToGantry()

    if 1:
        # make plots of the fiducials
        fig, ax = plt.subplots()
        for fid in fiducials_pos1_hexaboard.fiducials:
            ax.plot(fid.X, fid.Y, 'ro')
        for name, fid in fiducials_pos1_hexaboard_6Fid.fiducials.items():
            ax.plot(fid.X, fid.Y, 'bo')
            ax.text(fid.X, fid.Y, name, fontsize=12)
        ax.set_title("Fiducials pos1")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        plt.savefig(output_name.replace(
            ".png", "_pos1_fiducials_comp.png"), dpi=300)
        plt.close()
        fig, ax = plt.subplots()
        for fid in fiducials_pos2_hexaboard.fiducials:
            ax.plot(fid.X, fid.Y, 'ro')
        for name, fid in fiducials_pos2_hexaboard_6Fid.fiducials.items():
            ax.plot(fid.X, fid.Y, 'bo')
            ax.text(fid.X, fid.Y, name, fontsize=12)
        ax.set_title("Fiducials pos2")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        plt.savefig(output_name.replace(
            ".png", "_pos2_fiducials_comp.png"), dpi=300)
        plt.close()
    return fiducials_pos1_hexaboard, fiducials_pos2_hexaboard, fids_TFBF, fiducials_pos1_hexaboard_6Fid, fiducials_pos2_hexaboard_6Fid


def extractBPFiducials(file_name, sheet_name='WorkSheet_01', output_name='BPFiducial.png', ToGantry=False):
    xls = pd.ExcelFile(file_name)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_name = 'Unnamed: 6'
    # extra TF and BF fiducials for alignment
    TFX = float(df[col_name][9])
    TFY = float(df[col_name][10])
    TF = Fiducial(TFX, TFY)
    BFX = float(df[col_name][25])
    BFY = float(df[col_name][26])
    BF = Fiducial(BFX, BFY)
    fids_TFBF = {
        'TF': TF,
        'BF': BF
    }
    if ToGantry:
        fids_TFBF['TF'] = TF.FlipY()
        fids_TFBF['BF'] = BF.FlipY()
    # pos 1
    fiducials_pos1 = []
    for idx in [32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98, 104, 110, 116]:
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fiducials_pos1.append(fid)
    fiducials_pos1_BP = HexaEdgeFiducials(
        fiducials_pos1, TF=TF, BF=BF)
    fiducials_pos1_BP.visualize(output_name=output_name.replace(
        ".png", "_pos1.png"))
    if ToGantry:
        fiducials_pos1_BP.ToGantry()
    # pos 2
    fiducials_pos2 = []
    for idx in [122, 128, 134, 140, 146, 152, 158, 164, 170, 176, 182, 188, 194, 200, 206]:
        fidX = float(df[col_name][idx])
        fidY = float(df[col_name][idx + 1])
        fid = Fiducial(fidX, fidY)
        fiducials_pos2.append(fid)
    fiducials_pos2_BP = HexaEdgeFiducials(
        fiducials_pos2, TF=TF, BF=BF)
    fiducials_pos2_BP.visualize(output_name=output_name.replace(
        ".png", "_pos2.png"))
    if ToGantry:
        fiducials_pos2_BP.ToGantry()
    return fiducials_pos1_BP, fiducials_pos2_BP, fids_TFBF


def extractTrayFiducials(output_name='TrayFiducial.png', ToGantry=False):
    BF = Fiducial(0.0, 0.0)
    TF = Fiducial(0.0, 392.469)
    OP1 = Fiducial(60.494, 285.311)
    CP1 = Fiducial(135.383, 288.348)
    OP2 = Fiducial(167.248, 98.988)
    CP2 = Fiducial(92.356, 95.922)

    fiducials = {
        'TF': TF,
        'BF': BF,
        'CP1': CP1,
        'OP1': OP1,
        'CP2': CP2,
        'OP2': OP2
    }
    fiducials = AssemblyTrayFiducials(fiducials)
    fiducials.visualize(output_name=output_name)
    if ToGantry:
        fiducials.ToGantry()
    return fiducials


def plot_truth_vs_recos(truth, recos, line_length=20, output_name="plots/hexagon_comparison.png"):
    tx, ty, t_angle = truth
    t_angle = np.radians(t_angle)
    t_end = [tx + line_length *
             np.cos(t_angle), ty + line_length * np.sin(t_angle)]

    plt.figure(figsize=(5, 5))

    # Plot truth line and point
    plt.plot([tx, t_end[0]], [ty, t_end[1]], 'b--',
             label='Truth Line', linewidth=2)
    plt.scatter(tx, ty, color='blue', marker='o', s=100,
                edgecolors='black', label='Truth Point')

    colors = ['red', 'orange', 'green']

    for i, (rx, ry, r_angle) in enumerate(recos):
        r_angle = np.radians(r_angle)
        r_end = [rx + line_length *
                 np.cos(r_angle), ry + line_length * np.sin(r_angle)]
        r_end_goal = [rx + line_length *
                      np.cos(t_angle), ry + line_length * np.sin(t_angle)]

        # Plot reco line and point
        if i >= 7:
            col = colors[2]
        elif i >= 5:
            col = colors[1]
        else:
            col = colors[0]
        plt.plot([rx, r_end[0]], [ry, r_end[1]], '-',
                 label=f'Reco Line #{i}', linewidth=2, color=col)
        plt.plot([rx, r_end_goal[0]], [ry, r_end_goal[1]], f'--',
                 label=f'Reco Line Goal #{i}', linewidth=2, color=col)
        plt.scatter(rx, ry, color=col, marker='^', s=100,
                    edgecolors='black', label='Reco Point')

        # plt.gca().set_aspect('equal')
        # plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.tight_layout()
    plt.savefig(output_name)


def plot_truth_vs_recos_2plots(truth, recos, output_name="plots/hexagon_comparison.png", useOddEven=False, colors=None, legends=None):
    tx, ty, t_angle = truth
    t_angle_rad = np.radians(t_angle)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_main = axes[0]
    ax_angle = axes[1]

    # --- First plot: points only ---
    label = None
    if legends != None:
        label = legends[0]
    ax_main.scatter(0, 0, color='blue', marker='o', s=100,
                    edgecolors='blue', label=label)

    for i, (rx, ry, _) in enumerate(recos):
        if colors == None:
            colors = ['red', 'orange', 'green']
            if not useOddEven:
                if i >= 8:
                    col = colors[2]
                elif i >= 5:
                    col = colors[1]
                else:
                    col = colors[0]
            else:
                if i % 2 == 0:
                    col = colors[0]
                else:
                    col = colors[1]
                if i == 6:
                    col = colors[2]
        else:
            col = colors[i % len(colors)]
        label = None
        if legends != None and i < len(legends):
            label = legends[i + 1]
        ax_main.scatter(rx - tx, ry - ty, color=col, marker='^', s=100,
                        edgecolors=col, label=label)

    ax_main.grid(True)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X [mm]')
    ax_main.set_ylabel('Y [mm]')
    ax_main.ticklabel_format(useOffset=True, axis='x', style='sci')
    ax_main.set_title('XY Positions')
    if legends != None:
        ax_main.legend(loc='best')

    # --- Second plot: all arrows starting from the same point ---
    origin_x = 0
    origin_y = 0

    if t_angle_rad > np.pi/2.0 or t_angle_rad < -np.pi/2.0:
        t_angle_rad_approx = np.radians(180.0)
        doLeft = True
    else:
        t_angle_rad_approx = 0.0
        doLeft = False

    # Plot truth arrow
    ax_angle.quiver(origin_x, origin_y,
                    np.cos(t_angle_rad_approx), np.sin(t_angle_rad_approx),
                    angles='xy', scale_units='xy', scale=10,
                    color='blue', linewidth=2)

    # Plot reco arrows
    y_vals = []
    print("recos", recos)
    for i, (_, _, r_angle) in enumerate(recos):
        r_angle_rad = np.radians(r_angle)
        if colors == None:
            colors = ['red', 'orange', 'green']
            if not useOddEven:
                if i >= 8:
                    col = colors[2]
                elif i >= 5:
                    col = colors[1]
                else:
                    col = colors[0]
            else:
                if i % 2 == 0:
                    col = colors[0]
                else:
                    col = colors[1]
                if i == 6:
                    col = colors[2]
        else:
            col = colors[i % len(colors)]

        r_angle_rad_diff = r_angle_rad - t_angle_rad + t_angle_rad_approx
        ax_angle.quiver(origin_x, origin_y, np.cos(r_angle_rad_diff),
                        np.sin(r_angle_rad_diff),
                        angles='xy', scale_units='xy', scale=10,
                        color=col, linewidth=2)
        y_vals.append(np.sin(r_angle_rad))

    # Parameters
    angle_center = t_angle_rad_approx  # radians
    angle_plus = np.deg2rad(0.01)      # convert 0.01 deg to rad
    angle_plus2 = np.deg2rad(0.02)     # convert 0.02 deg to rad
    length = 10  # same as your quiver scale

    # Compute three points: center, left boundary, right boundary
    left_x = origin_x + length * np.cos(angle_center + angle_plus)
    left_x2 = origin_x + length * np.cos(angle_center + angle_plus2)
    left_y = origin_y + length * np.sin(angle_center + angle_plus)
    left_y2 = origin_y + length * np.sin(angle_center + angle_plus2)

    right_x = origin_x + length * np.cos(angle_center - angle_plus)
    right_x2 = origin_x + length * np.cos(angle_center - angle_plus2)
    right_y = origin_y + length * np.sin(angle_center - angle_plus)
    right_y2 = origin_y + length * np.sin(angle_center - angle_plus2)

    # Create the polygon for the cone
    cone = Polygon([[origin_x, origin_y],
                    [left_x, left_y],
                    [right_x, right_y]],
                   closed=True, color='blue', alpha=0.15, label='±0.01° cone')
    cone2 = Polygon([[origin_x, origin_y],
                     [left_x2, left_y2],
                     [right_x2, right_y2]],
                    closed=True, color='blue', alpha=0.08, label='±0.02° cone')

    # Add it to the plot
    ax_angle.add_patch(cone)
    ax_angle.add_patch(cone2)

    ax_angle.set_ylim(-0.0001, 0.0001)
    # ax_angle.set_ylim(-0.0011, 0.0011)
    # ax_angle.set_xlim(-0.11, 0.11)
    if doLeft:
        ax_angle.set_xlim(-0.11, 0.)
    else:
        ax_angle.set_xlim(0., 0.11)
    ax_angle.grid(True)
    # ax_angle.set_aspect('equal')
    ax_angle.legend(loc='best')
    ax_angle.set_title('Angle Directions')

    plt.savefig(output_name)
    plt.close()


def compareTray2HexaEdges(doSilicon=False):
    if not doSilicon:
        files = [
            # "data/Hex_Position_wPUT_DryRun1.xls",
            # "data/Hex_Position_wPUT_DryRun2.xls",
            # "data/Hex_Position_wPUT_DryRun1_04_24_2025.xls",
            # "data/Hex_Position_wPUT_DryRun2_04_24_2025.xls",
            # "data/Hex_Position_wPUT_DryRun3_04_24_2025.xls",
            # "data/Hex_Position_wPUT_DryRun1_afterchanges_04_24_2025.xls",
            # "data/Hex_Position_wPUT_DryRun2_afterchanges_04_24_2025.xls",
            # "data/Hex_Position_wPUT_DryRun1_afterchanges_04_25_2025.xls",
            # "data/Hex_Position_wPUT_DryRun2_afterchanges_04_25_2025.xls",
            # "data/Hex_Position_wPUT_DryRun1_04_27_2025.xls",
            # "data/Hex_Position_Edges_and_Fiducials_Run2_04_27_2025.xls",
            # "data/Hex_Position_wPUT_DryRun1_AfterChanges_04_28_2025.xls",
            # "data/Hex_Position_wPUT_DryRun2_AfterChanges_04_28_2025.xls",
            # "data/Hex_Position_wPUT_DryRun2_AfterChanges_Swapped_04_28_2025.xls",
            # "data/Ti_Hex_Position_wPUT_DryRun1_05_01_2025.xls",
            "data/Hex277P1_268P2_Position_Edges_and_Fiducials_Run1_05_02_2025.xls",
            "data/Hex277P1_268P2_Position_Edges_and_Fiducials_Run2_05_02_2025.xls"
        ]
    else:
        files = [
            "data/Si_wPUT_DryRun1_04-26-2025.xls",
        ]

    diff_X1 = []
    diff_Y1 = []
    diff_angle1 = []
    diff_X2 = []
    diff_Y2 = []
    diff_angle2 = []

    truths_1 = []
    recos_1 = []
    truths_2 = []
    recos_2 = []

    ToGantry = False
    if not doSilicon:
        # target_radius = 96.05376428507805
        target_radius = 96.296
    else:
        target_radius = 96.175

    fids_TFBF_new = {
        'TF': Fiducial(439.522, -699.113),
        'BF': Fiducial(423.441, -1091.249)
    }

    for idx, file in enumerate(files):
        tray_org = extractTrayFiducials(
            "plots/TrayFiducial.png", ToGantry=ToGantry)
        print(f"Processing {file}")
        hex1, hex2, fids_TFBF = extractHexaSideFiducials(
            file, output_name=f"plots/test_{idx}.png", ToGantry=ToGantry, doSilicon=doSilicon)

        hex1 = hex1.Align(fids_TFBF_new)
        hex2 = hex2.Align(fids_TFBF_new)

        results = fit_hexagon_with_radius_constraint(
            hex1, target_radius)
        plot_fitted_hexagon(
            results, f"plots/test_{idx}_fitted_hexagon_pos1.png")
        hex1.fitted_hexagon = results
        results = fit_hexagon_with_radius_constraint(
            hex2, target_radius)
        plot_fitted_hexagon(
            results, f"plots/test_{idx}_fitted_hexagon_pos2.png")
        hex2.fitted_hexagon = results

        angle1 = find_angle_to_rightmost_side_midpoint(
            hex1.fitted_hexagon['center'], hex1.fitted_hexagon['radius'], hex1.fitted_hexagon['theta'], False)
        angle2 = find_angle_to_rightmost_side_midpoint(
            hex2.fitted_hexagon['center'], hex2.fitted_hexagon['radius'], hex2.fitted_hexagon['theta'], True)
        print("Angle 1:", angle1[0])
        print("Angle 2:", angle2[0])

        # tray = tray_org.Align(fids_TFBF)
        tray = tray_org.Align(fids_TFBF_new)
        tray.visualize(f"plots/TrayFiducial_{idx}_aligned.png")
        print("Tray center pos1 :", tray.GetCenter(1))
        print("Tray angle pos1 :", tray.GetAngle(1))
        print("Hexa 1 center:", hex1.fitted_hexagon["center"])
        # print("Hexa 1 angle:", hex1.fitted_hexagon["theta"])
        print("Tray center pos2 :", tray.GetCenter(2))
        print("Tray angle pos2 :", tray.GetAngle(2))
        print("Hexa 2 center:", hex2.fitted_hexagon["center"])
        # print("Hexa 2 angle:", hex2.fitted_hexagon["theta"])

        diff_X1.append(
            tray.GetCenter(1)[0] - hex1.fitted_hexagon["center"][0])
        diff_Y1.append(
            tray.GetCenter(1)[1] - hex1.fitted_hexagon["center"][1])
        diff_angle1.append(
            tray.GetAngle(1) - angle1[0])
        diff_X2.append(
            tray.GetCenter(2)[0] - hex2.fitted_hexagon["center"][0])
        diff_Y2.append(
            tray.GetCenter(2)[1] - hex2.fitted_hexagon["center"][1])
        diff_angle2.append(
            tray.GetAngle(2) - angle2[0])

        truths_1.append(
            [tray.GetCenter(1)[0], tray.GetCenter(1)[1], tray.GetAngle(1)])
        truths_2.append(
            [tray.GetCenter(2)[0], tray.GetCenter(2)[1], tray.GetAngle(2)])
        recos_1.append([hex1.fitted_hexagon["center"][0],
                       hex1.fitted_hexagon["center"][1], angle1[0]])
        recos_2.append([hex2.fitted_hexagon["center"][0],
                       hex2.fitted_hexagon["center"][1], angle2[0]])

    print("Diff X1:", diff_X1)
    print("Diff Y1:", diff_Y1)
    print("Diff angle1:", diff_angle1)
    print("Diff X2:", diff_X2)
    print("Diff Y2:", diff_Y2)
    print("Diff angle2:", diff_angle2)

    plot_truth_vs_recos_2plots(truths_1[0], recos_1,
                               output_name="plots/hexa_tray_edge_comparison_pos1.png", colors=['red', 'orange', 'green', 'purple', 'pink', 'brown', 'gray'])
    plot_truth_vs_recos_2plots(truths_2[0], recos_2,
                               output_name="plots/hexa_tray_edge_comparison_pos2.png", colors=['red', 'orange', 'green', 'purple', 'pink', 'brown', 'gray'])

    return truths_1, recos_1, truths_2, recos_2


def compareHexaFid2HexaEdges():
    """
    compare the results using different methods of the fiducial measurements:
    1. four fiducials
    2. two fiducials
    3. three sides
    """
    files = [
        # "data/Hex_Position_Edges_and_Fiducials_Run1_04_26_2025.xls",
        # "data/Hex_Position_Edges_and_Fiducials_Run2_04_26_2025.xls",
        # "data/Hex_Position_Edges_and_Fiducials_Run3_04_26_2025.xls",
        # "data/Hex_Position_Edges_and_Fiducials_Run1_04_27_2025.xls",
        "data/Hex277P1_268P2_Position_Edges_and_Fiducials_Run1_05_01_2025.xls",
    ]

    target_radius = 96.05376428507805
    target_radius = 96.296

    truths_1 = []
    recos_1 = []
    truths_2 = []
    recos_2 = []

    # TF_new = Fiducial(141.981122, -700.740873)
    # TF_new = Fiducial(139.340, -699.570)
    # BF_new = Fiducial(122.303276, -1092.058439)
    # BF_new = Fiducial(123.390, -1091.704)
    # fids_TFBF_new = {
    #    'TF': TF_new,
    #    'BF': BF_new
    # }

    fids_TFBF_new = {
        'TF': Fiducial(439.522, -699.113),
        'BF': Fiducial(423.441, -1091.249)
    }

    for idx, file in enumerate(files):
        hex1, hex2, fids_TFBF, hex1_6Fids, hex2_6Fids = extractHexaFiducials(
            file, output_name=f"plots/test_{idx}.png", ToGantry=False)

        hex1 = hex1.Align(fids_TFBF_new)
        hex2 = hex2.Align(fids_TFBF_new)

        hex1_6Fids = hex1_6Fids.Align(fids_TFBF_new)
        hex2_6Fids = hex2_6Fids.Align(fids_TFBF_new)

        results = fit_hexagon_with_radius_constraint(
            hex1, target_radius)
        plot_fitted_hexagon(
            results, f"plots/test_{idx}_fitted_hexagon_pos1.png")
        hex1.fitted_hexagon = results
        results = fit_hexagon_with_radius_constraint(
            hex2, target_radius)
        plot_fitted_hexagon(
            results, f"plots/test_{idx}_fitted_hexagon_pos2.png")
        hex2.fitted_hexagon = results

        angle1 = find_angle_to_rightmost_side_midpoint(
            hex1.fitted_hexagon['center'], hex1.fitted_hexagon['radius'], hex1.fitted_hexagon['theta'], False)
        angle2 = find_angle_to_rightmost_side_midpoint(
            hex2.fitted_hexagon['center'], hex2.fitted_hexagon['radius'], hex2.fitted_hexagon['theta'], True)

        if idx == 0:
            truth_1_base = [hex1_6Fids.GetCenter()[0], hex1_6Fids.GetCenter()[
                1], hex1_6Fids.GetAngle()]
            truth_2_base = [hex2_6Fids.GetCenter()[0], hex2_6Fids.GetCenter()[
                1], hex2_6Fids.GetAngle()]

        truth = [hex1_6Fids.GetCenter()[0],
                 hex1_6Fids.GetCenter()[1], hex1_6Fids.GetAngle()]
        truth_diff = [truth[0] - truth_1_base[0], truth[1] -
                      truth_1_base[1], truth[2] - truth_1_base[2]]

        truths_1.append(truth)
        recos_1.append([hex1.fitted_hexagon["center"][0] - truth_diff[0],
                       hex1.fitted_hexagon["center"][1] - truth_diff[1], angle1[0] - truth_diff[2]])
        recos_1.append([hex1_6Fids.GetCenter(use4FDs=False)[0] - truth_diff[0],
                       hex1_6Fids.GetCenter(use4FDs=False)[1] - truth_diff[1], hex1_6Fids.GetAngle(use4FDs=False) - truth_diff[2]])

        truth = [hex2_6Fids.GetCenter()[0],
                 hex2_6Fids.GetCenter()[1], hex2_6Fids.GetAngle()]
        truth_diff = [truth[0] - truth_2_base[0], truth[1] -
                      truth_2_base[1], truth[2] - truth_2_base[2]]
        truths_2.append(truth)
        recos_2.append([hex2.fitted_hexagon["center"][0] - truth_diff[0],
                        hex2.fitted_hexagon["center"][1] - truth_diff[1], angle2[0] - truth_diff[2]])
        recos_2.append([hex2_6Fids.GetCenter(use4FDs=False)[0] - truth_diff[0],
                        hex2_6Fids.GetCenter(use4FDs=False)[1] - truth_diff[1], hex2_6Fids.GetAngle(use4FDs=False) - truth_diff[2]])

        print("\n\n********")
        print("Pos1: fitted hexagon center:", hex1.fitted_hexagon["center"])
        print("Pos1: fitted hexagon angle:", angle1[0])
        print("Pos1 diff", hex1.fitted_hexagon["center"][0] - hex1_6Fids.GetCenter()[0],
              hex1.fitted_hexagon["center"][1] - hex1_6Fids.GetCenter()[1],
              angle1[0] - hex1_6Fids.GetAngle())
        print("Pos1, center with 4 fiducials:", hex1_6Fids.GetCenter())
        print("Pos1, angle with 4 fiducials:", hex1_6Fids.GetAngle())
        print("Pos1, center with 2 fiducials:",
              hex1_6Fids.GetCenter(use4FDs=False))
        print("Pos1, angle with 2 fiducials:",
              hex1_6Fids.GetAngle(use4FDs=False))

        print("Pos2: fitted hexagon center:", hex2.fitted_hexagon["center"])
        print("Pos2: fitted hexagon angle:", angle2[0])
        print("Pos2 diff", hex2.fitted_hexagon["center"][0] - hex2_6Fids.GetCenter()[0],
              hex2.fitted_hexagon["center"][1] - hex2_6Fids.GetCenter()[1],
              angle2[0] - hex2_6Fids.GetAngle())
        print("Pos2, center with 4 fiducials:", hex2_6Fids.GetCenter())
        print("Pos2, angle with 4 fiducials:", hex2_6Fids.GetAngle())
        print("Pos2, center with 2 fiducials:",
              hex2_6Fids.GetCenter(use4FDs=False))
        print("Pos2, angle with 2 fiducials:",
              hex2_6Fids.GetAngle(use4FDs=False))

    plot_truth_vs_recos_2plots(truths_1[0], recos_1,
                               output_name="plots/hexa_Fid_Edge_comparison_pos1_2plots.png", useOddEven=True)
    plot_truth_vs_recos_2plots(truths_2[0], recos_2,
                               output_name="plots/hexa_Fid_Edge_comparison_pos2_2plots.png", useOddEven=True)

    return truths_1, recos_1, truths_2, recos_2


def compareBPFiducials():
    files = [
        "data/BP_Position_Run1_04_25_2025.xls",
        "data/BP_Position_Run2_04_25_2025.xls",
        "data/BP_Position_Run3_04_25_2025.xls",
        "data/BP_Position_Run4_04_25_2025.xls",
        "data/BP_Position_Run5_04_25_2025.xls",
        "data/BP_Position_Run1_Swapped_04_25_2025.xls",
        "data/BP_Position_Run2_Swapped_04_25_2025.xls",
        "data/BP_Position_Run3_Swapped_04_25_2025.xls",
    ]
    files = [
        "data/Ti_BP_Position_Run1_04_26_2025.xls",
        "data/Ti_BP_Position_Run2_04_26_2025.xls",
        "data/Ti_BP_Position_Run3_04_26_2025.xls",
        "data/Ti_BP_Position_Run4_04_26_2025.xls",
        "data/Ti_BP_Position_Run5_04_26_2025.xls",
    ]
    files = [
        "data/CuW_BP_Position_83P1_112P2_04_26_2025.xls"
    ]

    diff_X1 = []
    diff_Y1 = []
    diff_angle1 = []
    diff_X2 = []
    diff_Y2 = []
    diff_angle2 = []

    truths_1 = []
    recos_1 = []
    truths_2 = []
    recos_2 = []

    TF_new = Fiducial(141.981122, -700.740873)
    BF_new = Fiducial(122.303276, -1092.058439)
    fids_TFBF_new = {
        'TF': TF_new,
        'BF': BF_new
    }

    ToGantry = False
    target_radius = 96.383

    for idx, file in enumerate(files):
        tray_org = extractTrayFiducials(
            "plots/TrayFiducial.png", ToGantry=ToGantry)
        print(f"Processing {file}")
        hex1, hex2, fids_TFBF = extractBPFiducials(
            file, output_name=f"plots/BP_{idx}.png", ToGantry=ToGantry)

        hex1 = hex1.Align(fids_TFBF_new)
        hex2 = hex2.Align(fids_TFBF_new)

        results = fit_hexagon_with_radius_constraint(
            hex1, target_radius)
        plot_fitted_hexagon(
            results, f"plots/BP_{idx}_fitted_hexagon_pos1.png")

        hex1.fitted_hexagon = results
        results = fit_hexagon_with_radius_constraint(
            hex2, target_radius)
        plot_fitted_hexagon(
            results, f"plots/BP_{idx}_fitted_hexagon_pos2.png")
        hex2.fitted_hexagon = results
        angle1 = find_angle_to_rightmost_side_midpoint(
            hex1.fitted_hexagon['center'], hex1.fitted_hexagon['radius'], hex1.fitted_hexagon['theta'], False)
        angle2 = find_angle_to_rightmost_side_midpoint(
            hex2.fitted_hexagon['center'], hex2.fitted_hexagon['radius'], hex2.fitted_hexagon['theta'], True)
        print("Angle 1:", angle1[0])
        print("Angle 2:", angle2[0])
        # tray = tray_org.Align(fids_TFBF)
        tray = tray_org.Align(fids_TFBF_new)
        tray.visualize(f"plots/TrayFiducial_{idx}_aligned.png")
        print("Tray center pos1 :", tray.GetCenter(1))
        print("Tray angle pos1 :", tray.GetAngle(1))
        print("Hexa 1 center:", hex1.fitted_hexagon["center"])
        # print("Hexa 1 angle:", hex1.fitted_hexagon["theta"])
        print("Tray center pos2 :", tray.GetCenter(2))
        print("Tray angle pos2 :", tray.GetAngle(2))
        print("Hexa 2 center:", hex2.fitted_hexagon["center"])
        # print("Hexa 2 angle:", hex2.fitted_hexagon["theta"])

        diff_X1.append(
            tray.GetCenter(1)[0] - hex1.fitted_hexagon["center"][0])
        diff_Y1.append(
            tray.GetCenter(1)[1] - hex1.fitted_hexagon["center"][1])
        diff_angle1.append(
            tray.GetAngle(1) - angle1[0])
        diff_X2.append(
            tray.GetCenter(2)[0] - hex2.fitted_hexagon["center"][0])
        diff_Y2.append(
            tray.GetCenter(2)[1] - hex2.fitted_hexagon["center"][1])
        diff_angle2.append(
            tray.GetAngle(2) - angle2[0])
        truths_1.append(
            [tray.GetCenter(1)[0], tray.GetCenter(1)[1], tray.GetAngle(1)])
        truths_2.append(
            [tray.GetCenter(2)[0], tray.GetCenter(2)[1], tray.GetAngle(2)])
        recos_1.append([hex1.fitted_hexagon["center"][0],
                       hex1.fitted_hexagon["center"][1], angle1[0]])
        recos_2.append([hex2.fitted_hexagon["center"][0],
                        hex2.fitted_hexagon["center"][1], angle2[0]])
    print("Diff X1:", diff_X1)
    print("Diff Y1:", diff_Y1)
    print("Diff angle1:", diff_angle1)
    print("Diff X2:", diff_X2)
    print("Diff Y2:", diff_Y2)
    print("Diff angle2:", diff_angle2)
    plot_truth_vs_recos(truths_1[0], recos_1, line_length=0.2,
                        output_name="plots/BP_comparison_pos1.png")
    plot_truth_vs_recos(truths_2[0], recos_2, line_length=0.2,
                        output_name="plots/BP_comparison_pos2.png")
    plot_truth_vs_recos_2plots(truths_1[0], recos_1,
                               output_name="plots/BP_comparison_pos1_2plots.png")
    plot_truth_vs_recos_2plots(truths_2[0], recos_2,
                               output_name="plots/BP_comparison_pos2_2plots.png")


def checkProtoModuleFiducials():
    files = [
        ["pos1", "data/MLF3W2TT0116.xls"],
        ["pos2", "data/MLF3W2TT0117.xls"],
    ]

    truths_1 = []
    recos_1 = []
    truths_2 = []
    recos_2 = []
    for idx, (pos, file) in enumerate(files):
        tray_org = extractTrayFiducials(
            "plots/TrayFiducial.png", ToGantry=False)
        print(f"Processing {file}")

        silicon, fids_TFBF = extractProtoModuleFiducials(
            file, output_name=f"plots/ProtoModule_{idx}.png", ToGantry=False)

        tray_org.Align(fids_TFBF)

        truths_1.append(
            [tray_org.GetCenter(1)[0], tray_org.GetCenter(1)[1], tray_org.GetAngle(1)])
        truths_2.append(
            [tray_org.GetCenter(2)[0], tray_org.GetCenter(2)[1], tray_org.GetAngle(2)])
        if pos == "pos1":
            recos_1.append([silicon.GetCenter()[0],
                           silicon.GetCenter()[1], silicon.GetAngle()])
        else:
            recos_2.append([silicon.GetCenter()[0],
                           silicon.GetCenter()[1], silicon.GetAngle()])

    plot_truth_vs_recos_2plots(truths_1[0], recos_1,
                               output_name="plots/ProtoModule_comparison_pos1_2plots.png")
    plot_truth_vs_recos_2plots(truths_2[0], recos_2,
                               output_name="plots/ProtoModule_comparison_pos2_2plots.png")

    print("Pos1: fitted hexagon center:", truths_1[0])
    print("Pos1: reco hexagon center:", recos_1[0])
    print("Pos2: fitted hexagon center:", truths_2[0])
    print("Pos2: reco hexagon center:", recos_2[0])


def checkModuleFiducials():
    files = [
        ["pos1", "data/116.xlsx"],
        ["pos1", "data/116_2_04-30-2025.xlsx"],
        ["pos1", "data/116_3_04-30-2025.xlsx"],
        ["pos2", "data/117.xlsx"],
        ["pos2", "data/117_2_04-30-2025.xlsx"],
        ["pos2", "data/117_3_04-30-2025.xlsx"]
    ]

    fids_TFBF_new = {
        'TF': Fiducial(439.522, -699.113),
        'BF': Fiducial(423.441, -1091.249)
    }

    truths_1 = []
    recos_1 = []
    truths_2 = []
    recos_2 = []
    for idx, (pos, file) in enumerate(files):
        tray_org = extractTrayFiducials(
            "plots/TrayFiducial.png", ToGantry=False)
        print("tray_org", tray_org)
        print(f"Processing {file}")

        hexa_org, _ = extractModuleFiducials(
            file, output_name=f"plots/Module_{idx}.png", ToGantry=False)

        hexa = hexa_org.Align(fids_TFBF_new)

        tray = tray_org.Align(fids_TFBF_new)
        print("tray", tray)

        truths_1.append(
            [tray.GetCenter(1)[0], tray.GetCenter(1)[1], tray.GetAngle(1)])
        truths_2.append(
            [tray.GetCenter(2)[0], tray.GetCenter(2)[1], tray.GetAngle(2)])
        if pos == "pos1":
            recos_1.append([hexa.GetCenter()[0],
                           hexa.GetCenter()[1], hexa.GetAngle() - 90.0])
        else:
            recos_2.append([hexa.GetCenter()[0],
                           hexa.GetCenter()[1], hexa.GetAngle()])

    plot_truth_vs_recos_2plots(truths_1[0], recos_1,
                               output_name="plots/Module_comparison_pos1_2plots.png")
    plot_truth_vs_recos_2plots(truths_2[0], recos_2,
                               output_name="plots/Module_comparison_pos2_2plots.png")

    print("Pos1: tray center:", truths_1[0])
    print("Pos1: reco hexagon center:", recos_1[0])
    print("Pos2: tray center:", truths_2[0])
    print("Pos2: reco hexagon center:", recos_2[0])


def checkModuleFiducialsGantry(useGantry=1):
    tray_org = extractTrayFiducials(
        "plots/TrayFiducial.png", ToGantry=False)
    fids_TFBF_1 = {
        'TF': Fiducial(439.327, -699.328),
        'BF': Fiducial(423.444, -1091.469)
    }
    fids_TFBF_2 = {
        'TF': Fiducial(439.324, -699.330),
        'BF': Fiducial(423.442, -1091.474)
    }
    fids_TFBF_3 = {
        'TF': Fiducial(439.305, -699.308),
        'BF': Fiducial(423.438, -1091.442)
    }
    fids_TFBF_4 = {
        'TF': Fiducial(439.300, -699.304),
        'BF': Fiducial(423.387, -1091.438)
    }
    fids_TFBF_5 = {
        'TF': Fiducial(439.303, -699.306),
        'BF': Fiducial(423.390, -1091.439)
    }
    fids_TFBFs = [fids_TFBF_1, fids_TFBF_2,
                  fids_TFBF_3, fids_TFBF_4, fids_TFBF_5]
    if useGantry:
        hex11 = HexaFiducials(
            {"FD3": Fiducial(570.350, -728.827),
             "FD6": Fiducial(570.460, -888.831)},
            TF=Fiducial(439.324, -699.330),
            BF=Fiducial(423.442, -1091.474)
        )
        hex12 = HexaFiducials(
            {"FD3": Fiducial(570.350, -728.827),
             "FD6": Fiducial(570.460, -888.831)},
            TF=Fiducial(439.324, -699.330),
            BF=Fiducial(423.442, -1091.474)
        )
        hex13 = HexaFiducials(
            {"FD3": Fiducial(570.356, -728.821),
             "FD6": Fiducial(570.417, -888.819)},
            TF=Fiducial(439.305, -699.308),
            BF=Fiducial(423.438, -1091.442)
        )
        hex14 = HexaFiducials(
            {"FD3": Fiducial(570.357, -728.816),
             "FD6": Fiducial(570.351, -888.816)},
            TF=Fiducial(439.300, -699.304),
            BF=Fiducial(423.387, -1091.438)
        )
        hex15 = HexaFiducials(
            {"FD3": Fiducial(570.404, -728.777),
             "FD6": Fiducial(570.340, -888.797)},
            TF=Fiducial(439.303, -699.306),
            BF=Fiducial(423.390, -1091.439)
        )
        hex1s = [hex11, hex12, hex13, hex14]
        # hex1s = [hex15]
        hex21 = HexaFiducials(
            {"FD3": Fiducial(519.650, -1079.439),
             "FD6": Fiducial(519.690, -919.415)},
            TF=Fiducial(439.327, -699.328),
            BF=Fiducial(423.444, -1091.469)
        )
        hex22 = HexaFiducials(
            {"FD3": Fiducial(519.679, -1079.430),
             "FD6": Fiducial(519.660, -919.411)},
            TF=Fiducial(439.327, -699.328),
            BF=Fiducial(423.444, -1091.469)
        )
        hex23 = HexaFiducials(
            {"FD3": Fiducial(519.624, -1079.416),
             "FD6": Fiducial(519.619, -919.395)},
            TF=Fiducial(439.305, -699.308),
            BF=Fiducial(423.438, -1091.442)
        )
        hex24 = HexaFiducials(
            {"FD3": Fiducial(519.634, -1079.352),
             "FD6": Fiducial(519.532, -919.324)},
            TF=Fiducial(439.300, -699.304),
            BF=Fiducial(423.387, -1091.438)
        )
        hex25 = HexaFiducials(
            {"FD3": Fiducial(519.478, -1079.363),
             "FD6": Fiducial(519.629, -919.337)},
            TF=Fiducial(439.303, -699.306),
            BF=Fiducial(423.390, -1091.439)
        )
        hex2s = [hex21, hex22, hex23, hex24]
        # hex2s = [hex25]
    else:
        hex1 = HexaFiducials(
            {"FD3": Fiducial(132.024, 367.952),
             "FD6": Fiducial(138.450, 208.073)},
            TF=Fiducial(0, 392.421),
            BF=Fiducial(0, 0.)
        )
        hex2 = HexaFiducials(
            {"FD3": Fiducial(95.477, 15.729),
             "FD6": Fiducial(88.967, 175.588)},
            TF=Fiducial(0, 392.421),
            BF=Fiducial(0, 0)
        )
        # hex1 = hex1.Align(fids_TFBF)
        # hex2 = hex2.Align(fids_TFBF)
    tray = tray_org.Align(fids_TFBFs[-1])
    truths_1 = [
        tray.GetCenter(1)[0], tray.GetCenter(1)[1], tray.GetAngle(1)]
    truths_2 = [
        tray.GetCenter(2)[0], tray.GetCenter(2)[1], tray.GetAngle(2)]
    recos_1 = []
    recos_2 = []
    for idx, (hex1, hex2) in enumerate(zip(hex1s, hex2s)):
        hex1 = hex1.Align(fids_TFBFs[-1])
        hex2 = hex2.Align(fids_TFBFs[-1])
        recos_1.append([hex1.GetCenter(0)[0],
                        hex1.GetCenter(0)[1], hex1.GetAngle(0)])
        recos_2.append([hex2.GetCenter(0)[0],
                        hex2.GetCenter(0)[1], hex2.GetAngle(0)])
    # recos_1 = [[hex1.GetCenter(0)[0], hex1.GetCenter(0)[1], hex1.GetAngle(0)]]
    # recos_2 = [[hex2.GetCenter(0)[0], hex2.GetCenter(0)[1], hex2.GetAngle(0)]]
    plot_truth_vs_recos_2plots(truths_1, recos_1,
                               output_name="plots/Module_comparison_pos1_2plots_Gantry.png", legends=["Tray", "PCB1", "PCB2", "PCB3", "PCB4 with corrections", "PCB5 With Corrections"], colors=['red', 'orange', 'green', 'purple', 'pink', 'brown'])
    plot_truth_vs_recos_2plots(truths_2, recos_2,
                               output_name="plots/Module_comparison_pos2_2plots_Gantry.png", legends=["Tray", "PCB1", "PCB2", "PCB3", "PCB4 with corrections", "PCB5 With Corrections"], colors=['red', 'orange', 'green', 'purple', 'pink', 'brown'])
    print("Pos1: fitted hexagon center:", truths_1)
    print("Pos1: reco hexagon center:", recos_1)
    print("Pos2: fitted hexagon center:", truths_2)
    print("Pos2: reco hexagon center:", recos_2)


def checkWholeModuleFiducialsGantry():
    tray_org = extractTrayFiducials(
        "plots/TrayFiducial.png", ToGantry=False)
    fids_TFBF_5 = {
        'TF': Fiducial(439.303, -699.306),
        'BF': Fiducial(423.390, -1091.439)
    }
    fids_TFBFs = [fids_TFBF_5]

    hex1 = HexaFiducials(
        {"FD3": Fiducial(570.404, -728.777),
         "FD6": Fiducial(570.340, -888.797)},
        TF=Fiducial(439.303, -699.306),
        BF=Fiducial(423.390, -1091.439)
    )
    hex2 = HexaFiducials(
        {"FD3": Fiducial(519.478, -1079.363),
         "FD6": Fiducial(519.629, -919.337)},
        TF=Fiducial(439.303, -699.306),
        BF=Fiducial(423.390, -1091.439)
    )

    silicon1 = SiliconFiducials(
        {
            "FD2": Fiducial(487.316, -770.810),  # channel 8
            "FD1": Fiducial(487.318, -846.808),  # channel 1
            "FD3": Fiducial(653.323, -846.848),  # channel 191
            "FD4": Fiducial(653.321, -770.853),  # channel 197
        },
        TF=Fiducial(439.303, -699.301),
        BF=Fiducial(423.388, -1091.439)
    )
    silicon2 = SiliconFiducials(
        {
            "FD1": Fiducial(602.517, -961.359),  # channel 1
            "FD2": Fiducial(602.544, -1037.358),  # channel 8
            "FD3": Fiducial(436.501, -961.372),  # channel 191
            "FD4": Fiducial(436.531, -1037.373),  # channel 197
        },
        TF=Fiducial(439.303, -699.301),
        BF=Fiducial(423.388, -1091.439)
    )
    hex1 = hex1.Align(fids_TFBF_5)
    hex2 = hex2.Align(fids_TFBF_5)
    silicon1 = silicon1.Align(fids_TFBF_5)
    silicon2 = silicon2.Align(fids_TFBF_5)

    tray = tray_org.Align(fids_TFBFs[-1])
    truths_1 = [
        tray.GetCenter(1)[0], tray.GetCenter(1)[1], tray.GetAngle(1)]
    truths_2 = [
        tray.GetCenter(2)[0], tray.GetCenter(2)[1], tray.GetAngle(2)]
    recos_1 = []
    recos_2 = []
    recos_1.append(
        [silicon1.GetCenter()[0], silicon1.GetCenter()[1], silicon1.GetAngle()])
    recos_2.append(
        [silicon2.GetCenter()[0], silicon2.GetCenter()[1], silicon2.GetAngle()])
    recos_1.append([hex1.GetCenter(0)[0],
                    hex1.GetCenter(0)[1], hex1.GetAngle(0)])
    recos_2.append([hex2.GetCenter(0)[0],
                    hex2.GetCenter(0)[1], hex2.GetAngle(0)])
    plot_truth_vs_recos_2plots(truths_1, recos_1,
                               output_name="plots/WholeModule_comparison_pos1_2plots_Gantry.png", legends=["Tray", "Sensor", "PCB"])
    plot_truth_vs_recos_2plots(truths_2, recos_2,
                               output_name="plots/WholeModule_comparison_pos2_2plots_Gantry.png", legends=["Tray", "Sensor", "PCB"])
    print("Pos1: fitted hexagon center:", truths_1)
    print("Pos1: reco hexagon center:", recos_1)
    print("Pos2: fitted hexagon center:", truths_2)
    print("Pos2: reco hexagon center:", recos_2)


def checkProtoModuleFiducialsGantry(useGantry=1):
    tray_org = extractTrayFiducials(
        "plots/TrayFiducial.png", ToGantry=False)
    fids_TFBF_1 = {
        'TF': Fiducial(439.303, -699.301),
        'BF': Fiducial(423.388, -1091.439)
    }
    fids_TFBFs = [fids_TFBF_1]
    if useGantry:
        silicon1 = SiliconFiducials(
            {
                "FD2": Fiducial(487.316, -770.810),  # channel 8
                "FD1": Fiducial(487.318, -846.808),  # channel 1
                "FD3": Fiducial(653.323, -846.848),  # channel 191
                "FD4": Fiducial(653.321, -770.853),  # channel 197
            },
            TF=Fiducial(439.303, -699.301),
            BF=Fiducial(423.388, -1091.439)
        )
        silicon2 = SiliconFiducials(
            {
                "FD1": Fiducial(602.517, -961.359),  # channel 1
                "FD2": Fiducial(602.544, -1037.358),  # channel 8
                "FD3": Fiducial(436.501, -961.372),  # channel 191
                "FD4": Fiducial(436.531, -1037.373),  # channel 197
            },
            TF=Fiducial(439.303, -699.301),
            BF=Fiducial(423.388, -1091.439)
        )
        silicon1s = [silicon1]
        silicon2s = [silicon2]
    else:
        hex1 = HexaFiducials(
            {"FD3": Fiducial(132.024, 367.952),
             "FD6": Fiducial(138.450, 208.073)},
            TF=Fiducial(0, 392.421),
            BF=Fiducial(0, 0.)
        )
        hex2 = HexaFiducials(
            {"FD3": Fiducial(95.477, 15.729),
             "FD6": Fiducial(88.967, 175.588)},
            TF=Fiducial(0, 392.421),
            BF=Fiducial(0, 0)
        )
        # hex1 = hex1.Align(fids_TFBF)
        # hex2 = hex2.Align(fids_TFBF)
    tray = tray_org.Align(fids_TFBFs[-1])
    truths_1 = [
        tray.GetCenter(1)[0], tray.GetCenter(1)[1], tray.GetAngle(1)]
    truths_2 = [
        tray.GetCenter(2)[0], tray.GetCenter(2)[1], tray.GetAngle(2)]
    recos_1 = []
    recos_2 = []
    for idx, (silicon1, silicon2) in enumerate(zip(silicon1s, silicon2s)):
        silicon1 = silicon1.Align(fids_TFBFs[-1])
        silicon2 = silicon2.Align(fids_TFBFs[-1])
        recos_1.append([silicon1.GetCenter()[0],
                        silicon1.GetCenter()[1], silicon1.GetAngle()])
        recos_2.append([silicon2.GetCenter()[0],
                        silicon2.GetCenter()[1], silicon2.GetAngle()])
    # recos_1 = [[hex1.GetCenter(0)[0], hex1.GetCenter(0)[1], hex1.GetAngle(0)]]
    # recos_2 = [[hex2.GetCenter(0)[0], hex2.GetCenter(0)[1], hex2.GetAngle(0)]]
    plot_truth_vs_recos_2plots(truths_1, recos_1,
                               output_name="plots/ProtoModule_comparison_pos1_2plots_Gantry.png")
    plot_truth_vs_recos_2plots(truths_2, recos_2,
                               output_name="plots/ProtoModule_comparison_pos2_2plots_Gantry.png")
    print("Pos1: fitted hexagon center:", truths_1)
    print("Pos1: reco hexagon center:", recos_1)
    print("Pos2: fitted hexagon center:", truths_2)
    print("Pos2: reco hexagon center:", recos_2)


if __name__ == "__main__":
    # compareBPFiducials()
    # compareTray2HexaEdges(doSilicon=False)
    # compareHexaFid2HexaEdges()
    # sys.exit(0)
    # tray_pos1, hexEdge1, tray_pos2, hexEdge2 = compareTray2HexaEdges(
    #    doSilicon=False)
    # hex_fids_pos1, hexEdge1_2, hex_fids_pos2, hexEdge2_2 = compareHexaFid2HexaEdges()
    # hexEdge1_2 = hexEdge1_2[0]
    # hexEdge2_2 = hexEdge2_2[0]
    # print("tray_pos1", tray_pos1)
    # print("hexEdge1", hexEdge1)
    # print("tray_pos2", tray_pos2)
    # print("hexEdge2", hexEdge2)
    # print("hex_fids_pos1", hex_fids_pos1)
    # print("hexEdge1_2", hexEdge1_2)
    # print("hex_fids_pos2", hex_fids_pos2)
    # print("hexEdge2_2", hexEdge2_2)
    # hex_fids_pos1_new = addValues(
    #    hexEdge1[0], subtractValues(hex_fids_pos1[0], hexEdge1_2))
    # hex_fids_pos2_new = addValues(
    #    hexEdge2[0], subtractValues(hex_fids_pos2[0], hexEdge2_2))
    # plot_truth_vs_recos_2plots(tray_pos1[0], hexEdge1 + [hex_fids_pos1_new],
    #                           output_name="plots/hexa_tray_edge_fids_pos1_2plots.png")
    # plot_truth_vs_recos_2plots(tray_pos2[0], hexEdge2 + [hex_fids_pos2_new],
    #                           output_name="plots/hexa_tray_edge_fids_pos2_2plots.png")
    # checkProtoModuleFiducials()
    checkModuleFiducialsGantry()
    # checkProtoModuleFiducialsGantry()
    checkWholeModuleFiducialsGantry()
    # checkModuleFiducials()
