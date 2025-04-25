from modules.components import HexaEdgeFiducials, Fiducial, fit_hexagon_with_radius_constraint, plot_fitted_hexagon, AssemblyTrayFiducials, find_angle_to_rightmost_side_midpoint
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt


def extractHexaFiducials(file_name, sheet_name='WorkSheet_01', output_name='HexaFiducial.png', target_radius=96.4):
    xls = pd.ExcelFile(file_name)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_name = 'Unnamed: 6'

    # pos 1
    f1X = float(df[col_name][8])
    f1Y = float(df[col_name][9])
    f1 = Fiducial(f1X, f1Y)
    f2X = float(df[col_name][14])
    f2Y = float(df[col_name][15])
    f2 = Fiducial(f2X, f2Y)
    f3X = float(df[col_name][20])
    f3Y = float(df[col_name][21])
    f3 = Fiducial(f3X, f3Y)
    f4X = float(df[col_name][26])
    f4Y = float(df[col_name][27])
    f4 = Fiducial(f4X, f4Y)
    f5X = float(df[col_name][32])
    f5Y = float(df[col_name][33])
    f5 = Fiducial(f5X, f5Y)
    f6X = float(df[col_name][38])
    f6Y = float(df[col_name][39])
    f6 = Fiducial(f6X, f6Y)
    f7X = float(df[col_name][44])
    f7Y = float(df[col_name][45])
    f7 = Fiducial(f7X, f7Y)
    f8X = float(df[col_name][50])
    f8Y = float(df[col_name][51])
    f8 = Fiducial(f8X, f8Y)
    f9X = float(df[col_name][56])
    f9Y = float(df[col_name][57])
    f9 = Fiducial(f9X, f9Y)
    f10X = float(df[col_name][62])
    f10Y = float(df[col_name][63])
    f10 = Fiducial(f10X, f10Y)
    f11X = float(df[col_name][68])
    f11Y = float(df[col_name][69])
    f11 = Fiducial(f11X, f11Y)

    fiducials_pos1_hexaboard = HexaEdgeFiducials(
        [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11])
    fiducials_pos1_hexaboard.visualize(output_name=output_name.replace(
        ".png", "_pos1.png"))

    results = fit_hexagon_with_radius_constraint(
        fiducials_pos1_hexaboard, target_radius)
    plot_fitted_hexagon(results, output_name.replace(
        ".png", "_fitted_hexagon_pos1.png"))
    fiducials_pos1_hexaboard.fitted_hexagon = results

    # pos 2
    f1X = float(df[col_name][74])
    f1Y = float(df[col_name][75])
    f1 = Fiducial(f1X, f1Y)
    f2X = float(df[col_name][80])
    f2Y = float(df[col_name][81])
    f2 = Fiducial(f2X, f2Y)
    f3X = float(df[col_name][86])
    f3Y = float(df[col_name][87])
    f3 = Fiducial(f3X, f3Y)
    f4X = float(df[col_name][92])
    f4Y = float(df[col_name][93])
    f4 = Fiducial(f4X, f4Y)
    f5X = float(df[col_name][98])
    f5Y = float(df[col_name][99])
    f5 = Fiducial(f5X, f5Y)
    f6X = float(df[col_name][104])
    f6Y = float(df[col_name][105])
    f6 = Fiducial(f6X, f6Y)
    f7X = float(df[col_name][110])
    f7Y = float(df[col_name][111])
    f7 = Fiducial(f7X, f7Y)
    f8X = float(df[col_name][116])
    f8Y = float(df[col_name][117])
    f8 = Fiducial(f8X, f8Y)
    f9X = float(df[col_name][122])
    f9Y = float(df[col_name][123])
    f9 = Fiducial(f9X, f9Y)
    f10X = float(df[col_name][128])
    f10Y = float(df[col_name][129])
    f10 = Fiducial(f10X, f10Y)
    f11X = float(df[col_name][134])
    f11Y = float(df[col_name][135])
    f11 = Fiducial(f11X, f11Y)

    fiducials_pos2_hexaboard = HexaEdgeFiducials(
        [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11])
    fiducials_pos2_hexaboard.visualize(output_name=output_name.replace(
        ".png", "_pos2.png"))
    results = fit_hexagon_with_radius_constraint(
        fiducials_pos2_hexaboard, target_radius)
    plot_fitted_hexagon(results, output_name.replace(
        ".png", "_fitted_hexagon_pos2.png"))
    fiducials_pos2_hexaboard.fitted_hexagon = results

    # extra TF and BF fiducials for alignment
    TFX = float(df[col_name][141])
    TFY = float(df[col_name][142])
    TF = Fiducial(TFX, TFY)
    BFX = float(df[col_name][157])
    BFY = float(df[col_name][158])
    BF = Fiducial(BFX, BFY)
    fids_TFBF = {
        'TF': TF,
        'BF': BF
    }

    return fiducials_pos1_hexaboard, fiducials_pos2_hexaboard, fids_TFBF


def extractTrayFiducials(output_name='TrayFiducial.png'):
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
    return fiducials


if __name__ == "__main__":
    files = [
        "data/Hex_Position_wPUT_DryRun1.xls",
        "data/Hex_Position_wPUT_DryRun2.xls",
        "data/Hex_Position_wPUT_DryRun1_04_24_2025.xls",
        "data/Hex_Position_wPUT_DryRun2_04_24_2025.xls",
        "data/Hex_Position_wPUT_DryRun3_04_24_2025.xls",
        "data/Hex_Position_wPUT_DryRun1_afterchanges_04_24_2025.xls",
        "data/Hex_Position_wPUT_DryRun2_afterchanges_04_24_2025.xls"
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

    for idx, file in enumerate(files):
        tray_org = extractTrayFiducials("plots/TrayFiducial.png")
        print(f"Processing {file}")
        hex1, hex2, fids_TFBF = extractHexaFiducials(
            file, output_name=f"plots/test_{idx}.png", target_radius=96.05)

        angle1 = find_angle_to_rightmost_side_midpoint(
            hex1.fitted_hexagon['center'], hex1.fitted_hexagon['radius'], hex1.fitted_hexagon['theta'], False)
        angle2 = find_angle_to_rightmost_side_midpoint(
            hex2.fitted_hexagon['center'], hex2.fitted_hexagon['radius'], hex2.fitted_hexagon['theta'], True)
        print("Angle 1:", angle1[0])
        print("Angle 2:", angle2[0])

        tray = tray_org.Align(fids_TFBF)
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

    def plot_truth_vs_recos(truth, recos, line_length=20, output_name="plots/hexagon_comparison.png"):
        tx, ty, t_angle = truth
        t_angle = np.radians(t_angle)
        t_end = [tx + line_length *
                 np.cos(t_angle), ty + line_length * np.sin(t_angle)]

        plt.figure(figsize=(5, 5))

        # Plot truth line and point
        plt.plot([tx, t_end[0]], [ty, t_end[1]], 'b-',
                 label='Truth Line', linewidth=2)
        plt.scatter(tx, ty, color='blue', marker='o', s=100,
                    edgecolors='black', label='Truth Point')

        for i, (rx, ry, r_angle) in enumerate(recos):
            r_angle = np.radians(r_angle)
            r_end = [rx + line_length *
                     np.cos(r_angle), ry + line_length * np.sin(r_angle)]

            # Plot reco line and point
            if i >= 5:
                # running with fixes
                plt.plot([rx, r_end[0]], [ry, r_end[1]], 'g--',
                         label=f'Reco Line #{i}', linewidth=2)
                plt.scatter(rx, ry, color='green', marker='^', s=100,
                            edgecolors='black', label='Reco Point')
            else:
                plt.plot([rx, r_end[0]], [ry, r_end[1]], 'r--',
                         label=f'Reco Line #{i}', linewidth=2)
                plt.scatter(rx, ry, color='red', marker='^', s=100,
                            edgecolors='black', label='Reco Point')

            # plt.gca().set_aspect('equal')
            # plt.legend(loc='best')
            plt.grid(True)
            plt.xlabel('X')
            plt.ylabel('Y')
            # plt.tight_layout()
        plt.savefig(output_name)

    plot_truth_vs_recos(truths_1[0], recos_1, line_length=0.5,
                        output_name="plots/hexagon_comparison_pos1.png")
    plot_truth_vs_recos(truths_2[0], recos_2, line_length=0.5,
                        output_name="plots/hexagon_comparison_pos2.png")
