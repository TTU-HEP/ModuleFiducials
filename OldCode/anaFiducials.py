import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('Agg')
from modules.helpers import MeanAndRMS, Angle, GetCenterAndAngle  # noqa
from modules.components import HexaFiducials, BPFiducial, SiliconFiducial, TrayFiducial  # noqa


def extraInfo(file_path="NewMeasurements/320MLF3W2TT0114_addinfo2.xls", pos=1):
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name='WorkSheet_01')

    col_name = 'Unnamed: 6'

    # hexaboards
    if pos == 1:
        FD3_X = float(df[col_name][31])
        FD3_Y = float(df[col_name][32])
        FD6_X = float(df[col_name][37])
        FD6_Y = float(df[col_name][38])
        FD1_X = float(df[col_name][43])
        FD1_Y = float(df[col_name][44])
        FD5_X = float(df[col_name][50])
        FD5_Y = float(df[col_name][51])
        FD2_X = float(df[col_name][57])
        FD2_Y = float(df[col_name][58])
        FD4_X = float(df[col_name][64])
        FD4_Y = float(df[col_name][65])
        center_X = float(df[col_name][123])
        center_Y = float(df[col_name][124])

        # baseplates
        BP3_X = float(df[col_name][71])
        BP3_Y = float(df[col_name][72])
        BP4_X = float(df[col_name][80])
        BP4_Y = float(df[col_name][81])
        BP5_X = float(df[col_name][87])
        BP5_Y = float(df[col_name][88])
        BP6_X = float(df[col_name][94])
        BP6_Y = float(df[col_name][95])
        BP7_X = float(df[col_name][101])
        BP7_Y = float(df[col_name][102])
        BP8_X = float(df[col_name][108])
        BP8_Y = float(df[col_name][109])

        # silicon
        SiC_X = float(df[col_name][115])
        SiC_Y = float(df[col_name][116])
    else:
        FD3_X = float(df[col_name][25])
        FD3_Y = float(df[col_name][26])
        FD6_X = float(df[col_name][31])
        FD6_Y = float(df[col_name][32])
        FD1_X = float(df[col_name][37])
        FD1_Y = float(df[col_name][38])
        FD5_X = float(df[col_name][44])
        FD5_Y = float(df[col_name][45])
        FD2_X = float(df[col_name][51])
        FD2_Y = float(df[col_name][52])
        FD4_X = float(df[col_name][58])
        FD4_Y = float(df[col_name][59])
        center_X = float(df[col_name][124])
        center_Y = float(df[col_name][125])

        # baseplates
        BP3_X = float(df[col_name][72])
        BP3_Y = float(df[col_name][73])
        BP4_X = float(df[col_name][81])
        BP4_Y = float(df[col_name][82])
        BP5_X = float(df[col_name][88])
        BP5_Y = float(df[col_name][89])
        BP6_X = float(df[col_name][95])
        BP6_Y = float(df[col_name][96])
        BP7_X = float(df[col_name][102])
        BP7_Y = float(df[col_name][103])
        BP8_X = float(df[col_name][109])
        BP8_Y = float(df[col_name][110])

        # silicon
        SiC_X = float(df[col_name][116])
        SiC_Y = float(df[col_name][117])

    Fiducials_PCB = HexaFiducials(FD3_X, FD3_Y, FD6_X, FD6_Y, FD1_X, FD1_Y,
                                  FD2_X, FD2_Y, FD4_X, FD4_Y, FD5_X, FD5_Y,
                                  center_X, center_Y)
    Fiducials_BP = BPFiducial(BP3_X, BP3_Y, BP4_X, BP4_Y, BP5_X, BP5_Y,
                              BP6_X, BP6_Y, BP7_X, BP7_Y, BP8_X, BP8_Y)
    Fiducials_BP.visualize("BPFiducials.png", xlim=(0, 300), ylim=(0, 400))

    Fiducials_SiC = SiliconFiducial(SiC_X, SiC_Y)

    return Fiducials_PCB, Fiducials_BP, Fiducials_SiC


def extractTrayInfo():
    file_path = "/Users/yfeng/Desktop/TTU/OGP_ttu/GantryTrayFiducialData_repeated.xlsx"

    xls = pd.ExcelFile(file_path)
    sheetnames = [
        name for name in xls.sheet_names if name.startswith('WorkSheet')]

    fiducials_tray = []
    for sheet_name in sheetnames:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # print(f"Sheet name: {sheet_name}")
        # print(df.head())

        col_name = 'Unnamed: 4'
        tray_Z = float(df[col_name][8])
        OP1_X = float(df[col_name][11])
        OP1_Y = float(df[col_name][12])

        CP1_X = float(df[col_name][15])
        CP1_Y = float(df[col_name][16])

        OP2_X = float(df[col_name][19])
        OP2_Y = float(df[col_name][20])

        CP2_X = float(df[col_name][23])
        CP2_Y = float(df[col_name][24])

        Fiducials_tray = TrayFiducial(tray_Z,
                                      OP1_X, OP1_Y, CP1_X, CP1_Y, OP2_X, OP2_Y, CP2_X, CP2_Y)
        fiducials_tray.append(Fiducials_tray)

    return fiducials_tray


if __name__ == "__main__":
    for (pos, mod) in [(1, "114"), (2, "115")]:
        measurements = [
            f"NewMeasurements/320MLF3W2TT0{mod}_addinfo1.xls",
            f"NewMeasurements/320MLF3W2TT0{mod}_addinfo2.xls",
            f"NewMeasurements/320MLF3W2TT0{mod}_addinfo3.xls",
            f"NewMeasurements/320MLF3W2TT0{mod}_addinfo4.xls",
            f"NewMeasurements/320MLF3W2TT0{mod}_addinfo5.xls",
        ]

        fiducials_PCBs = []
        fiducials_BPs = []
        fiducials_SiCs = []
        for measurement in measurements:
            fiducials_PCB, fiducials_BP, fiducials_SiC = extraInfo(
                measurement, pos)
            fiducials_PCBs.append(fiducials_PCB)
            fiducials_BPs.append(fiducials_BP)
            fiducials_SiCs.append(fiducials_SiC)

        fiducials_trays = extractTrayInfo()

        trays_Xs = []
        trays_Ys = []
        trays_angles = []
        for fiducials_tray in fiducials_trays:
            center_X, center_Y = fiducials_tray.GetCenter(pos)
            angle = fiducials_tray.GetAngle(pos)

            trays_Xs.append(center_X)
            trays_Ys.append(center_Y)
            trays_angles.append(angle)

        mean_trays_X = MeanAndRMS(trays_Xs)
        mean_trays_Y = MeanAndRMS(trays_Ys)
        mean_trays_angle = MeanAndRMS(trays_angles)
        print("Mean and RMS of trays:")
        print("X:", mean_trays_X)
        print("Y:", mean_trays_Y)
        print("Angle:", mean_trays_angle)

        BP_Xs = []
        BP_Ys = []
        BP_angles = []
        for fiducials_BP in fiducials_BPs:
            center_X, center_Y = fiducials_BP.GetCenter()
            BP_Xs.append(center_X)
            BP_Ys.append(center_Y)
            angle = fiducials_BP.GetAngle()
            BP_angles.append(angle)

        print("Mean and RMS of baseplates:")
        print("X:", MeanAndRMS(BP_Xs))
        print("Y:", MeanAndRMS(BP_Ys))
        print("Angle:", MeanAndRMS(BP_angles))

        Hexa_4_Xs = []
        Hexa_4_Ys = []
        Hexa_4_angles = []
        Hexa_2_Xs = []
        Hexa_2_Ys = []
        Hexa_2_angles = []
        Hexa_Xs = []
        Hexa_Ys = []
        for fiducials_PCB in fiducials_PCBs:
            center_X, center_Y, angle = fiducials_PCB.GetCenter(4)
            Hexa_4_Xs.append(center_X)
            Hexa_4_Ys.append(center_Y)
            Hexa_4_angles.append(angle)
            center_2_X, center_2_Y, angle = fiducials_PCB.GetCenter(2)
            Hexa_2_Xs.append(center_2_X)
            Hexa_2_Ys.append(center_2_Y)
            Hexa_2_angles.append(angle)
            center_X, center_Y, angle = fiducials_PCB.GetCenter(1)
            Hexa_Xs.append(center_X)
            Hexa_Ys.append(center_Y)

        print("Mean and RMS of hexaboards:")
        print("Using 4 fiducials:")
        print("X:", MeanAndRMS(Hexa_4_Xs))
        print("Y:", MeanAndRMS(Hexa_4_Ys))
        print("Angle:", MeanAndRMS(Hexa_4_angles))
        print("Using 2 fiducials:")
        print("X:", MeanAndRMS(Hexa_2_Xs))
        print("Y:", MeanAndRMS(Hexa_2_Ys))
        print("Angle:", MeanAndRMS(Hexa_2_angles))
        print("Using 1 fiducial:")
        print("X:", MeanAndRMS(Hexa_Xs))
        print("Y:", MeanAndRMS(Hexa_Ys))

        # silicon
        SiC_Xs = []
        SiC_Ys = []
        for fiducials_SiC in fiducials_SiCs:
            center_X, center_Y = fiducials_SiC.GetCenter()
            SiC_Xs.append(center_X)
            SiC_Ys.append(center_Y)

        print("Mean and RMS of silicon:")
        print("X:", MeanAndRMS(SiC_Xs))
        print("Y:", MeanAndRMS(SiC_Ys))
