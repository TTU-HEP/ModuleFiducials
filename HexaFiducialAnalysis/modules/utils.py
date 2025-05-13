import json
from modules.components import Fiducial, HexaFiducials, SiliconFiducials, AssemblyTrayFiducials


def LoadTray(f_input):
    with open(f_input, "r") as f:
        data_protos = json.load(f)

    assert "tray" in data_protos, "No tray information found in json"

    tray_info = data_protos["tray"]
    tray = {}
    for key, value in tray_info.items():
        tray[key] = Fiducial(value[0], value[1])

    tray = AssemblyTrayFiducials(tray)

    return tray


def readJsonFile(f_input, isProto=False):
    with open(f_input, "r") as f:
        data_protos = json.load(f)

    assert "tray" in data_protos, "No tray information found in json"

    tray_info = data_protos["tray"]
    tray = {}
    for key, value in tray_info.items():
        tray[key] = Fiducial(value[0], value[1])

    if isProto:
        className = SiliconFiducials
    else:
        className = HexaFiducials

    if "pos1" in data_protos:
        pos1_info = data_protos["pos1"]
        pos1 = {}
        for key, value in pos1_info.items():
            pos1[key] = Fiducial(value[0], value[1])

        hex_pos1 = className(pos1, TF=tray["TF"], BF=tray["BF"])

    if "pos2" in data_protos:
        pos2_info = data_protos["pos2"]
        pos2 = {}
        for key, value in pos2_info.items():
            pos2[key] = Fiducial(value[0], value[1])

        hex_pos2 = className(pos2, TF=tray["TF"], BF=tray["BF"])

    return tray, hex_pos1, hex_pos2
