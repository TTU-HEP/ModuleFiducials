from modules.components import Fiducial
from modules.utils import readJsonFile, LoadTray
from modules.plotter import plot_truth_vs_recos_2plots, make_accuracy_plot
from modules.helpers import CleanAngle
from modules.components import SiliconFiducials, HexaFiducials


f_tray = "jsondata/tray.json"
tray_org = LoadTray(f_tray)

tray_base = {
    'TF': Fiducial(439.303, -699.306),
    'BF': Fiducial(423.390, -1091.439)
}


def checkModules(f_modules, isProto=False, outputname="plots/Module_comparison"):
    tray = tray_org.Align(tray_base)
    truths_pos1 = [tray.GetCenter(1)[0], tray.GetCenter(1)[
        1], tray.GetAngle(1)]
    truths_pos2 = [tray.GetCenter(2)[0], tray.GetCenter(2)[
        1], tray.GetAngle(2)]
    recos_pos1 = []
    recos_pos2 = []
    for f_module in f_modules:
        _, component_pos1, component_pos2 = readJsonFile(
            f_module, isProto=isProto)

        components = []
        if component_pos1 is not None:
            components.append(component_pos1)
        if component_pos2 is not None:
            components.append(component_pos2)

        tray = tray.Align(tray_base)

        for component in components:
            component = component.Align(tray_base)

            recos = recos_pos1 if component.isPos1 else recos_pos2
            if isProto:
                recos.append([component.GetCenter()[0], component.GetCenter()
                              [1], component.GetAngle()])
            else:
                use4FDs = 0
                recos.append([component.GetCenter(use4FDs)[0], component.GetCenter(use4FDs)
                              [1], component.GetAngle(use4FDs)])

    print("truths_pos1", truths_pos1)
    print("recos_pos1", recos_pos1)
    print("truths_pos2", truths_pos2)
    print("recos_pos2", recos_pos2)

    legends = ["Tray", "PCB1", "PCB2", "PCB3", "PCB4", "PCB5", "PCB6", "PCB7"]
    colors = ['gray', 'green', 'red', 'orange',
              'purple', 'pink', 'brown', 'blue', 'yellow']
    if len(recos_pos1) > 0:
        plot_truth_vs_recos_2plots(truths_pos1, recos_pos1,
                                   output_name=f"{outputname}_pos1_Gantry.png", legends=legends, colors=colors)
    if len(recos_pos2) > 0:
        plot_truth_vs_recos_2plots(truths_pos2, recos_pos2,
                                   output_name=f"{outputname}_pos2_Gantry.png", legends=legends, colors=colors)

    return recos_pos1, recos_pos2


def checkWholeModules(f_proto, f_module, suffix_pos1="", suffix_pos2=""):
    _, sil_pos1, sil_pos2 = readJsonFile(f_proto, isProto=True)
    _, hex_pos1, hex_pos2 = readJsonFile(f_module, isProto=False)

    components = []
    for component in [sil_pos1, sil_pos2, hex_pos1, hex_pos2]:
        if component is not None:
            components.append(component)

    # align everything to base
    tray = tray_org.Align(tray_base)

    truths_pos1 = [tray.GetCenter(1)[0], tray.GetCenter(1)[
        1], tray.GetAngle(1)]
    truths_pos2 = [tray.GetCenter(2)[0], tray.GetCenter(2)[
        1], tray.GetAngle(2)]

    recos_pos1 = []
    recos_pos2 = []
    for component in components:
        component = component.Align(tray_base)
        recos = recos_pos1 if component.isPos1 else recos_pos2
        if isinstance(component, SiliconFiducials):
            recos.append([component.GetCenter()[0], component.GetCenter()
                          [1], component.GetAngle()])
        else:
            recos.append([component.GetCenter(0)[0], component.GetCenter(0)
                          [1], component.GetAngle(0)])

    legends = ["Tray", "Silicon", "PCB"]
    colors = ['gray', 'green', 'red', 'orange', 'purple', 'pink', 'brown']

    diffs_sensor_tray = []
    diffs_pcb_tray = []

    if len(recos_pos1) > 0:
        plot_truth_vs_recos_2plots(truths_pos1, recos_pos1,
                                   output_name=f"plots/WholeModule_pos1_Gantry_{suffix_pos1}.png", legends=legends, colors=colors, module_id=suffix_pos1)
        diffs_sensor_tray.append([recos_pos1[0][0] - truths_pos1[0],
                                 recos_pos1[0][1] - truths_pos1[1],
                                 CleanAngle(recos_pos1[0][2] - truths_pos1[2])])
        diffs_pcb_tray.append([recos_pos1[1][0] - truths_pos1[0],
                              recos_pos1[1][1] - truths_pos1[1],
                              CleanAngle(recos_pos1[1][2] - truths_pos1[2])])
    if len(recos_pos2) > 0:
        plot_truth_vs_recos_2plots(truths_pos2, recos_pos2,
                                   output_name=f"plots/WholeModule_pos2_Gantry_{suffix_pos2}.png", legends=legends, colors=colors, module_id=suffix_pos2)
        diffs_sensor_tray.append([recos_pos2[0][0] - truths_pos2[0],
                                 recos_pos2[0][1] - truths_pos2[1],
                                 CleanAngle(recos_pos2[0][2] - truths_pos2[2])])
        diffs_pcb_tray.append([recos_pos2[1][0] - truths_pos2[0],
                              recos_pos2[1][1] - truths_pos2[1],
                              CleanAngle(recos_pos2[1][2] - truths_pos2[2])])

    print("suffix ", suffix)
    print("diffs_sensor_tray ", diffs_sensor_tray)
    print("diffs_pcb_tray ", diffs_pcb_tray)
    make_accuracy_plot(diffs_sensor_tray, diffs_pcb_tray, suffix)

    return diffs_sensor_tray, diffs_pcb_tray


if __name__ == "__main__":
    f_modules = [
        "jsondata/modules_320MLF3TCTT0210_320MLF3TCTT0211_dryrun1.json",
        "jsondata/modules_320MLF3TCTT0210_320MLF3TCTT0211_dryrun2.json",
        "jsondata/modules_320MLF3TCTT0210_320MLF3TCTT0211_dryrun3.json",
        "jsondata/modules_320MLF3TCTT0210_320MLF3TCTT0211_dryrun4.json",
        "jsondata/modules_320MLF3TCTT0210_320MLF3TCTT0211_dryrun5.json",
        "jsondata/modules_320MLF3TCTT0210_320MLF3TCTT0211_dryrun6.json",
        "jsondata/modules_320MLF3TCTT0210_320MLF3TCTT0211_dryrun7.json",
    ]
    f_modules = [
        "jsondata/modules_320MLF3TCTT0210.json",
        "jsondata/modules_320MLF3TCTT0210_OGP.json"
    ]
    checkModules(f_modules)
    # f_modules = [
    #    "jsondata/protomodules_320MLF3TCTT0208_320MLF3TCTT0209.json"
    # ]
    # checkModules(f_modules, isProto=True,
    #             outputname="plots/ProtoModule_208_209_comparison")

    data_modules = [
        ["jsondata/modules_320MLF3TCTT0210.json",
         "jsondata/protomodules_320MLF3TCTT0210.json", "X_0210"],
        ["jsondata/modules_320MLF3TCTT0208_320MLF3TCTT0209_OGP.json",
         "jsondata/protomodules_320MLF3TCTT0208_320MLF3TCTT0209.json", "0208_0209"],
        ["jsondata/modules_320MLF3TCTT0206_320MLF3TCTT0207.json",
         "jsondata/protomodules_320MLF3TCTT0206_320MLF3TCTT0207.json", "0206_0207"],
        ["jsondata/modules_320MLF3T2TT0105_320MLF3T2TT0106.json",
         "jsondata/protomodules_320MLF3T2TT0105_320MLF3T2TT0106.json", "0105_0106"],
        ["jsondata/modules_320MLF3T2TT0103_320MLF3T2TT0104.json",
         "jsondata/protomodules_320MLF3T2TT0103_320MLF3T2TT0104.json", "0103_0104"],
        ["jsondata/modules_320MLF3T2TT0101_320MLF3T2TT0102.json",
         "jsondata/protomodules_320MLF3T2TT0101_320MLF3T2TT0102.json", "0101_0102"],
    ]

    diffs_sensor_tray = []
    diffs_pcb_tray = []

    for f_module, f_proto, suffix in data_modules:
        suffix_pos1 = suffix.split("_")[0]
        suffix_pos2 = suffix.split("_")[1]
        diff_sensor_tray, diff_pcb_tray = checkWholeModules(
            f_proto, f_module, suffix_pos1=suffix_pos1, suffix_pos2=suffix_pos2)

        diffs_sensor_tray += diff_sensor_tray
        diffs_pcb_tray += diff_pcb_tray

    make_accuracy_plot(diffs_sensor_tray, diffs_pcb_tray, "summary")
