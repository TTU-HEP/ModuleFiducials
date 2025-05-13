from modules.components import Fiducial
from modules.utils import readJsonFile, LoadTray
from modules.plotter import plot_truth_vs_recos_2plots


f_tray = "jsondata/tray.json"
tray_org = LoadTray(f_tray)

tray_base = {
    'TF': Fiducial(439.303, -699.306),
    'BF': Fiducial(423.390, -1091.439)
}


def checkModules(f_modules):
    tray = tray_org.Align(tray_base)
    truths_pos1 = [tray.GetCenter(1)[0], tray.GetCenter(1)[
        1], tray.GetAngle(1)]
    truths_pos2 = [tray.GetCenter(2)[0], tray.GetCenter(2)[
        1], tray.GetAngle(2)]
    recos_pos1 = []
    recos_pos2 = []
    for f_module in f_modules:
        _, hex_pos1, hex_pos2 = readJsonFile(f_module, isProto=False)

        tray = tray.Align(tray_base)
        hex_pos1 = hex_pos1.Align(tray_base)
        hex_pos2 = hex_pos2.Align(tray_base)

        recos_pos1.append([hex_pos1.GetCenter(0)[0], hex_pos1.GetCenter(0)
                           [1], hex_pos1.GetAngle(0)])
        recos_pos2.append([hex_pos2.GetCenter(0)[0], hex_pos2.GetCenter(0)
                           [1], hex_pos2.GetAngle(0)])

    print("truths_pos1", truths_pos1)
    print("recos_pos1", recos_pos1)
    print("truths_pos2", truths_pos2)
    print("recos_pos2", recos_pos2)

    legends = ["Tray", "PCB1", "PCB2", "PCB3", "PCB4", "PCB5", "PCB6"]
    colors = ['gray', 'green', 'red', 'orange', 'purple', 'pink', 'brown']
    plot_truth_vs_recos_2plots(truths_pos1, recos_pos1,
                               output_name="plots/Module_comparison_pos1_Gantry.png", legends=legends, colors=colors)
    plot_truth_vs_recos_2plots(truths_pos2, recos_pos2,
                               output_name="plots/Module_comparison_pos2_Gantry.png", legends=legends, colors=colors)


def checkWholeModules(f_proto, f_module):
    f_proto = "jsondata/protomodules.json"
    f_module = "jsondata/modules.json"

    _, sil_pos1, sil_pos2 = readJsonFile(f_proto, isProto=True)
    _, hex_pos1, hex_pos2 = readJsonFile(f_module, isProto=False)

    # align everything to base
    tray = tray_org.Align(tray_base)
    sil_pos1 = sil_pos1.Align(tray_base)
    sil_pos2 = sil_pos2.Align(tray_base)
    hex_pos1 = hex_pos1.Align(tray_base)
    hex_pos2 = hex_pos2.Align(tray_base)

    truths_pos1 = [tray.GetCenter(1)[0], tray.GetCenter(1)[
        1], tray.GetAngle(1)]
    truths_pos2 = [tray.GetCenter(2)[0], tray.GetCenter(2)[
        1], tray.GetAngle(2)]

    recos_pos1 = []
    recos_pos2 = []
    recos_pos1.append([sil_pos1.GetCenter()[0], sil_pos1.GetCenter()
                       [1], sil_pos1.GetAngle()])
    recos_pos2.append([sil_pos2.GetCenter()[0], sil_pos2.GetCenter()
                       [1], sil_pos2.GetAngle()])

    recos_pos1.append([hex_pos1.GetCenter(0)[0], hex_pos1.GetCenter(0)
                      [1], hex_pos1.GetAngle(0)])
    recos_pos2.append([hex_pos2.GetCenter(0)[0], hex_pos2.GetCenter(0)
                      [1], hex_pos2.GetAngle(0)])

    print("truths_pos1", truths_pos1)
    print("recos_pos1", recos_pos1)
    print("truths_pos2", truths_pos2)
    print("recos_pos2", recos_pos2)

    legends = ["Tray", "Silicon", "PCB"]
    colors = ['gray', 'green', 'red', 'orange', 'purple', 'pink', 'brown']

    plot_truth_vs_recos_2plots(truths_pos1, recos_pos1,
                               output_name="plots/WholeModule_comparison_pos1_Gantry.png", legends=legends, colors=colors)
    plot_truth_vs_recos_2plots(truths_pos2, recos_pos2,
                               output_name="plots/WholeModule_comparison_pos2_Gantry.png", legends=legends, colors=colors)


if __name__ == "__main__":
    f_modules = [
        "jsondata/modules_dryrun1.json",
        "jsondata/modules_dryrun2.json",
    ]
    checkModules(f_modules)

    f_proto = "jsondata/protomodules.json"
    f_module = "jsondata/modules.json"
    checkWholeModules(f_proto, f_module)
