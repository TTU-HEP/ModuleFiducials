from modules.components import Fiducial
from modules.utils import readJsonFile, LoadTray
from modules.plotter import plot_truth_vs_recos_2plots, make_accuracy_plot


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

        tray = tray.Align(tray_base)
        component_pos1 = component_pos1.Align(tray_base)
        component_pos2 = component_pos2.Align(tray_base)

        if isProto:
            recos_pos1.append([component_pos1.GetCenter()[0], component_pos1.GetCenter()
                               [1], component_pos1.GetAngle()])
            recos_pos2.append([component_pos2.GetCenter()[0], component_pos2.GetCenter()
                               [1], component_pos2.GetAngle()])
        else:
            recos_pos1.append([component_pos1.GetCenter(0)[0], component_pos1.GetCenter(0)
                               [1], component_pos1.GetAngle(0)])
            recos_pos2.append([component_pos2.GetCenter(0)[0], component_pos2.GetCenter(0)
                               [1], component_pos2.GetAngle(0)])

    print("truths_pos1", truths_pos1)
    print("recos_pos1", recos_pos1)
    print("truths_pos2", truths_pos2)
    print("recos_pos2", recos_pos2)

    legends = ["Tray", "PCB1", "PCB2", "PCB3", "PCB4", "PCB5", "PCB6"]
    colors = ['gray', 'green', 'red', 'orange', 'purple', 'pink', 'brown']
    plot_truth_vs_recos_2plots(truths_pos1, recos_pos1,
                               output_name=f"{outputname}_pos1_Gantry.png", legends=legends, colors=colors)
    plot_truth_vs_recos_2plots(truths_pos2, recos_pos2,
                               output_name=f"{outputname}_pos2_Gantry.png", legends=legends, colors=colors)


def checkWholeModules(f_proto, f_module):
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

    make_accuracy_plot(recos_pos1[0][0] - truths_pos1[0],
                       recos_pos1[0][1] - truths_pos1[1],
                       recos_pos1[1][0] - truths_pos1[0],
                       recos_pos1[1][1] - truths_pos1[1],
                       recos_pos1[0][2] - truths_pos1[2],
                       recos_pos1[1][2] - truths_pos1[2])


if __name__ == "__main__":
    f_modules = [
        "jsondata/modules_dryrun1.json",
        "jsondata/modules_dryrun2.json",
    ]
    checkModules(f_modules)
    f_modules = [
        "jsondata/protomodules_320MLF3TCTT0208_320MLF3TCTT0209.json"
    ]
    checkModules(f_modules, isProto=True,
                 outputname="plots/ProtoModule_208_209_comparison")

    f_proto = "jsondata/protomodules_320MLF3TCTT0206_320MLF3TCTT0207.json"
    f_module = "jsondata/modules_320MLF3TCTT0206_320MLF3TCTT0207.json"
    checkWholeModules(f_proto, f_module)
