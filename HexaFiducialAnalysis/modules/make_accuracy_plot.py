####################################################################################
#
#  Filename    : EvtGenNtuplizer.cc
#  Description : Make an accuracy plot with offsets and angles of HGCAL
#                module components w.r.t. baseplate components
#  Author      : You-Ying Li [ you-ying.li@cern.ch ]
#
####################################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def make_accuracy_plot(
        module_name='M01',
        rel_sensor_X=0.,
        rel_sensor_Y=0.,
        rel_pcb_X=0.,
        rel_pcb_Y=0.,
        rel_sensor_angle=0.,
        rel_pcb_angle=0.
):
    """
    rel_sensor_X : relative X of sensor w.r.t. baseplate [unit : mm]
    rel_sensor_Y : relative Y of sensor w.r.t. baseplate [unit : mm]
    rel_pcb_X    : relative X of pcb w.r.t. baseplate    [unit : mm]
    rel_pcb_Y    : relative Y of pcb w.r.t. baseplate    [unit : mm]
    rel_sensor_angle : relative angle of sensor w.r.t. baseplate [unit : degree]
    rel_pcb_angle    : relative angle of pcb w.r.t. baseplate    [unit : degree]
    """

    # Need at least matplotlib 3.5 version to use the plt.subplots(layout)
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.set_box_aspect(1)
    ax.set_title(f'{module_name} accuracy plot', y=1.15, fontsize=20)

    #################################
    #          Offset part          #
    #################################
    rel_sensor_X = np.array([rel_sensor_X * 1000])
    rel_sensor_Y = np.array([rel_sensor_Y * 1000])
    rel_pcb_X = np.array([rel_pcb_X * 1000])
    rel_pcb_Y = np.array([rel_pcb_Y * 1000])

    ax.plot(rel_sensor_X, rel_sensor_Y, marker='o', markerfacecolor='#ff7f0e',
            markeredgecolor='#ff7f0e', linestyle='None', label='Sensor w.r.t. Baseplate')
    ax.plot(rel_pcb_X,    rel_pcb_Y,    marker='o', markerfacecolor='#2ca02c',
            markeredgecolor='#2ca02c', linestyle='None', label='PCB w.r.t. Baseplate')
    ax.plot(np.array([0.]), np.array([0.]), marker='o', markerfacecolor='k',
            markeredgecolor='k', linestyle='None', label='Baseplate')

    ax.set_xlabel('$\Delta x$ [$\mu m$]',  fontsize=18)
    ax.set_ylabel('$\Delta y$ [$\mu m$]', fontsize=18)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    ax.set_xlim(-200, 300)
    ax.set_ylim(-200, 300)
    ax.vlines(-50, -50, 50, colors='b')
    ax.vlines(50, -50, 50, colors='b')
    ax.hlines(-50, -50, 50, colors='b')
    ax.hlines(50, -50, 50, colors='b')
    ax.text(-50, 55, '50 $\mu m$', color='b', fontsize=12)
    ax.vlines(-100, -100, 100, colors='r')
    ax.vlines(100, -100, 100, colors='r')
    ax.hlines(-100, -100, 100, colors='r')
    ax.hlines(100, -100, 100, colors='r')
    ax.text(-100, 105, '100 $\mu m$', color='r', fontsize=12)
    ax.vlines(0, -200, 300, colors='k')
    ax.hlines(0, -200, 300, colors='k')

    plt.tick_params(axis='both', which='minor', direction='in',
                    labelsize=0, length=5, width=1, right=True, top=True)
    plt.tick_params(axis='both', which='major', direction='in',
                    labelsize=18, length=7, width=1.5, right=True, top=True)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
              loc='lower right', ncol=2, borderaxespad=0.)

    #################################
    #      Rotation angle part      #
    #################################
    ax_sub = fig.add_axes([.52, .58, .42, .25], polar=True)

    gauge_angle_max = 0.4
    gauge_angle_unit = 0.1
    orig_gauge_angle_max = 40
    transfer_factor = 40. / gauge_angle_max
    orig_gauge_angle_unit = transfer_factor * gauge_angle_unit

    ax_sub.set_rmax(2)
    ax_sub.get_yaxis().set_visible(False)
    ax_sub.grid(False)

    ax_sub.set_theta_offset(np.pi/2)
    ax_sub.set_thetamin(-orig_gauge_angle_max)
    ax_sub.set_thetamax(orig_gauge_angle_max)
    ax_sub.set_rorigin(-2.5)

    tick = [ax_sub.get_rmax(), ax_sub.get_rmax()*0.97]
    for t in np.deg2rad(np.arange(0, 360, orig_gauge_angle_unit*0.5)):
        ax_sub.plot([t, t], tick, lw=0.72, color="k")

    tick = [ax_sub.get_rmax(), ax_sub.get_rmax()*0.9]
    for t in np.deg2rad(np.arange(0, 360, orig_gauge_angle_unit)):
        ax_sub.plot([t, t], tick, lw=0.72, color="k")

    ax_sub.set_thetagrids(np.arange(
        orig_gauge_angle_max, -orig_gauge_angle_max-orig_gauge_angle_unit, -orig_gauge_angle_unit))
    ax_sub.set_xticklabels(np.round(np.arange(
        gauge_angle_max, -gauge_angle_max-gauge_angle_unit, -gauge_angle_unit), decimals=2))

    ax_sub.annotate('', xy=(transfer_factor * rel_sensor_angle * np.pi / 180., 2),
                    xytext=(0., -2.5),
                    arrowprops=dict(color='#ff7f0e',
                                    arrowstyle="->"),
                    )
    ax_sub.annotate('', xy=(transfer_factor * rel_pcb_angle * np.pi / 180., 1.6),
                    xytext=(0., -2.5),
                    arrowprops=dict(color='#2ca02c',
                                    arrowstyle="->"),
                    )

    ax_sub.annotate('', xy=(transfer_factor * 0. * np.pi / 180., 2),
                    xytext=(0., -2.5),
                    arrowprops=dict(color='k',
                                    arrowstyle="->",
                                    ),
                    )
    ax_sub.annotate('', xy=(transfer_factor * 0.2 * np.pi / 180., 2),
                    xytext=(transfer_factor * 0.2 * np.pi / 180., 0.),
                    arrowprops=dict(color='b',
                                    arrowstyle="-",
                                    linestyle="dotted"
                                    ),
                    )
    ax_sub.annotate('', xy=(transfer_factor * -0.2 * np.pi / 180., 2),
                    xytext=(transfer_factor * -0.2 * np.pi / 180., 0.),
                    arrowprops=dict(color='b',
                                    arrowstyle="-",
                                    linestyle="dotted"
                                    ),
                    )
    ax_sub.annotate('', xy=(transfer_factor * 0.4 * np.pi / 180., 2),
                    xytext=(transfer_factor * 0.4 * np.pi / 180., 0.),
                    arrowprops=dict(color='r',
                                    arrowstyle="-",
                                    linestyle="dotted"
                                    ),
                    )
    ax_sub.annotate('', xy=(transfer_factor * -0.4 * np.pi / 180., 2),
                    xytext=(transfer_factor * -0.4 * np.pi / 180., 0.),
                    arrowprops=dict(color='r',
                                    arrowstyle="-",
                                    linestyle="dotted"
                                    ),
                    )

    plt.savefig(f'{module_name}_accuracy.png')
    plt.close()


if __name__ == "__main__":

    """
    rel_sensor_X : relative X of sensor w.r.t. baseplate [unit : mm]
    rel_sensor_Y : relative Y of sensor w.r.t. baseplate [unit : mm]
    rel_pcb_X    : relative X of pcb w.r.t. baseplate    [unit : mm]
    rel_pcb_Y    : relative Y of pcb w.r.t. baseplate    [unit : mm]
    rel_sensor_angle : relative angle of sensor w.r.t. baseplate [unit : degree]
    rel_pcb_angle    : relative angle of pcb w.r.t. baseplate    [unit : degree]
    """

    """make_accuracy_plot(
        module_name = 'M56',
        rel_sensor_X = -0.02163,
        rel_sensor_Y = 0.0469,
        rel_pcb_X = -0.04258,
        rel_pcb_Y = 0.04145,
        rel_sensor_angle = 0.3377,
        rel_pcb_angle = 0.3325
    )"""
