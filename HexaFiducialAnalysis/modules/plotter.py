import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator


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


def plot_truth_vs_recos_2plots(truth, recos, output_name="plots/hexagon_comparison.png", useOddEven=False, colors=None, legends=None, xyrange=150, markers=None):
    tx, ty, t_angle = truth
    t_angle_rad = np.radians(t_angle)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_main = axes[0]
    ax_angle = axes[1]

    # --- First plot: points only ---
    label = None
    if legends != None:
        label = legends[0]
    marker = 'o'
    if markers and len(markers) > 0:
        marker = markers[0]
    ax_main.scatter(0, 0, color='blue', marker=marker, s=100,
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
        marker = 'h'
        if markers and len(markers) > i + 1:
            marker = markers[i + 1]
        ax_main.scatter((rx - tx)*1e3, (ry - ty)*1e3, color=col, marker=marker, s=100,
                        edgecolors=col, label=label)

    ax_main.grid(True)
    ax_main.set_xlim(-xyrange, xyrange)
    ax_main.set_ylim(-xyrange, xyrange)
    # draw box of 100 x 100
    # fill with shadow
    ax_main.fill_between([-100, 100], -100, 100, color='blue', alpha=0.08)
    # draw box of 50 x 50
    # fill with quarter shadow
    ax_main.fill_between([-50, 50], -50, 50, color='blue', alpha=0.15)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X [um]')
    ax_main.set_ylabel('Y [um]')
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
    angle_plus3 = np.deg2rad(0.03)     # convert 0.03 deg to rad
    length = 10  # same as your quiver scale

    # Compute three points: center, left boundary, right boundary
    left_x = origin_x + length * np.cos(angle_center + angle_plus)
    left_x2 = origin_x + length * np.cos(angle_center + angle_plus2)
    left_x3 = origin_x + length * np.cos(angle_center + angle_plus3)
    left_y = origin_y + length * np.sin(angle_center + angle_plus)
    left_y2 = origin_y + length * np.sin(angle_center + angle_plus2)
    left_y3 = origin_y + length * np.sin(angle_center + angle_plus3)

    right_x = origin_x + length * np.cos(angle_center - angle_plus)
    right_x2 = origin_x + length * np.cos(angle_center - angle_plus2)
    right_x3 = origin_x + length * np.cos(angle_center - angle_plus3)
    right_y = origin_y + length * np.sin(angle_center - angle_plus)
    right_y2 = origin_y + length * np.sin(angle_center - angle_plus2)
    right_y3 = origin_y + length * np.sin(angle_center - angle_plus3)

    # Create the polygon for the cone
    cone = Polygon([[origin_x, origin_y],
                    [left_x, left_y],
                    [right_x, right_y]],
                   closed=True, color='blue', alpha=0.15, label='±0.01° cone')
    cone2 = Polygon([[origin_x, origin_y],
                     [left_x2, left_y2],
                     [right_x2, right_y2]],
                    closed=True, color='blue', alpha=0.08, label='±0.02° cone')
    cone3 = Polygon([[origin_x, origin_y],
                     [left_x3, left_y3],
                     [right_x3, right_y3]],
                    closed=True, color='blue', alpha=0.04, label='±0.03° cone')

    # Add it to the plot
    ax_angle.add_patch(cone)
    ax_angle.add_patch(cone2)
    # ax_angle.add_patch(cone3)

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


def make_accuracy_plot(
        rel_sensor_X=0.,
        rel_sensor_Y=0.,
        rel_pcb_X=0.,
        rel_pcb_Y=0.,
        rel_sensor_angle=0.,
        rel_pcb_angle=0.
):
    # Need at least matplotlib 3.5 version to use the plt.subplots(layout)
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.set_box_aspect(1)

    #################################
    #          Offset part          #
    #################################
    rel_sensor_X = np.array([rel_sensor_X * 1000])
    rel_sensor_Y = np.array([rel_sensor_Y * 1000])
    rel_pcb_X = np.array([rel_pcb_X * 1000])
    rel_pcb_Y = np.array([rel_pcb_Y * 1000])

    ax.plot(rel_sensor_X, rel_sensor_Y, marker='o', markerfacecolor='#ff7f0e',
            markeredgecolor='#ff7f0e', linestyle='None', label='Sensor w.r.t. Tray')
    ax.plot(rel_pcb_X,    rel_pcb_Y,    marker='h', markerfacecolor='#2ca02c',
            markeredgecolor='#2ca02c', linestyle='None', label='PCB w.r.t. Tray')
    ax.plot(np.array([0.]), np.array([0.]), marker='o', markerfacecolor='k',
            markeredgecolor='k', linestyle='None', label='Tray')

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
    #ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
    #          loc='lower right', ncol=2, borderaxespad=0.)
    ax.legend(loc='upper left')

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

    plt.savefig(f'plots/wholemodule_accuracy.png')
    plt.close()
