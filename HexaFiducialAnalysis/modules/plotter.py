import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


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
