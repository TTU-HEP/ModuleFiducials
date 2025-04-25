import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib
import numpy as np
from modules.helpers import Angle, GetCenterAndAngle, MeanAndRMS  # noqa
from scipy.optimize import minimize


matplotlib.use('Agg')


class Fiducial(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __str__(self):
        return f"Fiducial(X={self.X}, Y={self.Y})"

    def __repr__(self):
        return self.__str__()

    def __item__(self):
        return self.X, self.Y

    def __getitem__(self, index):
        if index == 0:
            return self.X
        elif index == 1:
            return self.Y
        else:
            raise IndexError("Index out of range. Use 0 for X and 1 for Y.")

    def XY(self):
        return self.X, self.Y

    def GetX(self):
        return self.X

    def GetY(self):
        return self.Y

    def Rotate(self, X0, Y0, angle):
        # Rotate the fiducial points around the center (X0, Y0) by the given angle
        X_rotated = (self.X - X0) * np.cos(angle) - \
            (self.Y - Y0) * np.sin(angle) + X0
        Y_rotated = (self.X - X0) * np.sin(angle) + \
            (self.Y - Y0) * np.cos(angle) + Y0
        return Fiducial(X_rotated, Y_rotated)


def FidAngle(fid1, fid2):
    # Calculate the angle between two fiducials
    x1, y1 = fid1.XY()
    x2, y2 = fid2.XY()
    return Angle(x1, y1, x2, y2)


class AssemblyTrayFiducials(object):
    def __init__(self, fiducials):
        for fiducial in fiducials.values():
            assert isinstance(fiducial, Fiducial), \
                "All fiducials must be instances of the Fiducial class"
        assert fiducials.keys() == {'TF', 'BF', 'CP1', 'OP1', 'CP2', 'OP2'} or \
            fiducials.keys() == {'TF', 'BF', 'CP1', 'OP1', 'CP2', 'OP2', 'F3'}, \
            "Fiducial must have 6 or 7 elements (TF, BF, CP1, OP1, CP2, OP2, [F3])"
        self.fiducials = fiducials

    def __str__(self):
        return f"AssemblyTrayFiducials({self.fiducials})"

    def __repr__(self):
        return self.__str__()

    def GetCenter(self, pos=1):
        assert pos in [1, 2], "pos must be 1 or 2"
        return self.fiducials['CP' + str(pos)].XY()

    def GetAngle(self, pos=1):
        assert pos in [1, 2], "pos must be 1 or 2"
        return FidAngle(self.fiducials['CP' + str(pos)], self.fiducials['OP' + str(pos)])

    def visualize(self, output_name):
        plt.figure(figsize=(8, 8))
        for point, fiducial in self.fiducials.items():
            x, y = fiducial.XY()
            plt.scatter(x, y)
            plt.text(x, y, point, fontsize=12,
                     ha='right', va='bottom', color='black')

        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("Assembly Tray Fiducials")
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(output_name)
        plt.close()
        return

    def Align(self, fids_TF_BF):
        """
        rotate the fiducials to the new TF and BF positions
        """
        assert fids_TF_BF.keys() == {'TF', 'BF'}, \
            "fids_TF_BF must have 2 elements (TF, BF)"
        assert isinstance(fids_TF_BF['TF'], Fiducial) and isinstance(fids_TF_BF['BF'], Fiducial), \
            "TF and BF must be instances of the Fiducial class"

        fids_new = {
            'TF': fids_TF_BF['TF'],
            'BF': fids_TF_BF['BF'],
            'CP1': self.fiducials['CP1'],
            'OP1': self.fiducials['OP1'],
            'CP2': self.fiducials['CP2'],
            'OP2': self.fiducials['OP2']
        }

        base = 'BF'
        target = 'TF'

        shift_X = fids_TF_BF[base].GetX() - self.fiducials[base].GetX()
        shift_Y = fids_TF_BF[base].GetY() - self.fiducials[base].GetY()
        angle_current = FidAngle(self.fiducials[base], self.fiducials[target])
        angle_new = FidAngle(fids_TF_BF[base], fids_TF_BF[target])
        angle_shift = angle_new - angle_current
        angle_shift = np.deg2rad(angle_shift)  # Convert to radians
        # print("fid TF_BF: TF", fids_TF_BF['TF'])
        # print("fid TF_BF: BF", fids_TF_BF['BF'])
        # print("fid current: TF", self.fiducials['TF'])
        # print("fid current: BF", self.fiducials['BF'])
        # print("angle current:", angle_current)
        # print("angle new:", angle_new)
        # print("angle shift:", angle_shift)
        for pos, fiducial in self.fiducials.items():
            if pos == 'TF' or pos == 'BF':
                continue
            # print("fiducials before rotation for pos", pos, ":", fiducial)
            fids_new[pos] = fiducial.Rotate(
                self.fiducials[base].X, self.fiducials[base].Y, angle_shift)
            # print("fiducials after rotation for pos", pos, ":", fids_new[pos])
            fids_new[pos].X += shift_X
            fids_new[pos].Y += shift_Y

        return AssemblyTrayFiducials(fids_new)


class HexaEdgeFiducials(object):
    def __init__(self, fiducials):
        for fiducial in fiducials:
            assert isinstance(fiducial, Fiducial), \
                "All fiducials must be instances of the Fiducial class"
        self.fiducials = fiducials

    def __str__(self):
        return f"HexaEdgeFiducials({self.fiducials})"

    def __repr__(self):
        return self.__str__()

    def XYPoints(self):
        return np.array([fiducial.XY() for fiducial in self.fiducials])

    def visualize(self, output_name):
        plt.figure(figsize=(8, 8))
        for fiducial in self.fiducials:
            x, y = fiducial.XY()
            plt.scatter(x, y)

        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("HexaEdge Fiducials")
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(output_name)
        plt.close()
        return


def generate_hexagon_vertices(center, radius, angle):
    cx, cy = center
    return np.array([
        (cx + radius * np.cos(angle + i * np.pi / 3),
         cy + radius * np.sin(angle + i * np.pi / 3))
        for i in range(6)
    ])


def point_to_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    projection = a + t * ab
    return np.linalg.norm(p - projection)


def hexagon_fit_objective(params, points, target_radius):
    x0, y0, r, theta = params
    hex_vertices = generate_hexagon_vertices((x0, y0), r, theta)

    dist_sum = 0
    for p in points:
        dists = [point_to_segment_distance(
            p, hex_vertices[i], hex_vertices[(i+1) % 6]) for i in range(6)]
        dist_sum += min(dists)**2

    # Radius regularization toward target
    radius_penalty = (r - target_radius)**2 * 1000
    return dist_sum + radius_penalty


def fit_hexagon_with_radius_constraint(hexagon: HexaEdgeFiducials, target_radius=95.0):
    points = hexagon.XYPoints()
    centroid = np.mean(points, axis=0)
    r0 = target_radius
    theta0 = 0.0

    result = minimize(
        hexagon_fit_objective,
        x0=[centroid[0], centroid[1], r0, theta0],
        args=(points, target_radius),
        bounds=[(None, None), (None, None), (1e-3, None), (None, None)],
        options={"maxiter": 1000}
    )

    x0, y0, r, theta = result.x
    fitted_vertices = generate_hexagon_vertices((x0, y0), r, theta)

    # print("Fitting result:")
    # print(
    #    f"Center: ({x0:.3f}, {y0:.3f}), Radius: {r:.3f}, Angle: {np.degrees(theta):.3f}°")
    # print(f"Success: {result.success}, Fun: {result.fun:.2f}")
    # print(f"Original points: {points}")
    # print(f"Fitted vertices: {fitted_vertices}")

    return {
        "center": (x0, y0),
        "radius": r,
        "theta": theta,
        "vertices": fitted_vertices,
        "success": result.success,
        "fun": result.fun,
        "original_points": points
    }


def plot_fitted_hexagon(result, output_name="fitted_hexagon.png"):
    vertices = result['vertices']
    points = result['original_points']
    center = result['center']

    plt.figure(figsize=(6, 6))
    plt.plot(*np.append(vertices, [vertices[0]],
             axis=0).T, label='Fitted Hexagon', linewidth=2)
    plt.scatter(points[:, 0], points[:, 1], c='red', label='Input Points')
    plt.scatter(*center, color='blue', marker='x', label='Fitted Center')
    plt.gca().set_aspect('equal')
    plt.title(f"Fitted Radius: {result['radius']:.3f}")
    plt.legend(
        title=f"Center: ({center[0]:.3f}, {center[1]:.3f}), Angle: {np.degrees(result['theta']):.3f}°")
    plt.grid(True)
    plt.savefig(output_name)


def find_angle_to_rightmost_side_midpoint(center, radius, theta, rightmost=True):
    # Step 1: Generate vertices
    vertices = generate_hexagon_vertices(center, radius, theta)

    # Step 2: Compute midpoints of each edge
    midpoints = np.array([
        (vertices[i] + vertices[(i+1) % 6]) / 2 for i in range(6)
    ])

    # Step 3: Find midpoint with largest x-coordinate (rightmost)
    # If rightmost is False, find the leftmost midpoin
    if rightmost:
        max_x_index = np.argmax(midpoints[:, 0])
    else:
        max_x_index = np.argmin(midpoints[:, 0])
    rightmost_midpoint = midpoints[max_x_index]

    # Step 4: Vector from center to that midpoint
    vec = rightmost_midpoint - np.array(center)

    # Step 5: Compute angle from center to midpoint
    angle = np.arctan2(vec[1], vec[0])  # returns angle in radians
    angle = np.degrees(angle)  # convert to degrees
    # if angle > 90.0:
    #    angle -= 180.0
    # if angle < -90.0:
    #    angle += 180.0
    return angle, rightmost_midpoint
