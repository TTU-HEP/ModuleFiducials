import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib
import numpy as np
from modules.helpers import Angle, GetCenterAndAngle, MeanAndRMS  # noqa
from scipy.optimize import minimize


matplotlib.use('Agg')


def AlignTFBF(fiducial, fids_TF_BF_current, fids_TF_BF_new, base="BF"):
    """
    rotate and shift the fiducials to the new TF and BF positions
    """
    assert fids_TF_BF_current.keys() == {'TF', 'BF'} and fids_TF_BF_new.keys() == {'TF', 'BF'}, \
        "fids_TF_BF must have 2 elements (TF, BF)"
    assert isinstance(fids_TF_BF_new['TF'], Fiducial) and isinstance(fids_TF_BF_new['BF'], Fiducial) and isinstance(fids_TF_BF_current['TF'], Fiducial) and isinstance(fids_TF_BF_current['BF'], Fiducial),  \
        "TF and BF must be instances of the Fiducial class"

    # use TF/BF to shift the other fiducials
    # use TF and BF angle to rotate the other fiducials
    target = 'TF' if base == 'BF' else 'BF'

    shift_X = fids_TF_BF_new[base].GetX() - fids_TF_BF_current[base].GetX()
    shift_Y = fids_TF_BF_new[base].GetY() - fids_TF_BF_current[base].GetY()
    angle_current = FidAngle(
        fids_TF_BF_current[base], fids_TF_BF_current[target])
    angle_new = FidAngle(fids_TF_BF_new[base], fids_TF_BF_new[target])
    angle_shift = angle_new - angle_current
    angle_shift = np.deg2rad(angle_shift)  # Convert to radians

    # print("fiducials before rotation for pos", pos, ":", fiducial)
    fiducial_new = fiducial.Rotate(fids_TF_BF_current[base], angle_shift)
    # print("fiducials after rotation for pos", pos, ":", fiducial_new)
    fiducial_new.X += shift_X
    fiducial_new.Y += shift_Y
    return fiducial_new


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

    def RotateXY(self, X0, Y0, angle):
        # Rotate the fiducial points around the center (X0, Y0) by the given angle
        X_rotated = (self.X - X0) * np.cos(angle) - \
            (self.Y - Y0) * np.sin(angle) + X0
        Y_rotated = (self.X - X0) * np.sin(angle) + \
            (self.Y - Y0) * np.cos(angle) + Y0
        return Fiducial(X_rotated, Y_rotated)

    def Rotate(self, fiducial, angle):
        return self.RotateXY(fiducial.X, fiducial.Y, angle)

    def FlipY(self):
        """
        OGP and gantry has opposite y directions
        """
        return Fiducial(self.X, -self.Y)


def FidAngle(fid1, fid2):
    # Calculate the angle between two fiducials
    x1, y1 = fid1.XY()
    x2, y2 = fid2.XY()
    return Angle(x1, y1, x2, y2)


class AssemblyTrayFiducials(object):
    def __init__(self, fiducials, isOGP=True):
        for fiducial in fiducials.values():
            assert isinstance(fiducial, Fiducial), \
                "All fiducials must be instances of the Fiducial class"
        assert fiducials.keys() == {'TF', 'BF', 'CP1', 'OP1', 'CP2', 'OP2'} or \
            fiducials.keys() == {'TF', 'BF', 'CP1', 'OP1', 'CP2', 'OP2', 'F3'}, \
            "Fiducial must have 6 or 7 elements (TF, BF, CP1, OP1, CP2, OP2, [F3])"
        self.fiducials = fiducials
        self.isOGP = isOGP

    def __str__(self):
        return f"AssemblyTrayFiducials({self.fiducials})"

    def __repr__(self):
        return self.__str__()

    def IsOGPCoord(self):
        # Gantry and OGP have opposite y directions
        # OGP is the default
        return self.isOGP

    def IsGantryCoord(self):
        return not self.isOGP

    def ToGantry(self):
        if self.IsOGPCoord():
            fiducials_new = self.fiducials.copy()
            print("Converting to Gantry coordinates")
            for key, fiducial in self.fiducials.items():
                fiducials_new[key] = fiducial.FlipY()
            self.fiducials = fiducials_new
            self.isOGP = False
        return self

    def ToOGP(self):
        if self.IsGantryCoord():
            print("Converting to OGP coordinates")
            fiducials_new = self.fiducials.copy()
            for key, fiducial in self.fiducials.items():
                fiducials_new[key] = fiducial.FlipY()
            self.fiducials = fiducials_new
            self.isOGP = True
        return self

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

    def Align(self, fids_TF_BF, base="BF"):
        """
        rotate the fiducials to the new TF and BF positions
        """
        fids_TF_BF_current = {}
        fids_TF_BF_current['TF'] = self.fiducials['TF']
        fids_TF_BF_current['BF'] = self.fiducials['BF']

        fids_new = self.fiducials.copy()

        for pos, fiducial in self.fiducials.items():
            fids_new[pos] = AlignTFBF(
                fiducial, fids_TF_BF_current, fids_TF_BF, base=base)

        return AssemblyTrayFiducials(fids_new)


class HexaFiducials(object):
    """
    Use the four fiducials to define the center and angle of the hexagon 
    (FD1, FD2, FD4, FD5)
    or the two fiducials (FD3, FD6)
    """

    def __init__(self, fiducials: dict, isOGP=True, TF=None, BF=None):
        for fiducial in fiducials.values():
            assert isinstance(fiducial, Fiducial), \
                "All fiducials must be instances of the Fiducial class"
        self.fiducials = fiducials
        self.TF = TF
        self.BF = BF
        self.isOGP = isOGP

    def __str__(self):
        return f"HexaFiducials({self.fiducials})"

    def __repr__(self):
        return self.__str__()

    def IsOGPCoord(self):
        # Gantry and OGP have opposite y directions
        # OGP is the default
        return self.isOGP

    def IsGantryCoord(self):
        return not self.isOGP

    def ToGantry(self):
        if self.IsOGPCoord():
            print("Converting to Gantry coordinates")
            fiducials_new = self.fiducials.copy()
            for name, fid in self.fiducials.items():
                fiducials_new[name] = fid.FlipY()
            self.fiducials = fiducials_new
            self.isOGP = False
        return self

    def ToOGP(self):
        if self.IsGantryCoord():
            print("Converting to OGP coordinates")
            fiducials_new = self.fiducials.copy()
            for name, fid in self.fiducials.items():
                fiducials_new[name] = fid.FlipY()
            self.isOGP = True
        return self

    def Align(self, fids_TF_BF, base="BF"):
        fids_TF_BF_current = {}
        fids_TF_BF_current['TF'] = self.TF
        fids_TF_BF_current['BF'] = self.BF
        fids_new = self.fiducials.copy()
        for name, fiducial in self.fiducials.items():
            fids_new[name] = AlignTFBF(
                fiducial, fids_TF_BF_current, fids_TF_BF, base=base)
        return HexaFiducials(fids_new, isOGP=self.isOGP, TF=fids_TF_BF['TF'], BF=fids_TF_BF['BF'])

    def XYPoints(self):
        return np.array([fiducial.XY() for fiducial in self.fiducials.values()])

    def GetCenter(self, use4FDs=True):
        if use4FDs:
            for fd in ["FD1", "FD2", "FD4", "FD5"]:
                assert fd in self.fiducials, f"{fd} not in fiducials"

            x1, y1 = self.fiducials["FD1"].XY()
            x2, y2 = self.fiducials["FD2"].XY()
            x4, y4 = self.fiducials["FD4"].XY()
            x5, y5 = self.fiducials["FD5"].XY()
            x = (x1 + x2 + x4 + x5) / 4
            y = (y1 + y2 + y4 + y5) / 4
        else:
            for fd in ["FD3", "FD6"]:
                assert fd in self.fiducials, f"{fd} not in fiducials"
            x3, y3 = self.fiducials["FD3"].XY()
            x6, y6 = self.fiducials["FD6"].XY()
            x = (x3 + x6) / 2
            y = (y3 + y6) / 2
        return x, y

    def GetAngle(self, use4FDs=True):
        if use4FDs:
            for fd in ["FD1", "FD2", "FD4", "FD5"]:
                assert fd in self.fiducials, f"{fd} not in fiducials"
            avg12X = (self.fiducials["FD1"].GetX() +
                      self.fiducials["FD2"].GetX()) / 2
            avg12Y = (self.fiducials["FD1"].GetY() +
                      self.fiducials["FD2"].GetY()) / 2
            avg45X = (self.fiducials["FD4"].GetX() +
                      self.fiducials["FD5"].GetX()) / 2
            avg45Y = (self.fiducials["FD4"].GetY() +
                      self.fiducials["FD5"].GetY()) / 2
            # Calculate the angle between the two average points
            angle = Angle(avg12X, avg12Y, avg45X, avg45Y) + 180.0
            return angle
        else:
            for fd in ["FD3", "FD6"]:
                assert fd in self.fiducials, f"{fd} not in fiducials"
            x3, y3 = self.fiducials["FD3"].XY()
            x6, y6 = self.fiducials["FD6"].XY()
            angle = Angle(x3, y3, x6, y6) - 90.0
            return angle

    def visualize(self, output_name, includeTFBF=False):
        plt.figure(figsize=(8, 8))
        for name, fiducial in self.fiducials.items():
            x, y = fiducial.XY()
            plt.scatter(x, y, label=name)
            plt.text(x, y, name, fontsize=12,
                     ha='right', va='bottom', color='black')

        if includeTFBF:
            if self.TF is not None:
                x, y = self.TF.XY()
                plt.scatter(x, y, color='red', label='TF')
                plt.text(x, y, 'TF', fontsize=12,
                         ha='right', va='bottom', color='red')
            if self.BF is not None:
                x, y = self.BF.XY()
                plt.scatter(x, y, color='blue', label='BF')
                plt.text(x, y, 'BF', fontsize=12,
                         ha='right', va='bottom', color='blue')
            plt.legend()
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("Hexa Fiducials")
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(output_name)
        plt.close()
        return


class SiliconFiducials(object):
    """
    Use the four fiducials to define the center and angle of the hexagon 
    (FD1, FD2, FD3, FD4)
    """

    def __init__(self, fiducials: dict, isOGP=True, TF=None, BF=None):
        for fiducial in fiducials.values():
            assert isinstance(fiducial, Fiducial), \
                "All fiducials must be instances of the Fiducial class"
        self.fiducials = fiducials
        self.TF = TF
        self.BF = BF
        self.isOGP = isOGP

    def __str__(self):
        return f"SiliconFiducials({self.fiducials})"

    def __repr__(self):
        return self.__str__()

    def IsOGPCoord(self):
        # Gantry and OGP have opposite y directions
        # OGP is the default
        return self.isOGP

    def IsGantryCoord(self):
        return not self.isOGP

    def ToGantry(self):
        if self.IsOGPCoord():
            print("Converting to Gantry coordinates")
            fiducials_new = self.fiducials.copy()
            for name, fid in self.fiducials.items():
                fiducials_new[name] = fid.FlipY()
            self.fiducials = fiducials_new
            self.isOGP = False
        return self

    def ToOGP(self):
        if self.IsGantryCoord():
            print("Converting to OGP coordinates")
            fiducials_new = self.fiducials.copy()
            for name, fid in self.fiducials.items():
                fiducials_new[name] = fid.FlipY()
            self.isOGP = True
        return self

    def Align(self, fids_TF_BF, base="BF"):
        fids_TF_BF_current = {}
        fids_TF_BF_current['TF'] = self.TF
        fids_TF_BF_current['BF'] = self.BF
        fids_new = self.fiducials.copy()
        for name, fiducial in self.fiducials.items():
            fids_new[name] = AlignTFBF(
                fiducial, fids_TF_BF_current, fids_TF_BF, base=base)
        return HexaFiducials(fids_new, isOGP=self.isOGP, TF=fids_TF_BF['TF'], BF=fids_TF_BF['BF'])

    def XYPoints(self):
        return np.array([fiducial.XY() for fiducial in self.fiducials.values()])

    def GetCenter(self):
        for fd in ["FD1", "FD2", "FD3", "FD4"]:
            assert fd in self.fiducials, f"{fd} not in fiducials"

        x1, y1 = self.fiducials["FD1"].XY()
        x2, y2 = self.fiducials["FD2"].XY()
        x4, y4 = self.fiducials["FD3"].XY()
        x5, y5 = self.fiducials["FD4"].XY()
        x = (x1 + x2 + x4 + x5) / 4
        y = (y1 + y2 + y4 + y5) / 4
        return x, y

    def GetAngle(self):
        for fd in ["FD1", "FD2", "FD3", "FD4"]:
            assert fd in self.fiducials, f"{fd} not in fiducials"
        avg12X = (self.fiducials["FD1"].GetX() +
                  self.fiducials["FD2"].GetX()) / 2
        avg12Y = (self.fiducials["FD1"].GetY() +
                  self.fiducials["FD2"].GetY()) / 2
        avg34X = (self.fiducials["FD3"].GetX() +
                  self.fiducials["FD4"].GetX()) / 2
        avg34Y = (self.fiducials["FD3"].GetY() +
                  self.fiducials["FD4"].GetY()) / 2
        # Calculate the angle between the two average points
        angle = Angle(avg12X, avg12Y, avg34X, avg34Y) - 90.0
        return angle

    def visualize(self, output_name, includeTFBF=False):
        plt.figure(figsize=(8, 8))
        for name, fiducial in self.fiducials.items():
            x, y = fiducial.XY()
            plt.scatter(x, y, label=name)

        if includeTFBF:
            if self.TF is not None:
                x, y = self.TF.XY()
                plt.scatter(x, y, color='red', label='TF')
            if self.BF is not None:
                x, y = self.BF.XY()
                plt.scatter(x, y, color='blue', label='BF')
            plt.legend()
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("Silicon Fiducials")
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(output_name)
        plt.close()
        return


class HexaEdgeFiducials(object):
    def __init__(self, fiducials, isOGP=True, TF=None, BF=None):
        for fiducial in fiducials:
            assert isinstance(fiducial, Fiducial), \
                "All fiducials must be instances of the Fiducial class"
        self.fiducials = fiducials
        self.TF = TF
        self.BF = BF
        self.isOGP = isOGP

    def __str__(self):
        return f"HexaEdgeFiducials({self.fiducials})"

    def __repr__(self):
        return self.__str__()

    def IsOGPCoord(self):
        # Gantry and OGP have opposite y directions
        # OGP is the default
        return self.isOGP

    def IsGantryCoord(self):
        return not self.isOGP

    def ToGantry(self):
        if self.IsOGPCoord():
            print("Converting to Gantry coordinates")
            fiducials_new = self.fiducials.copy()
            for idx, fiducial in enumerate(self.fiducials):
                fiducials_new[idx] = fiducial.FlipY()
            self.fiducials = fiducials_new
            self.isOGP = False
        return self

    def ToOGP(self):
        if self.IsGantryCoord():
            print("Converting to OGP coordinates")
            fiducials_new = self.fiducials.copy()
            for idx, fiducial in enumerate(self.fiducials):
                fiducials_new[idx] = fiducial.FlipY()
            self.isOGP = True
        return self

    def Align(self, fids_TF_BF, base="BF"):
        fids_TF_BF_current = {}
        fids_TF_BF_current['TF'] = self.TF
        fids_TF_BF_current['BF'] = self.BF
        fids_new = self.fiducials.copy()
        for pos, fiducial in enumerate(self.fiducials):
            fids_new[pos] = AlignTFBF(
                fiducial, fids_TF_BF_current, fids_TF_BF, base=base)
        return HexaEdgeFiducials(fids_new, isOGP=self.isOGP, TF=fids_TF_BF['TF'], BF=fids_TF_BF['BF'])

    def XYPoints(self):
        return np.array([fiducial.XY() for fiducial in self.fiducials])

    def visualize(self, output_name, includeTFBF=False):
        plt.figure(figsize=(8, 8))
        for fiducial in self.fiducials:
            x, y = fiducial.XY()
            plt.scatter(x, y)

        if includeTFBF:
            if self.TF is not None:
                x, y = self.TF.XY()
                plt.scatter(x, y, color='red', label='TF')
            if self.BF is not None:
                x, y = self.BF.XY()
                plt.scatter(x, y, color='blue', label='BF')
            plt.legend()
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
        bounds=[(None, None), (None, None), (target_radius -
                                             5.0, target_radius + 5.0), (None, None)],
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
