import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib
import numpy as np
from modules.helpers import Angle, GetCenterAndAngle, MeanAndRMS  # noqa
matplotlib.use('Agg')


class TrayFiducial(object):
    def __init__(self, tray_Z, OP1_X, OP1_Y, CP1_X, CP1_Y, OP2_X, OP2_Y, CP2_X, CP2_Y):
        self.tray_Z = tray_Z
        self.OP1_X = OP1_X
        self.OP1_Y = OP1_Y
        self.CP1_X = CP1_X
        self.CP1_Y = CP1_Y
        self.OP2_X = OP2_X
        self.OP2_Y = OP2_Y
        self.CP2_X = CP2_X
        self.CP2_Y = CP2_Y
        self.checkNan()

    def checkNan(self):
        if any(math.isnan(value) for value in [self.tray_Z, self.OP1_X, self.OP1_Y, self.CP1_X, self.CP1_Y,
                                               self.OP2_X, self.OP2_Y, self.CP2_X, self.CP2_Y]):
            raise ValueError("NaN values found in TrayFiducial attributes",
                             self.tray_Z, self.OP1_X, self.OP1_Y, self.CP1_X, self.CP1_Y,
                             self.OP2_X, self.OP2_Y, self.CP2_X, self.CP2_Y)

    def __str__(self):
        return f"TrayFiducial(tray_Z={self.tray_Z}, OP1_X={self.OP1_X}, OP1_Y={self.OP1_Y}, CP1_X={self.CP1_X}, CP1_Y={self.CP1_Y}, OP2_X={self.OP2_X}, OP2_Y={self.OP2_Y}, CP2_X={self.CP2_X}, CP2_Y={self.CP2_Y}) \n\
            angle1={self.GetAngle(1)}, angle2={self.GetAngle(2)}, \n\
            centers={self.GetCenters()}"

    def __repr__(self):
        return self.__str__()

    def GetAngles(self):
        angle1 = Angle(self.OP1_X, self.OP1_Y, self.CP1_X, self.CP1_Y)
        angle2 = Angle(self.OP2_X, self.OP2_Y, self.CP2_X, self.CP2_Y)
        return angle1, angle2

    def GetAngle(self, position):
        if position == 1:
            return Angle(self.OP1_X, self.OP1_Y, self.CP1_X, self.CP1_Y)
        elif position == 2:
            return Angle(self.OP2_X, self.OP2_Y, self.CP2_X, self.CP2_Y)
        else:
            raise ValueError("Position must be 1 or 2")

    def GetCenters(self):
        return self.OP1_X, self.OP1_Y, self.OP2_X, self.OP2_Y

    def GetCenter(self, position):
        if position == 1:
            return self.OP1_X, self.OP1_Y
        elif position == 2:
            return self.OP2_X, self.OP2_Y
        else:
            raise ValueError("Position must be 1 or 2")


class HexaFiducials(object):
    def __init__(self, FD3_X, FD3_Y, FD6_X, FD6_Y, FD1_X, FD1_Y, FD2_X, FD2_Y, FD4_X, FD4_Y, FD5_X, FD5_Y, center_X, center_Y):
        self.FD3_X = FD3_X
        self.FD3_Y = FD3_Y
        self.FD6_X = FD6_X
        self.FD6_Y = FD6_Y
        self.FD1_X = FD1_X
        self.FD1_Y = FD1_Y
        self.FD2_X = FD2_X
        self.FD2_Y = FD2_Y
        self.FD4_X = FD4_X
        self.FD4_Y = FD4_Y
        self.FD5_X = FD5_X
        self.FD5_Y = FD5_Y
        self.center_X = center_X
        self.center_Y = center_Y

    def __str__(self):
        center4_X, center4_Y, angle4 = self.GetCenter(4)
        cneter2_X, center2_Y, angle2 = self.GetCenter(2)
        center1_X, center1_Y, angle1 = self.GetCenter(1)
        return f"HexaFiducials(FD3_X={self.FD3_X}, FD3_Y={self.FD3_Y}, FD6_X={self.FD6_X}, FD6_Y={self.FD6_Y}, FD1_X={self.FD1_X}, FD1_Y={self.FD1_Y}, FD2_X={self.FD2_X}, FD2_Y={self.FD2_Y}, FD4_X={self.FD4_X}, FD4_Y={self.FD4_Y}, FD5_X={self.FD5_X}, FD5_Y={self.FD5_Y}, center_X={self.center_X}, center_Y={self.center_Y}) \n\
        center4_X={center4_X}, center4_Y={center4_Y}, angle={angle4} \n\
        center2_X={cneter2_X}, center2_Y={center2_Y}, angle={angle2} \n\
        center1_X={center1_X}, center1_Y={center1_Y}, angle={angle1} \n"

    def __repr__(self):
        return self.__str__()

    def GetCenter(self, nfds=4):
        if nfds == 4:
            center_x, center_y, angle = GetCenterAndAngle(self.FD1_X, self.FD1_Y, self.FD2_X, self.FD2_Y,
                                                          self.FD4_X, self.FD4_Y, self.FD5_X, self.FD5_Y)
        elif nfds == 2:
            center_x, center_y, angle = GetCenterAndAngle(
                self.FD3_X, self.FD3_Y, self.FD6_X, self.FD6_Y)
            angle = 90 + angle
        elif nfds == 1:
            center_x, center_y, angle = self.center_X, self.center_Y, 0.0
        else:
            raise ValueError("Number of fiducials must be 4 or 6")
        return center_x, center_y, angle


class BPFiducial(object):
    def __init__(self, BP3_X, BP3_Y, BP4_X, BP4_Y, BP5_X, BP5_Y, BP6_X, BP6_Y, BP7_X, BP7_Y, BP8_X, BP8_Y):
        self.BP3_X = BP3_X
        self.BP3_Y = BP3_Y
        self.BP4_X = BP4_X
        self.BP4_Y = BP4_Y
        self.BP5_X = BP5_X
        self.BP5_Y = BP5_Y
        self.BP6_X = BP6_X
        self.BP6_Y = BP6_Y
        self.BP7_X = BP7_X
        self.BP7_Y = BP7_Y
        self.BP8_X = BP8_X
        self.BP8_Y = BP8_Y

        self.points = {
            "BP3": np.array([BP3_X, BP3_Y]),
            "BP4": np.array([BP4_X, BP4_Y]),
            "BP5": np.array([BP5_X, BP5_Y]),
            "BP6": np.array([BP6_X, BP6_Y]),
            "BP7": np.array([BP7_X, BP7_Y]),
            "BP8": np.array([BP8_X, BP8_Y])
        }
        self._compute_all()

    def __str__(self):
        return f"BPFiducial({self.points}), vertices={self.vertices}, \n\
            centroid={self.centroid},\n\
            angle_deg={self.angle_deg})"

    def __repr__(self):
        return self.__str__()

    def _line_from_points(self, p1, p2):
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = -(A * p1[0] + B * p1[1])
        return A, B, C

    def _intersection(self, L1, L2):
        A1, B1, C1 = L1
        A2, B2, C2 = L2
        det = A1 * B2 - A2 * B1
        if np.isclose(det, 0):
            raise ValueError("Lines do not intersect")
        x = (B1 * C2 - B2 * C1) / det
        y = (C1 * A2 - C2 * A1) / det
        return np.array([x, y])

    def _triangle_centroid(self, p1, p2, p3):
        return (p1 + p2 + p3) / 3

    def _orientation_angle(self, pts):
        pts = np.array(pts)
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        principal_axis = eigvecs[:, np.argmax(eigvals)]
        angle = math.atan2(principal_axis[1], principal_axis[0])
        return np.degrees(angle), principal_axis

    def _orientation_average(self, pts):
        # Compute the direction vectors for each triangle edge
        v01 = pts[1] - pts[0]
        v12 = pts[2] - pts[1]
        v20 = pts[0] - pts[2]
        # Normalize all vectors
        v01 /= np.linalg.norm(v01)
        v12 /= np.linalg.norm(v12)
        v20 /= np.linalg.norm(v20)
        # Average them
        avg_vec = (v01 + v12 + v20) / 3
        norm = np.linalg.norm(avg_vec)
        if norm < 1e-10:
            avg_vec = np.array([1.0, 0.0])  # fallback
        else:
            avg_vec /= norm
        angle = math.atan2(avg_vec[1], avg_vec[0])
        return np.degrees(angle), avg_vec

    def _orientation_relative_to_BP3_BP4(self, pts):
        # Triangle edge directions
        v01 = pts[1] - pts[0]
        v12 = pts[2] - pts[1]
        v20 = pts[0] - pts[2]
        v01 /= np.linalg.norm(v01)
        v12 /= np.linalg.norm(v12)
        v20 /= np.linalg.norm(v20)
        avg_vec = (v01 + v12 + v20) / 3
        norm = np.linalg.norm(avg_vec)
        if norm < 1e-10:
            avg_vec = np.array([1.0, 0.0])
        else:
            avg_vec /= norm

        # Reference direction = BP3–BP4
        ref = self.points["BP4"] - self.points["BP3"]
        if np.linalg.norm(ref) < 1e-10:
            ref = np.array([1.0, 0.0])  # fallback
        else:
            ref /= np.linalg.norm(ref)

        # Compute angle difference (avg_vec relative to ref)
        dot = np.clip(np.dot(ref, avg_vec), -1.0, 1.0)
        cross = ref[0]*avg_vec[1] - ref[1]*avg_vec[0]
        angle = math.atan2(cross, dot)  # signed angle from ref to avg
        angle_deg = np.degrees(angle) - 90.0
        return angle_deg, avg_vec

    def _compute_all(self):
        # Create lines from fixed pairs
        self.line1 = self._line_from_points(
            self.points["BP3"], self.points["BP4"])
        self.line2 = self._line_from_points(
            self.points["BP5"], self.points["BP6"])
        self.line3 = self._line_from_points(
            self.points["BP7"], self.points["BP8"])

        # Intersections = triangle vertices
        self.v1 = self._intersection(self.line1, self.line2)
        self.v2 = self._intersection(self.line2, self.line3)
        self.v3 = self._intersection(self.line3, self.line1)
        self.vertices = [self.v1, self.v2, self.v3]

        # Centroid and orientation
        self.centroid = self._triangle_centroid(self.v1, self.v2, self.v3)
        self.angle_deg, self.orientation_vector = self._orientation_relative_to_BP3_BP4(
            self.vertices)

    def visualize(self, output="test.png", xlim=(-10, 10), ylim=(-10, 10)):
        def plot_line(A, B, C):
            if B != 0:
                x = np.linspace(xlim[0], xlim[1], 500)
                y = (-A * x - C) / B
            else:
                x = -C / A * np.ones(500)
                y = np.linspace(ylim[0], ylim[1], 500)
            plt.plot(x, y, '--', alpha=0.6)

        # Plot lines
        plt.figure(figsize=(6, 6))
        for line in [self.line1, self.line2, self.line3]:
            plot_line(*line)

        # Plot triangle
        tri = np.array(self.vertices + [self.vertices[0]])
        plt.scatter(*self.points["BP3"], c='orange', label='BP3 & BP4')
        plt.scatter(*self.points["BP4"], c='orange')
        plt.scatter(*self.points["BP5"], c='purple', label='BP5 & BP6')
        plt.scatter(*self.points["BP6"], c='purple')
        plt.scatter(*self.points["BP7"], c='cyan', label='BP7 & BP8')
        plt.scatter(*self.points["BP8"], c='cyan')
        plt.plot(tri[:, 0], tri[:, 1], 'b-', label='Triangle')
        plt.scatter(*zip(*self.vertices), c='r', label='Vertices')
        plt.scatter(*self.centroid, c='k', marker='x', s=100, label='Centroid')

        # Orientation arrow
        arrow_scale = 0.5
        plt.arrow(self.centroid[0], self.centroid[1],
                  arrow_scale * self.orientation_vector[0],
                  arrow_scale * self.orientation_vector[1],
                  head_width=0.2, color='g', label='Orientation')

        plt.title(f'Orientation: {self.angle_deg:.2f}°')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.show()
        plt.savefig(output)

    def GetCenter(self):
        return self.centroid[0], self.centroid[1]

    def GetAngle(self):
        return self.angle_deg


class SiliconFiducial(object):
    def __init__(self, SiC_X, SiC_Y):
        self.SiC_X = SiC_X
        self.SiC_Y = SiC_Y

    def __str__(self):
        return f"SiliconFiducial(SiC_X={self.SiC_X}, SiC_Y={self.SiC_Y})"

    def __repr__(self):
        return self.__str__()

    def GetCenter(self):
        return self.SiC_X, self.SiC_Y
