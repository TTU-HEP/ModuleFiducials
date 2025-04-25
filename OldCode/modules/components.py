import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib
import numpy as np
from modules.helpers import Angle, GetCenterAndAngle, MeanAndRMS  # noqa
from scipy.optimize import minimize
from itertools import permutations

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


class HexaEdgeFiducials(object):
    def __init__(self, CH1_X=None, CH1_Y=None, CH81_X=None, CH81_Y=None,
                 CH8_X=None, CH8_Y=None, CH95_X=None, CH95_Y=None,
                 CH198_X=None, CH198_Y=None, CH190_X=None, CH190_Y=None):
        self.CH1_X = CH1_X
        self.CH1_Y = CH1_Y
        self.CH81_X = CH81_X
        self.CH81_Y = CH81_Y
        self.CH8_X = CH8_X
        self.CH8_Y = CH8_Y
        self.CH95_X = CH95_X
        self.CH95_Y = CH95_Y
        self.CH198_X = CH198_X
        self.CH198_Y = CH198_Y
        self.CH190_X = CH190_X
        self.CH190_Y = CH190_Y

    def visualize(self, output_name):
        print("Visualizing HexaEdgeFiducials")
        plt.figure(figsize=(8, 8))
        channels = {
            "CH1": (self.CH1_X, self.CH1_Y),
            "CH81": (self.CH81_X, self.CH81_Y),
            "CH8": (self.CH8_X, self.CH8_Y),
            "CH95": (self.CH95_X, self.CH95_Y),
            "CH198": (self.CH198_X, self.CH198_Y),
            "CH190": (self.CH190_X, self.CH190_Y)
        }

        for label, (x, y) in channels.items():
            if x is not None and y is not None:
                plt.scatter(x, y, label=label)
                plt.text(x, y, label, fontsize=9, ha='right')

        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("Channel Positions")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        print("Saving figure to", output_name)
        plt.show()
        plt.savefig(output_name)
        plt.close()
        return

    def get_hex_edges(self, cx, cy, r, theta):
        vertices = [
            (cx + r * np.cos(theta + i * np.pi / 3),
             cy + r * np.sin(theta + i * np.pi / 3)) for i in range(6)
        ]
        return [(vertices[i], vertices[(i+1) % 6]) for i in range(6)]

    def get_hex_vertices(self, cx, cy, r, theta):
        return [
            (cx + r * np.cos(theta + i * np.pi / 3),
             cy + r * np.sin(theta + i * np.pi / 3)) for i in range(6)
        ]

    def point_to_segment_dist(self, p, a, b):
        p, a, b = np.array(p), np.array(a), np.array(b)
        ap = p - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def point_to_closest_corner_dist(self, point, vertices):
        return min(np.linalg.norm(np.array(point) - np.array(v)) for v in vertices)

    def loss_fixed_assignment(self, params, points, edge_indices):
        if len(points) != len(edge_indices):
            return 1e6  # invalid, skip
        cx, cy, r, theta = params
        edges = self.get_hex_edges(cx, cy, r, theta)
        total = 0
        for i, p in enumerate(points):
            a, b = edges[edge_indices[i]]
            total += self.point_to_segment_dist(p, a, b) ** 2
        return total

    def find_all_fits(self, points, tol=1e-3):
        results = []
        n = len(points)
        for edge_indices in permutations(range(6), n):
            centroid = np.mean(points, axis=0)
            init = [centroid[0], centroid[1], 1.0, 0.0]
            res = minimize(self.loss_fixed_assignment, init, args=(
                points, edge_indices), method='Nelder-Mead')
            if res.success and res.fun < tol:
                params = tuple(np.round(res.x, 6))
                if params not in results:
                    results.append(params)
        return results

    def select_closest_to_corner_solution(self, points, hex_solutions):
        best_score = float('inf')
        best_params = None
        for params in hex_solutions:
            cx, cy, r, theta = params
            vertices = self.get_hex_vertices(cx, cy, r, theta)
            total_dist = sum(self.point_to_closest_corner_dist(
                p, vertices) for p in points)
            if total_dist < best_score:
                best_score = total_dist
                best_params = params
        return best_params

    def select_closest_to_radius_solution(self, points, hex_solutions, radius=94.59):
        best_score = float('inf')
        best_params = None
        for params in hex_solutions:
            cx, cy, r, theta = params
            if abs(r - radius) < best_score:
                best_score = abs(r - radius)
                best_params = params
        return best_params

    def plot_hexagon(self, cx, cy, r, theta, points, output_name):
        vertices = self.get_hex_vertices(cx, cy, r, theta)
        hexagon = np.array(vertices + [vertices[0]])  # close loop

        plt.plot(hexagon[:, 0], hexagon[:, 1], 'b-', label="Fitted Hexagon")
        plt.plot(*zip(*points), 'ro', label="Input Points")
        plt.gca().set_aspect('equal')
        plt.title("Best Hexagon Fit (Closest to Corners)")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(output_name)

    def runFit(self, output_name):
        points = []
        if self.CH1_X is not None and self.CH1_Y is not None:
            points.append((self.CH1_X, self.CH1_Y))
        if self.CH81_X is not None and self.CH81_Y is not None:
            points.append((self.CH81_X, self.CH81_Y))
        if self.CH8_X is not None and self.CH8_Y is not None:
            points.append((self.CH8_X, self.CH8_Y))
        if self.CH95_X is not None and self.CH95_Y is not None:
            points.append((self.CH95_X, self.CH95_Y))
        if self.CH198_X is not None and self.CH198_Y is not None:
            points.append((self.CH198_X, self.CH198_Y))
        if self.CH190_X is not None and self.CH190_Y is not None:
            points.append((self.CH190_X, self.CH190_Y))

        if len(points) < 3:
            raise ValueError(
                "At least 3 points are required to fit a hexagon.")

        points = points[:3]  # Use only the first three points for fitting

        # Fit hexagon to boundary points
        all_fits = self.find_all_fits(points)

        print(f"Total valid fits found: {len(all_fits)}")
        if not all_fits:
            print("No valid hexagon fits found.")
        else:
            # best_fit = self.select_closest_to_corner_solution(points, all_fits)
            best_fit = self.select_closest_to_radius_solution(points, all_fits)
            print(f"Best fit (closest to corners):")
            print(f"  Center: ({best_fit[0]}, {best_fit[1]})")
            print(f"  Radius: {best_fit[2]}")
            print(f"  Angle (rad): {best_fit[3]}")
            self.plot_hexagon(*best_fit, points, output_name)
        # Return the fitted parameters
        return best_fit


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

    def _orientation_vertices(self):
        v01 = self.v1 - self.v2
        v12 = self.v2 - self.v3
        v23 = self.v3 - self.v1
        # Normalize all vectors
        v01 /= np.linalg.norm(v01)
        v12 /= np.linalg.norm(v12)
        v23 /= np.linalg.norm(v23)
        # angle
        angle12 = np.degrees(math.atan2(v12[1], v12[0]))
        angle23 = np.degrees(math.atan2(v23[1], v23[0]))
        angle01 = np.degrees(math.atan2(v01[1], v01[0]))
        # average
        avgle = (angle12 + angle23 + angle01) / 3
        avgle_vec = (v12 + v23 + v01) / 3
        norm = np.linalg.norm(avgle_vec)
        if norm < 1e-10:
            avgle_vec = np.array([1.0, 0.0])
        else:
            avgle_vec /= norm
        return avgle, avgle_vec

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
        # self.angle_deg, self.orientation_vector = self._orientation_relative_to_BP3_BP4(
        #    self.vertices)
        self.angle_deg, self.orientation_vector = self._orientation_vertices()

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
