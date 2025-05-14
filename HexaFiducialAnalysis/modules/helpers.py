import matplotlib.pyplot as plt
import mplhep as hep
import math


def DrawFids(Xs, Ys, title):
    hep.set_style(hep.style.CMS)
    plt.figure(figsize=(8, 6))
    plt.plot(Xs, Ys, 'ro')
    plt.title(title)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.grid()
    plt.show()
    plt.savefig(title + ".png")


def DrawTrays(Xs, Ys, Xs_PCB, Ys_PCB, Xs_centers_PCB, Ys_centers_PCB, title):
    hep.set_style(hep.style.CMS)
    plt.figure(figsize=(6, 10))
    plt.xlim(0, 250)
    plt.ylim(0, 400)
    plt.plot(Xs, Ys, 'ro')
    plt.plot(Xs_PCB, Ys_PCB, 'bo')
    plt.plot(Xs_centers_PCB, Ys_centers_PCB, 'b*')
    plt.title(title)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.grid()
    plt.show()
    plt.savefig(title + ".png")


def MeanAndRMS(data):
    mean = sum(data) / len(data)
    rms = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return mean, rms


def Angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def CleanAngle(angle):
    # Normalize angle to be between -180 and 180 degrees
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle


def GetCenterAndAngle(X1, Y1, X2, Y2, X3=None, Y3=None, X4=None, Y4=None):
    if X3 is not None and Y3 is not None and X4 is not None and Y4 is not None:
        # take the average of the two pairs
        X1 = (X1 + X2) / 2.0
        Y1 = (Y1 + Y2) / 2.0
        X2 = (X3 + X4) / 2.0
        Y2 = (Y3 + Y4) / 2.0
    center_x = (X1 + X2) / 2.0
    center_y = (Y1 + Y2) / 2.0
    angle = Angle(X1, Y1, X2, Y2)
    return center_x, center_y, angle
