import numpy as np
from pyquaternion import Quaternion


def mprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="   ")
        print("")


def Euler2A(phi, theta, psi):
    A = np.zeros(shape=(3, 3))

    A[0][0] = np.cos(theta) * np.cos(psi)
    A[0][1] = np.cos(psi) * np.sin(theta) * np.sin(phi) - np.cos(phi) * np.sin(psi)
    A[0][2] = np.cos(phi) * np.cos(psi) * np.sin(theta) + np.sin(phi) * np.sin(psi)

    A[1][0] = np.cos(theta) * np.sin(psi)
    A[1][1] = np.cos(phi) * np.cos(psi) + np.sin(theta) * np.sin(phi) * np.sin(psi)
    A[1][2] = np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi)

    A[2][0] = -np.sin(theta)
    A[2][1] = np.cos(theta) * np.sin(phi)
    A[2][2] = np.cos(theta) * np.cos(phi)

    return A


def AxisAngle(A):
    if np.linalg.det(A) == 1:
        A1 = A - np.eye(3)
        p = np.cross(A1[0], A1[1])
        p = p / np.linalg.norm(p)
        u = A1[0]
        u = u / np.linalg.norm(u)
        u_prime = A.dot(u)
        u_prime = u_prime / np.linalg.norm(u_prime)
        phi = np.arccos(u.dot(u_prime))
        if np.linalg.det(np.array([u, u_prime, p])) < 0:
            p = -p
        return p, phi


def RodriguesFormula(p, phi):
    px = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    ppt = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ppt[i][j] = p[i] * p[j]
    return ppt + np.cos(phi) * (np.eye(3) - ppt) + np.sin(phi) * px


def A2Euler(A):
    if A[2][0] < 1:
        if A[2][0] > -1:
            psi = np.arctan2(A[1][0], A[0][0])
            theta = np.arcsin(-A[2][0])
            phi = np.arctan2(A[2][1], A[2][2])
        else:
            psi = np.arctan2(-A[0][1], A[1][1])
            theta = np.pi / 2
            phi = 0
    else:
        psi = np.arctan2(-A[0][1], A[1][1])
        theta = -np.pi / 2
        phi = 0

    return phi, theta, psi


def AxisAngle2Q(p, phi):
    w = np.cos(phi / 2)
    p = p / np.linalg.norm(p)
    im = np.sin(phi / 2) * p
    q = Quaternion(real=w, imaginary=im)
    return q


def Q2AxisAngle(q):
    q = q.normalised
    if q.real < 0:
        q = -q
    phi = 2 * np.arccos(q.real)
    if q.real == 1:
        p = np.eye(1, 3)
    else:
        p = q.imaginary
        p = p / np.linalg.norm(p)

    return p, phi


def main():
    # phi = -np.arctan(1/4)
    # theta = -np.arcsin(8/9)
    # psi = np.arctan(4)

    phi = np.pi / 3
    theta = np.pi / 4
    psi = np.pi / 5

    print("Input rotation angles: phi = %s, theta = %s, psi = %s\n" % (phi, theta, psi))
    print("-Euler2A-")
    print("A =")
    A = Euler2A(phi, theta, psi)
    mprint(A)

    print("\n-AxisAngle-")
    p, rotation_angle = AxisAngle(A)
    print("p = %s, rotation angle phi = %s" % (p, rotation_angle))

    print("\n-Rodrigues formula-")
    A = RodriguesFormula(p, rotation_angle)
    print("Back to A.\nA = ")
    mprint(A)

    print("\n-A2Euler")
    phi, theta, psi = A2Euler(A)
    print("phi = %s, theta = %s, psi = %s" % (phi, theta, psi))

    print("\n-AxisAngle2Q-")
    q = AxisAngle2Q(p, rotation_angle)
    print("Quaternion q = %s" % q)

    print("\n-Q2AxisAngle-")
    p, rotation_angle = Q2AxisAngle(q)
    print("p = %s, rotation angle phi = %s" % (p, rotation_angle))


if __name__ == '__main__':
    main()
