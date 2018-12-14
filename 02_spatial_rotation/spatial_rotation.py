import numpy as np
from pyquaternion import Quaternion
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from time import sleep

tm = 200
t = 0
frame_rate = 50


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
    eps = 1e-10
    # Dodato epsilon zbog uocenih malih gresaka zaokruzivanja numpy-a, prilikom testiranja
    # Kada su vrednosti 0.9999999.... prihvati kao 1
    if np.abs(np.linalg.det(A) - 1) < eps:
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
    q = Quaternion(imaginary=im, real=w)
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


def slerp(q1, q2, t, tm):
    dot = q1.conjugate.real * q2.real
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.95:
        return q1

    phi = np.arccos(dot)
    qs = (np.sin(phi * (1 - t / tm)) / np.sin(phi)) * q1 + (np.sin(phi * (t / tm)) / np.sin(phi)) * q2
    return qs


def center_translation(c1, c2, t, tu):
    return (1 - t / tu) * c1 + (t / tu) * c2


def glut_initialization():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(650, 650)
    glutCreateWindow("Slerp animation")

    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)


def glut_set_callbacks():
    glutDisplayFunc(display)
    glutIdleFunc(animate)
    glutKeyboardFunc(keyboard)


def glut_perspective():
    glMatrixMode(GL_PROJECTION)
    gluPerspective(50, 1, 1, 50)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(10, 10, 10,
              0, 0, 0,
              0, 0, 1)
    glPushMatrix()
    glutMainLoop()


def keyboard(key, x, y):
    if key == b'q':
        exit()


def material():
    glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
    glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.8, 0.8, 0.8, 1.0))
    glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(1.0, 0.0, 1.0, 1.0))
    glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(50.0))


def lighting():
    glLightfv(GL_LIGHT0, GL_AMBIENT, GLfloat_4(1.0, 1.0, 1.0, 0.8))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, GLfloat_4(0.25, 0.5, 0.95, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(1.0, 1.0, 1.0, 0.0))
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

def red_lighting():
    glLightfv(GL_LIGHT1, GL_AMBIENT, GLfloat_4(1.0, 0.0, 0.0, 0.8))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, GLfloat_4(0.7, 0.0, 0.0, 0.5))
    glLightfv(GL_LIGHT0, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(1.0, 1.0, 1.0, 0.0))
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT1)


def draw_axes():
    glLineWidth(1.5)

    # x
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)

    glVertex3f(-5.0, 0.0, 0.0)
    glVertex3f(5.0, 0.0, 0.0)

    # arrow
    glVertex3f(5.0, 0.0, 0.0)
    glVertex3f(4.0, 1.0, 0.0)

    glVertex3f(5.0, 0.0, 0.0)
    glVertex3f(4.0, -1.0, 0.0)
    glEnd()
    glFlush()

    # y
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, -5.0, 0.0)
    glVertex3f(0.0, 5.0, 0.0)

    # arrow
    glVertex3f(0.0, 5.0, 0.0)
    glVertex3f(1.0, 4.0, 0.0)

    glVertex3f(0.0, 5.0, 0.0)
    glVertex3f(-1.0, 4.0, 0.0)
    glEnd()
    glFlush()

    # z
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, -5.0)
    glVertex3f(0.0, 0.0, 5.0)

    # arrow
    glVertex3f(0.0, 0.0, 5.0)
    glVertex3f(0.0, 1.0, 4.0)

    glVertex3f(0.0, 0.0, 5.0)
    glVertex3f(0.0, -1.0, 4.0)
    glEnd()
    glFlush()


def display():
    global t, tm
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # point1 = np.array([3, 2, 1])
    point1 = np.array([3, -1, -1])
    # point1 = np.array([0, 0, 0])
    euler_angles1 = np.array([19 * np.pi / 7, 3 * np.pi / 7, 4 * np.pi / 5])
    A1 = Euler2A(euler_angles1[0], euler_angles1[1], euler_angles1[2])
    p1, rotation_angle1 = AxisAngle(A1)
    q1 = AxisAngle2Q(p1, rotation_angle1)
    # print(q1)

    # point2 = np.array([1.5, 0, 0])
    # point2 = np.array([5, 3, 3])
    point2 = np.array([-3, 1, -2])
    euler_angles2 = np.array([np.pi / 2, -np.pi / 3, 5 * np.pi / 4])
    A2 = Euler2A(euler_angles2[0], euler_angles2[1], euler_angles2[2])
    p2, rotation_angle2 = AxisAngle(A2)
    q2 = AxisAngle2Q(p2, rotation_angle2)
    # print(q2)

    qs = slerp(q1, q2, t, tm)

    p_s, rotation_angle_s = Q2AxisAngle(qs)
    A_s = RodriguesFormula(p_s, rotation_angle_s)
    phi, theta, psi = A2Euler(A_s)
    c = center_translation(point1, point2, t, tm)

    red_lighting()

    glPushMatrix()
    glTranslatef(point1[0], point1[1], point1[2])
    glRotatef(np.rad2deg(euler_angles1[0]), 1, 0, 0)
    glRotatef(np.rad2deg(euler_angles1[1]), 0, 1, 0)
    glRotatef(np.rad2deg(euler_angles1[2]), 0, 0, 1)
    glutWireTeapot(0.75)
    glPopMatrix()

    glDisable(GL_LIGHT1)
    lighting()
    material()

    glPushMatrix()
    # glTranslatef(point1[0], point1[1], point1[2])
    # glRotatef(np.rad2deg(euler_angles1[0]), 1, 0, 0)
    # glRotatef(np.rad2deg(euler_angles1[1]), 0, 1, 0)
    # glRotatef(np.rad2deg(euler_angles1[2]), 0, 0, 1)
    # glRotatef((euler_angles1[0]), 1, 0, 0)
    # glRotatef((euler_angles1[1]), 0, 1, 0)
    # glRotatef((euler_angles1[2]), 0, 0, 1)
    glTranslatef(c[0], c[1], c[2])
    glRotatef(np.rad2deg(phi), 1, 0, 0)
    glRotatef(np.rad2deg(theta), 0, 1, 0)
    glRotatef(np.rad2deg(psi), 0, 0, 1)
    # glRotatef((phi), 1, 0, 0)
    # glRotatef((theta), 0, 1, 0)
    # glRotatef((psi), 0, 0, 1)
    glutSolidTeapot(0.75)
    glPopMatrix()

    glDisable(GL_LIGHTING)
    glDisable(GL_LIGHT0)

    red_lighting()

    glPushMatrix()
    glTranslatef(point2[0], point2[1], point2[2])
    glRotatef(np.rad2deg(euler_angles2[0]), 1, 0, 0)
    glRotatef(np.rad2deg(euler_angles2[1]), 0, 1, 0)
    glRotatef(np.rad2deg(euler_angles2[2]), 0, 0, 1)
    glutWireTeapot(0.75)
    glPopMatrix()

    glDisable(GL_LIGHTING)
    glDisable(GL_LIGHT0)
    glDisable(GL_LIGHT1)
    draw_axes()

    print(q1)
    print(qs)
    print(q2)
    print(c)

    glutSwapBuffers()


def animate():
    global t, tm
    glutPostRedisplay()
    if t == tm:
        sleep(1)
        t = 0
    t += 1
    sleep(1 / float(frame_rate))


def main():
    # phi = -np.arctan(1/4)
    # theta = -np.arcsin(8/9)
    # psi = np.arctan(4)

    # phi = 3.14/3
    # theta = 3.14/2
    # psi = -3.14/4

    phi = np.pi / 3
    theta = np.pi / 4
    psi = np.pi / 5

    # phi, theta, psi = [np.pi/5, np.pi/6, np.pi/4]

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

    print("\n-Slerp-")
    glut_initialization()
    glut_set_callbacks()
    glut_perspective()
    animate()


if __name__ == '__main__':
    main()
