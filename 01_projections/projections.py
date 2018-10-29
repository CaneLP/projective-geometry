import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math


def set_homogeneous(dots):
    return np.c_[dots, np.ones(shape=(dots.shape[0], 1))]


def naive_algorithm(dots, dots_prime):
    p1 = np.zeros(shape=(3, 3))
    p2 = np.zeros(shape=(3, 3))
    coefficients = np.linalg.solve(dots[:-1].transpose(), dots[-1].transpose())
    for j in range(len(coefficients)):
        p1[:, j] = coefficients[j]*dots[j]
    coefficients_prime = np.linalg.solve(dots_prime[: -1].transpose(), dots_prime[-1].transpose())
    for j in range(len(coefficients_prime)):
        p2[:, j] = coefficients_prime[j]*dots_prime[j]
    return np.matmul(p2, np.linalg.inv(p1))


def dlt_algorithm(dots, dots_prime):
    correspondence_matrix = np.zeros(shape=(2*dots.shape[0], 9))
    for i in range(dots.shape[0]):
        correspondence_matrix[2*i, 3:6] = -dots_prime[i][2] * dots[i]
        correspondence_matrix[2*i, 6:9] = dots_prime[i][1] * dots[i]
        correspondence_matrix[2*i+1, 0:3] = dots_prime[i][2] * dots[i]
        correspondence_matrix[2*i+1, 6:9] = -dots_prime[i][0] * dots[i]
    u, s, vh = np.linalg.svd(correspondence_matrix, full_matrices=True)
    p = np.array([vh.transpose()[0:3, -1]])
    p = np.r_[p, np.array([vh.transpose()[3:6, -1]])]
    p = np.r_[p, np.array([vh.transpose()[6:9, -1]])]
    return p


def normalization_matrix(dots):
    translation_vector = np.mean(dots, axis=0)
    G = np.eye(3, 3)
    G[0][2] = -translation_vector[0]
    G[1][2] = -translation_vector[1]
    avg_distance = 0
    for dot in dots:
        avg_distance += math.sqrt(dot[0]*dot[0] + dot[1]*dot[1])
    avg_distance /= dots.shape[0]
    S = np.diagflat([math.sqrt(2) / avg_distance] * 3)
    S[2][2] = 1
    return np.matmul(S, G)


def normalized_dlt_algorithm(dots, dots_prime):
    t = normalization_matrix(dots)
    t_prime = normalization_matrix(dots_prime)

    dots_normalized = np.zeros(shape=(dots.shape[0], dots.shape[1]))
    dots_prime_normalized = np.zeros(shape=(dots_prime.shape[0], dots_prime.shape[1]))

    for i in range(dots.shape[0]):
        dots_normalized[i] = np.matmul(t, dots[i].transpose())
        dots_prime_normalized[i] = np.matmul(t_prime, dots_prime[i].transpose())

    p = dlt_algorithm(dots_normalized, dots_prime_normalized)

    return np.matmul(np.matmul(np.linalg.inv(t_prime), p), t)


def main():
    # with open('input_homogeneous.txt') as f:
    #     data = f.read()
    with open('input/input5_sum.txt') as f:
        data = f.read()

    num_of_dots = int(data.split()[0])
    num_of_colons = int(data.split()[1])

    dots = np.zeros(shape=(num_of_dots, num_of_colons))
    for i in range(num_of_dots):
        for j in range(num_of_colons):
            dots[i][j] = data.split()[num_of_colons * i + j + 2]

    dots_prime = np.zeros(shape=(num_of_dots, num_of_colons))
    for i in range(num_of_dots):
        for j in range(num_of_colons):
            dots_prime[i][j] = data.split()[num_of_colons * i + j + 2 + num_of_dots * num_of_colons]

    if num_of_colons == 2:
        # dots are not homogeneous
        dots = set_homogeneous(dots)
        dots_prime = set_homogeneous(dots_prime)

    if num_of_dots == 4:
        p_naive = naive_algorithm(dots, dots_prime)
        print("Naive algorithm:\n%s" % p_naive)

    p_dlt = dlt_algorithm(dots, dots_prime)
    print("DLT algorithm:\n%s" % p_dlt)
    if num_of_dots == 4:
        print("DLT algorithm scaled to naive algorithm:\n%s" % ((p_naive[0][0]/p_dlt[0][0])*p_dlt))
    p_dlt_normalized = normalized_dlt_algorithm(dots, dots_prime)
    print("Normalized DLT algorithm:\n%s" % p_dlt_normalized)
    if num_of_dots == 4:
        print("Normalized DLT algorithm scaled to naive algorithm:\n%s" % ((p_naive[0][0]/p_dlt_normalized[0][0])*p_dlt_normalized))
    print("Normalized DLT algorithm scaled to DLT:\n%s" % ((p_dlt[0][0]/p_dlt_normalized[0][0])*p_dlt_normalized))

    # plt.axis([-10, 20, -10, 20])

    x = np.r_[dots[:, 0], dots[0, 0]]
    y = np.r_[dots[:, 1], dots[0, 1]]
    # x = np.flip(x)
    # y = np.flip(y)
    plt.plot(x, y, 'b', lw=3)
    plt.scatter(x, y, c='b', s=120)
    x_prime = np.r_[dots_prime[:, 0], dots_prime[0, 0]]
    y_prime = np.r_[dots_prime[:, 1], dots_prime[0, 1]]
    # x_prime = np.flip(x_prime)
    # y_prime = np.flip(y_prime)
    plt.plot(x_prime, y_prime, 'r', lw=3)
    plt.scatter(x_prime, y_prime, c='r', s=120)

    x1 = -4
    y1 = 1
    plt.scatter(x1, y1, c='black', lw=3)
    new_dot = np.array([x1, y1, 1])
    new_dot_dlt = np.matmul(p_dlt, new_dot)
    new_dot_dlt_norm = np.matmul(p_dlt_normalized, new_dot)

    plt.grid(alpha=0.7, axis='both')

    green_patch = mpatches.Patch(color='green', label='Point projected with naive algorithm')
    red_patch = mpatches.Patch(color='red', label='Projected points')
    blue_patch = mpatches.Patch(color='blue', label='Original points')
    orange_patch = mpatches.Patch(color='orange', label='Point projected with DLT algorithm')
    yellow_patch = mpatches.Patch(color='yellow', label='Point projected with normalized DLT algorithm')

    if num_of_dots == 4:
        new_dot_naive = np.matmul(p_naive, new_dot)
        plt.scatter(new_dot_naive[0], new_dot_naive[1], c='green', lw=3)
        plt.legend(handles=[blue_patch, red_patch, green_patch, orange_patch, yellow_patch])
        plt.plot([new_dot_naive[0], new_dot_dlt[0]], [new_dot_naive[1], new_dot_dlt[1]], c='gray', alpha=0.8)
        plt.plot([new_dot_naive[0], new_dot_dlt_norm[0]], [new_dot_naive[1], new_dot_dlt_norm[1]], c='gray', alpha=0.8)
    else:
        plt.legend(handles=[blue_patch, red_patch, orange_patch, yellow_patch])

    plt.scatter(new_dot_dlt[0], new_dot_dlt[1], c='orange', lw=3)
    plt.scatter(new_dot_dlt_norm[0], new_dot_dlt_norm[1], c='yellow', lw=3)

    plt.plot([new_dot_dlt[0], new_dot_dlt_norm[0]], [new_dot_dlt[1], new_dot_dlt_norm[1]], c='white', alpha=0.0)

    plt.show()


if __name__ == '__main__':
    main()
