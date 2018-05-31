import random

# Dempster Shafer Module to work with 2 classes data
# Mass asigment function are represented as 4-tuple:
#  (m_null = 0, m_cls_0, m_cls_1, m_either)
import torch


def dempster_rule(m1, m2):
    """
    Computes the Dempster Rule between m1 and m2
    :param m1: The fisrt mass assinment function
    :param m2: The second mass assinment function
    :return: The combined mass assinment function
    """
    k = m1[1] * m2[2] + m1[2] * m2[1]
    if k == 1:  # Full uncertainty
        return 0, 0, 0, 1
    nk = 1.0 / (1 - k)
    mf_null = 0
    mf_no_stroke = nk * (m1[3] * m2[1] + m1[1] * m2[3] + m1[1] * m2[1])
    mf_stroke = nk * (m1[3] * m2[2] + m1[2] * m2[3] + m1[2] * m2[2])
    mf_either = nk * m1[3] * m2[3]
    return mf_null, mf_no_stroke, mf_stroke, mf_either


def dempster_rule_chain(*masses):
    """
    Computes the chained Dempster Rule of the given masses
    :param masses: Mass assignment functions
    :return: The combined mass assinment function
    """
    m = masses[0]
    for i in range(1, len(masses)):
        m = dempster_rule(m, masses[i])
    return m


def belief(m, cls):
    """
    Computes the belief of a class
    :param m: Mass assignment function
    :param cls: Class 0 or 1
    :return: Belief of the class
    """
    return m[cls + 1]


def plausability(m, cls):
    """
    Computes the plausability of a class
    :param m: Mass assignment function
    :param cls: Class 0 or 1
    :return: Plausability of the class
    """
    return 1 - m[2 - cls]


def cls_max(m, threshold=0):
    """
    Computes the class with maximum belief
    :param m: Mass assignment function
    :param threshold: Threshold (handicap) for the second class
    :return: Class with max belief
    """
    return 0 if m[1] > m[2] + threshold else 1


def cls_pla_max(m, threshold=0):
    """
    Computes the class with maximum plausibility
    :param m: Mass assignment function
    :param threshold: Threshold (handicap) for the second class
    :return: Class with max plausibility
    """
    return 0 if 1 - m[1] > 1 - m[2] + threshold else 1


def cls_score(m, threshold=0):
    """
    Computes the score of membership for the first class based on belief
    :param m: Mass assignment function
    :param threshold: Threshold (handicap) for the second class
    :return: Score of membership for the first class
    """
    return float(m[2])/(m[1] + m[2] + threshold)


def cls_pla_score(m, threshold=0):
    """
    Computes the score of membership for the first class based on plausibility
    :param m: Mass assignment function
    :param threshold: Threshold (handicap) for the second class
    :return: Score of membership for the first class
    """
    return float(1 - m[2] - m[3])/(m[1] + m[2] + threshold)


def create_full_uncertainty():
    """
    Returns the full uncertainty mass assignment
    :return: the full uncertainty vector
    """
    return 0, 0, 0, 1


def create_random_maf(uncertainty=0.8):
    """
    Creates a random mass assignment function with the given uncertainty
    :param uncertainty: mass value for uncertainty
    :return: mass assignment function vector
    """
    a = random.random() * (1 - uncertainty)
    return 0, a, 1 - a - uncertainty, uncertainty


def compute_gradfient_maf(ax, ay, rx, ry):
    """
    Computes the gradient of a single mass assignment function given the outputs
    :param ax: Mass of the first element
    :param ay: Mass of the second element
    :param rx: Output for the first element
    :param ry: Output for the second element
    :return: The grandient of the mean squared error with respect to ax, and ay
    """
    dJdax = -(rx - ax / (ax + ay)) * (1 / (ax + ay) - ax / (ax + ay) ** 2) + \
            ay * (ry - ay / (ax + ay)) / (ax + ay) ** 2
    dJday = -(ry - ay / (ax + ay)) * (1 / (ax + ay) - ay / (ax + ay) ** 2) + \
            ax * (rx - ax / (ax + ay)) / (ax + ay) ** 2
    return dJdax, dJday


def compute_gradient_dempster_rule(ax, ay, bx, by, rx, ry):
    """
    Computes the gradient of the dempster rule given the source masses and outputs
    :param ax: Mass of the first element of the first source
    :param ay: Mass of the second element of the first source
    :param bx: Mass of the first element of the second source
    :param by: Mass of the second element of the second source
    :param rx: Output for the first element
    :param ry: Output for the second element
    :return: The grandient of the mean squared error with respect to ax, ay, bx and by
    """
    dJdax = (rx - (ax * bx + ay * bx + ax * by - ax - bx) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ax * bx + ay * bx + ax * by - ax - bx) * (bx + 2 * by - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             (bx + by - 1) / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) + \
            (ry - (ay * bx + ax * by + ay * by - ay - by) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ay * bx + ax * by + ay * by - ay - by) * (bx + 2 * by - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             by / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by))
    dJday = (rx - (ax * bx + ay * bx + ax * by - ax - bx) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ax * bx + ay * bx + ax * by - ax - bx) * (2 * bx + by - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             bx / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) + \
            (ry - (ay * bx + ax * by + ay * by - ay - by) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ay * bx + ax * by + ay * by - ay - by) * (2 * bx + by - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             (bx + by - 1) / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by))
    dJdbx = (rx - (ax * bx + ay * bx + ax * by - ax - bx) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ax * bx + ay * bx + ax * by - ax - bx) * (ax + 2 * ay - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             (ax + ay - 1) / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) + \
            (ry - (ay * bx + ax * by + ay * by - ay - by) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ay * bx + ax * by + ay * by - ay - by) * (ax + 2 * ay - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             ay / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by))
    dJdby = (rx - (ax * bx + ay * bx + ax * by - ax - bx) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ax * bx + ay * bx + ax * by - ax - bx) * (2 * ax + ay - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             ax / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) + \
            (ry - (ay * bx + ax * by + ay * by - ay - by) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by)) * \
            ((ay * bx + ax * by + ay * by - ay - by) * (2 * ax + ay - 1) / (
            ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by) ** 2 -
             (ax + ay - 1) / (ax * bx + 2 * ay * bx + 2 * ax * by + ay * by - ax - ay - bx - by))
    return dJdax, dJday, dJdbx, dJdby


def dempster_rule_t(m1, m2):
    """
    Computes the Dempster Rule between m1 and m2
    :param m1: The fisrt mass assinment function
    :param m2: The second mass assinment function
    :return: The combined mass assinment function
    """
    k = m1[0] * m2[1] + m1[1] * m2[0]

    # if k == 1:  # Full uncertainty
    #    return torch.Tensor([0, 0, 1])
    nk = 1.0 / (1 - k)

    # mf_null = 0
    mf_no_stroke = nk * (m1[2] * m2[0] + m1[0] * m2[2] + m1[0] * m2[0])
    mf_stroke = nk * (m1[2] * m2[1] + m1[1] * m2[2] + m1[1] * m2[1])
    mf_either = nk * m1[2] * m2[2]

    return torch.cat([mf_no_stroke, mf_stroke, mf_either])


if __name__ == '__main__':
    m1 = (0, 0.1, 0.1, 0.8)
    m2 = (0, 0, 0.25, 0.75)
    print dempster_rule_chain(m2, m1)