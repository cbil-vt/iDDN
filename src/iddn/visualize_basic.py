"""Some shared basic visualization functions

1. Find positions of each node in each group (line or ellipse)
2. Add text outside the circle or line
"""

import math
import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist


def get_pos_multi_parts(
    nodes_all,
    nodes_type,
    cen_lst=np.array([[-0.6, 0], [0.6, 0]]),
    rad_lst=np.array([[0.4, 1], [0.4, 1]]),
):
    """Find positions of nodes with multiple isolated components or sub-graphs.

    Each subgraph is an ellipse or a line segment.
    Also provides the minimum distances between nodes.

    Let `K` be the number of components to draw.

    Parameters
    ----------
    nodes_all : list of str
        Node names
    nodes_type : dict
        Component/grouping name (or index) for each node
    cen_lst : array_like
        The center of each component. For component whose index is `i`, set cen_lst[i]
        Shape (k, 2), k is the number of types. 2 for (x, y)
    rad_lst : array_like
        The radius of each component. For component whose index is `i`, set rad_lst[i]
        Shape (k, 2), k is the number of types. 2 for shorter and longer axis length.

    Returns
    -------
    pos : dict
        The positions for each node
    d_min : float
        Minimum distances between nodes
    """
    # types of features
    fea_type_lst = []
    for node in nodes_type:
        fea_type_lst.append(nodes_type[node])
    fea_type_enu = np.unique(np.array(fea_type_lst))

    # create positions for each type of features
    pos = dict()
    d_min_lst = []
    cntt = 0
    for fea_id in fea_type_enu:
        nodes = []
        for node in nodes_all:
            if nodes_type[node] == fea_id:
                nodes.append(node)
        if rad_lst[cntt][0] > 0:
            pos, d_min = _get_pos_one_part(
                nodes, cen=cen_lst[cntt], rad=rad_lst[cntt], pos=pos
            )
        else:
            pos, d_min = _get_pos_one_part_line(
                nodes, cen=cen_lst[cntt], rad=float(rad_lst[cntt][1]), pos=pos
            )
        d_min_lst.append(d_min)
        cntt += 1

    # The minimum of minimum distances in all components
    d_min = np.min(np.array(d_min_lst))

    return pos, d_min


def _get_pos_one_part(
    nodes_show,
    cen=(0.0, 0.0),
    rad=(1.0, 1.0),
    pos=None,
):
    """Find positions of nodes on an ellipse

    Also provides the minimum distances between nodes.

    Parameters
    ----------
    nodes_show : tuple of str
        Node names
    cen : ndarray, optional
        The center of ellipse for each type of node. Shape (2, )
    rad : ndarray, optional
        The radius of ellipse for each type of node. Shape (2, )
    pos : dict
        NetworkX-like position dictionary

    Returns
    -------
    pos : dict
        The positions for each node
    d_min : float
        Minimum distances between nodes
    """
    n = len(nodes_show)
    r0 = float(rad[0])
    r1 = float(rad[1])
    if r0 == r1:
        r0 = r0 * 0.99999
    if r0 > r1:
        r0, r1 = r1, r0
    angle = _angles_in_ellipse(n, r0, r1)

    if pos is None:
        pos = dict()
    d_min = _add_node_to_a_circle(pos, nodes_show, cen, rad, angle)
    return pos, d_min


def _get_pos_one_part_line(
    nodes_show,
    cen=(0.0, 0.0),
    rad=1.0,
    pos=None,
):
    """Find positions of nodes on a line segment

    Also provides the minimum distances between nodes.

    Parameters
    ----------
    nodes_show : tuple of str
        Node names
    cen : ndarray
        The center of ellipse for each type of node. Shape (2, )
    rad : float or int
        The half-length of the line segment
    pos : dict
        NetworkX-like position dictionary

    Returns
    -------
    pos : dict
        The positions for each node
    d_min : float
        Minimum distances between nodes
    """
    n = len(nodes_show)
    if pos is None:
        pos = dict()
    pos_y = np.linspace(-rad, rad, n, endpoint=True)
    for i, node in enumerate(nodes_show):
        pos[node] = np.array([cen[0], cen[1] + pos_y[i]])
    if len(pos_y) > 1:
        d_min = pos_y[1] - pos_y[0]
        # Prevent nodes from being too large or too far away
        if d_min > 0.1:
            d_min = 0.1
    else:
        d_min = 0.1
    return pos, d_min


def draw_network_labels(
    ax,
    pos,
    d_min,
    node_type,
    cen_lst,
    rad_lst,
    labels,
    font_size_lst,
    font_alpha_lst,
    font_col_lst,
):
    """Add labels to a graph

    Modified from Networkx add label function

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes of the figure
    pos : dict
        Node position dictionary
    d_min : float
        Minimum distance between nodes
    node_type : dict
        Feature type index of each node
    cen_lst : array_like
        The center of each component.
    rad_lst : array_like
        The radius of each component.
    labels : dict
        Alternative names for each node, to be shown in the graph
    font_size_lst : array_like
        Font size for each node, in points
    font_alpha_lst : array_like
        Font alpha for each node, value from 0 to 1
    font_col_lst : array_like
        Font color for each node
    """
    cnt = 0
    for node, label in labels.items():
        (x, y) = pos[node]
        fea_idx = node_type[node]
        x1 = x - cen_lst[fea_idx][0]
        y1 = y - cen_lst[fea_idx][1]

        # this makes "1" and 1 labeled the same
        if not isinstance(label, str):
            label = str(label)

        # rotate labels to make it point to the center
        # this allows larger fonts
        # treat labels on the right and those on the left differently
        if rad_lst[fea_idx][0] > 0:
            angle = math.atan2(y1, np.abs(x1)) / math.pi * 180
            if x1 < 0:
                angle = -angle

            nn = np.sqrt(x1**2 + y1**2)
            x1n = x1 / nn
            y1n = y1 / nn

            x_ext = x + x1n * (len(label) / 2 * font_size_lst[cnt] / 72 + d_min / 4 * 2)
            y_ext = y + y1n * (len(label) / 2 * font_size_lst[cnt] / 72 + d_min / 4 * 2)

            _ = ax.text(
                x_ext,
                y_ext,
                label,
                size=font_size_lst[cnt],
                color=font_col_lst[cnt],
                family="sans-serif",
                weight="normal",
                alpha=font_alpha_lst[cnt],
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transData,
                rotation=angle,
                bbox=None,
                clip_on=True,
            )
        else:
            # angle = 0
            x_ext = x + d_min / 4 * 2
            # x_ext = x + len(label) / 2 * font_size_lst[cnt] / 80 + d_min / 4 * 2
            y_ext = y

            _ = ax.text(
                x_ext,
                y_ext,
                label,
                size=font_size_lst[cnt],
                color=font_col_lst[cnt],
                family="sans-serif",
                weight="normal",
                alpha=font_alpha_lst[cnt],
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transData,
                rotation=0,
                bbox=None,
                clip_on=True,
            )
        cnt += 1

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )


def _add_node_to_a_circle(pos, nodes, cen, rad, angles):
    """Find positions in an ellipse, and calculate the minimum distances between points

    Parameters
    ----------
    pos : dict
        For saving the position results for NetworkX
    nodes : list of str
        Name of nodes
    cen : array_like
        Central position of this circle, shape (2,)
    rad : array_like
        Length of two axes of the ellipse, shape (2,)
    angles : array_like
        Angles of the points

    Returns
    -------
    float
        Minimum distances between points
    """
    pos_lst = []
    cnt = 0
    for node in nodes:
        theta = angles[cnt]
        pos0 = np.array(
            [cen[0] + np.cos(theta) * rad[0], cen[1] + np.sin(theta) * rad[1]]
        )
        pos[node] = pos0
        pos_lst.append(pos0)
        cnt += 1
    pos_lst = np.array(pos_lst)
    if len(pos_lst) > 1:
        d = pdist(pos_lst)
        return np.min(d)
    else:
        return 0.5


def _angles_in_ellipse(num, a, b):
    """Calculate angles of evenly spaced points in an ellipse

    Based on https://stackoverflow.com/a/52062369,
    which is from https://pypi.org/project/flyingcircus/

    Parameters
    ----------
    num : int
        Sample number to get
    a : float
        Length of shorter axis
    b : float
        Length of longer axis

    Returns
    -------
    angles : ndarray
        Angles of sampled points
    """
    assert num > 0
    assert a < b
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e2 = 1.0 - a**2.0 / b**2.0
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e2)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(lambda x: (sp.special.ellipeinc(x, e2) - arcs), angles)
        angles = res.x
    return angles
