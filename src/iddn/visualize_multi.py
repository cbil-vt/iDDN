"""Visualization of networks for multi-omics data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iddn import visualize_basic


def draw_multi_layer_network(
    cen_lst,
    rad_lst,
    mol_name,
    mol_grp=None,
    mol_label=None,
    mol_color=None,
    mol_size=None,
    mol_font_alpha=None,
    mol_font_color=None,
    edge_mol0=None,
    edge_mol1=None,
    edge_color=None,
    edge_weight=None,
    edge_style=None,
    fig_size=np.array((4, 2)),
    fig_scale=5,
    font_size_scale=1.5,
    out_name="",
):
    """Draw networks in multiple components

    For multi-omics data, it is often needed to use one component of an omics type.
    Sometimes it is also beneficial to further divide one omics type to more than one component for better view.
    This function provide the flexibility for these tasks. Each component can be an ellipse or line segment.
    Each node can have its own size and color. Each edge can have different color, weight, and style.

    Several examples can be found in the visualization tutorial (Tutorial 3) of iDDN package.

    We define `C` as the number of components. `M` is the number of nodes to show, `E` is the number of edges.

    Parameters
    ----------
    cen_lst : (C,2) array_like
        Each row is the x-y coordinate of the position of the center of each component.
        For details and more explanation, see Tutorial 3.
    rad_lst : (C,2) array_like
        Each row is the axes lengths of each component. For details, see tutorial 3.
    mol_name : (M) array_like
        The list of node names in the network. This is the internal name of each node.
        The edges must be described according to these names.
    mol_label : (M) array_like, optional
        The list of node labels to draw. This could be different from `node_name`.
        If not provided, we will display `mol_name` on the figure.
    mol_grp : (M) array_like, optional
        The component (grouping) membership of each node.
        The index of component is the same as in `cen_lst` and `rad_lst`.
    mol_color : (M) array_like, optional
        The color of each node. For eample, each element can be the name of color, like `black`.
    mol_size : (M) array_like, optional
        The size of each node. The value is relative to the standard size, which is automatically determined according
        to the canvas size and node number. Set to 1 will be the same as standard size, to 1.5 to get larger nodes, etc.
    mol_font_alpha : (M) array_like, optional
        The transparency of the text label of each node
    mol_font_color : (M) array_like, optional
        The color of the text label of each node
    edge_mol0 : (E) array_like
        For each edge, list the name of the first node
    edge_mol1 : (E) array_like
        For each edge, list the name of the second node
    edge_color : (E) array_like, optional
        The color of each edge
    edge_weight : (E) array_like, optional
        The weight of each edge, relative to the automatically determined standard value.
    edge_style : (E) array_like, optional
        The style of each edge, each element can be `solid` or `dashed`
    fig_size: (2) array_like, optional
        The size of the canvas. All distance related parameters appeared above should be based on this canvas size.
        See ``Tutorial 3`` for details.
    fig_scale: float or int
        The global scaling of the figure when printing or showing.
    font_size_scale : float or int
        The font size relative to the automatically determined standard font size.
        For example, use 1.5 to get a larger font.
    out_name : str, optional
        The name of the output file. Need to specify the file extension as well, like `pdf` or `png`.

    Returns
    -------
    None

    """
    if mol_size is None:
        mol_size = np.zeros(len(mol_name)) + 1  # relative size
    if mol_label is None:
        mol_label = mol_name
    if mol_grp is None:
        mol_grp = np.zeros(len(mol_name), dtype=int)
    if mol_color is None:
        mol_color = ["darkgrey" for _ in range(len(mol_name))]
    if mol_font_alpha is None:
        mol_font_alpha = np.ones(len(mol_name)) * 1.0
    if edge_mol0 is not None:
        if edge_style is None:
            edge_style = ["solid" for _ in range(len(edge_mol0))]
        if edge_color is None:
            edge_color = ["blue" for _ in range(len(edge_mol0))]
        if edge_weight is None:
            edge_weight = np.ones(len(edge_mol0)) * 0.5
    else:
        print("No edge will be drawn.")

    # Real size of the plot
    # The unit of figure is inch
    fig_size_print = fig_size * fig_scale
    cen_lst_print = cen_lst * fig_scale
    rad_lst_print = rad_lst * fig_scale
    n_node = len(mol_name)

    # Node radius according to figure size, circle size, and node number
    # The unit length is 1/72 inch
    node_radius_base_print = np.sum(np.array(rad_lst)) * 3 / n_node / 2 * fig_scale
    mol_font_size = np.zeros(n_node) + node_radius_base_print * 40 * font_size_scale
    node_df = pd.DataFrame(
        data=dict(
            grp=mol_grp,
            color=mol_color,
            size=mol_size,
            labels=mol_name,
            font_size=mol_font_size,
            font_alpha=mol_font_alpha,
            font_color=mol_font_color,
            x=np.zeros(n_node),
            y=np.zeros(n_node),
        ),
        index=mol_name,
    )

    # Get positions of each node
    mol_grp_dict = dict(zip(mol_name, mol_grp))
    mol_label_dict = dict(zip(mol_name, mol_label))
    pos, d_min = visualize_basic.get_pos_multi_parts(
        mol_name, mol_grp_dict, cen_lst, rad_lst
    )
    d_min_print = d_min * fig_scale
    pos_print = dict()
    for key, val in pos.items():
        pos_print[key] = val * fig_scale
    for key, val in pos_print.items():
        node_df.loc[key, "x"] = val[0]
        node_df.loc[key, "y"] = val[1]

    # Draw the network
    fig, ax = plt.subplots(figsize=fig_size_print)

    # Add edges
    if edge_mol0 is not None:
        for n in range(len(edge_mol0)):
            p0 = pos_print[edge_mol0[n]]
            p1 = pos_print[edge_mol1[n]]
            col = edge_color[n]
            wt = edge_weight[n]
            sty = edge_style[n]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=col,
                linewidth=wt,
                linestyle=sty,
            )

    # Add nodes after edges
    x = node_df["x"].to_numpy()
    y = node_df["y"].to_numpy()
    c = node_df["color"]
    s = (node_radius_base_print * 72) ** 2 * node_df["size"]
    ax.scatter(x=x, y=y, c=c, s=s)

    # Add labels to the figure
    visualize_basic.draw_network_labels(
        ax,
        pos_print,
        d_min_print,
        mol_grp_dict,
        cen_lst_print,
        rad_lst_print,
        mol_label_dict,
        node_df["font_size"].to_numpy(),
        node_df["font_alpha"].to_numpy(),
        node_df["font_color"].to_numpy(),
    )

    ax.set_xlim((-fig_size_print[0] / 2, fig_size_print[0] / 2))
    ax.set_ylim((-fig_size_print[1] / 2, fig_size_print[1] / 2))
    ax.axis("off")

    # Save the plot
    if len(out_name) > 0:
        plt.savefig(out_name, bbox_inches="tight")
