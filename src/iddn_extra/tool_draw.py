import matplotlib.pyplot as plt


def plot_res(
    res_comm, res_diff, curve_type="roc", color="C0", ax=None, alpha=1.0, linewidth=1
):
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    if curve_type == "roc":
        ax[0, 0].plot(
            res_comm[:, 3],
            res_comm[:, 2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax[1, 0].plot(
            res_diff[:, 3],
            res_diff[:, 2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
    else:
        ax[0, 0].plot(
            res_comm[:, 0],
            res_comm[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax[1, 0].plot(
            res_diff[:, 0],
            res_diff[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    ax[0, 1].plot(
        res_comm[:, 2], res_comm[:, 4], color=color, alpha=alpha, linewidth=linewidth
    )
    ax[1, 1].plot(
        res_diff[:, 2], res_diff[:, 4], color=color, alpha=alpha, linewidth=linewidth
    )


def plot_res_multi_approach(res_comm_lst, res_diff_lst, ylim0=(0, 50), ylim1=(0, 50)):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(len(res_comm_lst)):
        res_comm = res_comm_lst[i]
        res_diff = res_diff_lst[i]
        if i == 0:
            plot_res(res_comm, res_diff, color="orange", ax=ax, alpha=1, linewidth=2)
            # print(res_diff[:,2])
            # print(res_diff[:,4])
        elif i == len(res_comm_lst) - 1:
            plot_res(res_comm, res_diff, color="green", ax=ax, alpha=1, linewidth=2)
            # print(res_diff[:,2])
            # print(res_diff[:,4])
        else:
            plot_res(res_comm, res_diff, color="blue", ax=ax, alpha=0.25)
            # plot_res(res_comm, res_diff, color="blue", ax=ax, alpha=1-i/len(res_comm_lst))
    ax[0, 0].set_ylim(ylim0)
    ax[1, 0].set_ylim(ylim1)
    # ax[0, 1].set_xlim([0.1,1.1])
    # ax[1, 1].set_xlim([0.1,1.1])


def plot_res_multi(res_comm_lst, res_diff_lst, curve_type="roc"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for i in range(len(res_comm_lst)):
        res_comm = res_comm_lst[i]
        res_diff = res_diff_lst[i]
        plot_res(
            res_comm,
            res_diff,
            curve_type=curve_type,
            color=f"C{i}",
            ax=ax,
            alpha=1,
            linewidth=2,
        )
    return ax


def draw_lines_rho1_rho2(
    res_comm_ddn_mean, res_comm_jgl_mean, rho2_rg=None, title_mean="Common network"
):
    if rho2_rg is None:
        n_rho2 = res_comm_ddn_mean.shape[1]
        rho2_rg = range(n_rho2)

    plt.figure()
    for i in rho2_rg:
        plt.plot(
            res_comm_ddn_mean[:, i, 2],
            res_comm_ddn_mean[:, i, 4],
            "-o",
            color="blue",
            markersize=2,
            linewidth=0.25,
        )
        plt.plot(
            res_comm_jgl_mean[:, i, 2],
            res_comm_jgl_mean[:, i, 4],
            "-o",
            color="orange",
            markersize=2,
            linewidth=0.25,
        )
        # plt.plot(
        #     res_comm_iddn_mean[:, i, 2],
        #     res_comm_iddn_mean[:, i, 4],
        #     "-o",
        #     color="green",
        #     markersize=2,
        #     linewidth=0.25,
        # )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title(title_mean)
    plt.show()
