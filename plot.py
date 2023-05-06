import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    smoothed_points = np.array(smoothed_points)
    return smoothed_points


def get_plotxy(model_name, x, y, max_epoch, smoothed=0.9, origin=False):
    plot_y = []
    for i in range(min(max_epoch, len(y))):
        plot_y.append(y[i])

    plot_x = [i for i in range(len(plot_y))]
    plot_x = np.array(plot_x)
    plot_y = np.array(plot_y)
    # original
    if origin:
        return plot_x, plot_y
    # smoothed
    plot_y = smooth_curve(plot_y, factor=smoothed)

    return plot_x, plot_y


def set_plt(x_tick, y_tick):
    # ax = plt.gca()
    # milocx = plt.MultipleLocator(10)
    # ax.xaxis.set_minor_locator(milocx)
    # milocy = plt.MultipleLocator(0.1)
    # ax.yaxis.set_minor_locator(milocy)

    # ax.xaxis.set_ticks_position('bottom')
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['left'].set_color('gray')
    # ax.spines['bottom'].set_color('gray')
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')

    # set background color
    # ax.patch.set_facecolor('gray')
    # ax.patch.set_alpha(0.1)

    # set axis numbers
    x_ticks = (np.arange(0, x_tick + 10, 20), np.arange(0, (x_tick + 1) * 500, 20 * 500))
    plt.xticks(x_ticks[0], x_ticks[1])
    y_ticks = np.arange(0, y_tick, 0.1)
    plt.yticks(y_ticks)


def plot_loss(max_epoch, smoothed=0.8, linewidth=1, origin=True):
    base_path = './plot'
    model_names = ["3D CNN", "3D ResNet"]

    for model_name in model_names:
        plt.figure()
        for mod in ["_Train", "_Valid"]:
            csv_name = os.path.join(base_path, model_name, 'run-Compare_Loss' + mod + '_Loss-tag-Compare_Loss.csv')
            if os.path.exists(csv_name):
                data = pd.read_csv(csv_name)
            else:
                print(f'Warning: {csv_name} not exist')

            # original
            if origin:
                plot_x, plot_y = get_plotxy(model_name=model_name, x=data['Step'], y=data['Value'],
                                            max_epoch=max_epoch, smoothed=smoothed, origin=origin)
                plt.plot(plot_x, plot_y, linewidth=linewidth, alpha=0.2, label=model_name + mod + '_original')
            # smoothed
            plot_x, plot_y = get_plotxy(model_name=model_name, x=data['Step'], y=data['Value'],
                                        max_epoch=max_epoch, smoothed=smoothed)
            plt.plot(plot_x, plot_y, linewidth=linewidth, label=model_name + mod + '_smoothed')

        set_plt(x_tick=max_epoch, y_tick=1.1)

        # set plot region
        plt.xlim(0, max_epoch + 10)
        plt.ylim(0, 1)

        # set axis name
        plt.xlabel('Step', fontdict={'weight': 'normal'}, fontsize=12, labelpad=6)
        plt.ylabel('Loss', fontdict={'weight': 'normal'}, fontsize=12, labelpad=6)

        # set grid
        plt.grid(linewidth=0.7, color='gray', alpha=0.3, which='both', linestyle='--')
        plt.legend()

        # set plot size
        ax = plt.gcf()
        ax.set_size_inches(8, 8)
        plt.savefig(os.path.join(base_path, 'result', model_name + '_Loss.png'), bbox_inches='tight', pad_inches=0.0,
                    transparent=True)
        print(f'Saved to {model_name}_Loss.png')
        plt.close()
        # plt.show()


def plot_metrics(max_epoch, smoothed=0.8, linewidth=1, origin=True):
    base_path = './plot'
    model_names = ["3D CNN", "3D ResNet"]
    metrics = ['Accuracy', 'F1', 'Loss', 'Precision', 'Recall']

    # record the max/min value for every metric for every model
    metrics_model = [[i] for i in model_names]

    for metric in metrics:
        plt.figure()
        for model_name in model_names:
            csv_name = os.path.join(base_path, model_name, 'run-.-tag-Validation_' + metric + '.csv')
            if os.path.exists(csv_name):
                data = pd.read_csv(csv_name)
            else:
                print(f'Warning: {csv_name} not exist')

            # original
            if origin:
                plot_x, plot_y = get_plotxy(model_name=model_name, x=data['Step'], y=data['Value'],
                                            max_epoch=max_epoch, smoothed=smoothed, origin=origin)
                plt.plot(plot_x, plot_y, linewidth=linewidth, alpha=0.2, label=model_name + '_original')
            # smoothed
            plot_x, plot_y = get_plotxy(model_name=model_name, x=data['Step'], y=data['Value'],
                                        max_epoch=max_epoch, smoothed=smoothed)
            plt.plot(plot_x, plot_y, linewidth=linewidth, label=model_name + '_smoothed')

            # mark the max/min value
            if metric == 'Loss':
                min_xy = np.argmin(plot_y[5:]) + 5, np.min(plot_y[5:])
                plt.plot(*min_xy, marker='o', markersize=5)
                plt.annotate(round(min_xy[1], 4), xy=min_xy, xytext=(min_xy[0] - 3, min_xy[1] + 0.01))
            else:
                max_xy = np.argmax(plot_y[5:]) + 5, np.max(plot_y[5:])
                plt.plot(*max_xy, marker='o', markersize=5)
                plt.annotate(round(max_xy[1], 4), xy=max_xy, xytext=(max_xy[0] - 3, max_xy[1] + 0.01))

            # record the max/min value
            if metric == 'Loss':
                metrics_model[model_names.index(model_name)].append(min(plot_y))
            else:
                metrics_model[model_names.index(model_name)].append(max(plot_y))

        set_plt(x_tick=max_epoch, y_tick=1.1)

        # set plot region
        plt.xlim(0, max_epoch + 10)
        plt.ylim(0, 1)

        # set axis name
        plt.xlabel('Step', fontdict={'weight': 'normal'}, fontsize=12, labelpad=6)
        plt.ylabel(metric, fontdict={'weight': 'normal'}, fontsize=12, labelpad=6)
        # plt.suptitle(metric, fontdict={'family': 'Times New Roman'}, fontsize=25)

        # set grid
        plt.grid(linewidth=0.7, color='gray', alpha=0.3, which='both', linestyle='--')
        plt.legend()

        # set plot size
        ax = plt.gcf()
        # ax.set_size_inches(12, 5.25)
        ax.set_size_inches(8, 8)
        plt.savefig(os.path.join(base_path, 'result', metric + '.png'), bbox_inches='tight', pad_inches=0.0,
                    transparent=True)
        print(f'Saved to {metric}.png')
        plt.close()
        # plt.show()

    # write to csv
    data = pd.DataFrame(metrics_model, columns=(['model_name'] + metrics))
    data.to_csv(base_path + '/result/metrics_model.csv')


if __name__ == '__main__':
    # set xtick and ytick direction:in, out, inout
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # plot curve
    os.makedirs('./plot/result', exist_ok=True)
    plot_metrics(max_epoch=120)
    plot_loss(max_epoch=120)
