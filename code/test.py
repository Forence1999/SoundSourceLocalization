import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def hdf5_test():
    import h5py
    import numpy as np
    
    ''' example.h5 file structure
    +-- '/'
    |   +--	group "bar"
    |   |   +-- group "car"
    |   |   |   +-- None
    |   |   +-- dataset "dset"
    |   +-- dataset "dset"
    |   |   +-- attribute "Attr1"
    |   |   +-- attribute "Attr2"
    '''
    # Write HDF5 file.
    f = h5py.File("h5py_example.h5", "w")  # file.mode = {'w', 'r', 'a'}
    
    ds = f.create_dataset("dset", data=np.arange(16).reshape([4, 4]))
    ds.attrs["Attr1"] = [100, 200]
    ds.attrs["Attr2"] = "Hello, world!"
    
    group = f.create_group("bar")
    c = group.create_group("car")
    d = group.create_dataset("dset", data=np.arange(10))
    
    f.close()
    
    # Read HDF5 file.
    f = h5py.File("example.hdf5", "r")  # mode = {'w', 'r', 'a'}
    print(f.filename, ":")
    print([key for key in f.keys()])
    
    # Read dataset 'dset' under '/'.
    d = f["dset"]
    # Print the data of 'dset'.
    print(d.name, ":")
    print(d[:])
    # Print the attributes of dataset 'dset'.
    for key in d.attrs.keys():
        print(key, ":", d.attrs[key])
    
    # Read group 'bar'.
    g = f["bar"]
    # Print the keys of groups and datasets under group 'bar'.
    print([key for key in g.keys()])
    # Three methods to print the data of 'dset'.
    print(f["/bar/dset"][:])  # . absolute path
    print(f["bar"]["dset"][:])  # 2. relative path: file[][]
    print(g['dset'][:])  # 3. relative path: group[]
    
    # Delete a database.
    f = h5py.File("example.hdf5", "a")  # mode = {'w', 'r', 'a'}
    del g["dset"]
    
    # Save and exit the file
    f.close()


def create_multi_bars(data, color=None, title=None, x_labels=None, y_label=None, y_lim=None, tick_step=1.,
                      group_gap=0.2,
                      bar_gap=0., plt_show=True, value_show=True, dpi=300, value_fontsize=5, value_interval=0.01,
                      value_format='%.2f', save_path=None):
    '''
    x_labels: x轴坐标标签序列
    data: 二维列表，每一行为同一颜色的各个bar值，每一列为同一个横坐标的各个bar值
    tick_step: 默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap: 组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组中柱子间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    data = np.asarray(data)
    bar_per_group_num = len(data[0])
    if color is not None:
        assert len(color) >= bar_per_group_num
    x_ticks = np.arange(bar_per_group_num) * tick_step
    group_num = len(data)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num  # 组内每个bar的宽度
    bar_width = bar_span - bar_gap
    baseline_x = x_ticks - (group_width - bar_span) / 2  # baseline_x为每组柱子第一个柱子的基准x轴位置
    plt.figure(dpi=dpi)
    for index, y in enumerate(data):
        x = baseline_x + index * bar_span
        if color is not None:
            plt.bar(x, y, bar_width, color=color[index])
        else:
            plt.bar(x, y, bar_width)
        if value_show:
            for x0, y0 in zip(x, y):
                plt.text(x0, y0 + value_interval, value_format % y0, ha='center', va='bottom', fontsize=value_fontsize)
    if title is not None:
        plt.title(title)
    if x_labels is not None:
        plt.xticks(x_ticks, x_labels)
    if y_label is not None:
        plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    if save_path is not None:
        plt.savefig(save_path)
    if plt_show:
        plt.show()


if __name__ == "__main__":
    train_accs, val_accs, test_accs, train_acc3s, val_acc3s, test_acc3s = [], [], [], [], [], []
    
    for fold in range(6):
        res_path = '/home/swang/project/SmartWalker/SSL/model/ResCNN_' + str(
            fold) + '_256ms_norm_drop_denoised_norm_ini_hann_np_STFT_clip_ms_64_overlap_0.5_epoch_20/_norm_whole_label_None/res.npz'
        res = np.load(res_path)
        train_accs.append(res['train_accs'])
        val_accs.append(res['val_accs'])
        test_accs.append(res['test_accs'])
        train_acc3s.append(res['train_acc3s'])
        val_acc3s.append(res['val_acc3s'])
        test_acc3s.append(res['test_acc3s'])
    train_accs, val_accs, test_accs, train_acc3s, val_acc3s, test_acc3s = np.asarray(train_accs), np.asarray(
        val_accs), np.asarray(test_accs), np.asarray(train_acc3s), np.asarray(val_acc3s), np.asarray(test_acc3s)
    
    train_accs = np.concatenate((train_accs, np.mean(train_accs, axis=1).reshape((-1, 1))), axis=1)
    val_accs = np.concatenate((val_accs, np.mean(val_accs, axis=1).reshape((-1, 1))), axis=1)
    test_accs = np.concatenate((test_accs, np.mean(test_accs, axis=1).reshape((-1, 1))), axis=1)

    # print('train_accs: ', [round(i, 3) for i in train_accs], '-' * 5, round(np.mean(train_accs), 3))
    # print('val_accs: ', [round(i, 3) for i in val_accs], '-' * 5, round(np.mean(val_accs), 3))
    # print('test_accs: ', [round(i, 3) for i in test_accs], '-' * 5, round(np.mean(test_accs), 3))
    # print('train_accs: ', [round(i, 3) for i in train_acc3s], '-' * 5, round(np.mean(train_acc3s), 3))
    # print('val_accs: ', [round(i, 3) for i in val_acc3s], '-' * 5, round(np.mean(val_acc3s), 3))
    # print('test_accs: ', [round(i, 3) for i in test_acc3s], '-' * 5, round(np.mean(test_acc3s), 3))
    
    x_labels = ['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'Mean', ]
    create_multi_bars(train_accs, title='Fold-wise Train Acc of ResConv_i with Whole Norm', x_labels=x_labels,
                      y_label='acc', y_lim=(0, 1), tick_step=1, group_gap=0.2, bar_gap=0.05, save_path=None)
