import numpy as np
import seaborn as sn
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
from sklearn.metrics import confusion_matrix
from copy import deepcopy


def plot_2d_heatmap(data, is_confusion_matrix=False, title=None, x_label=None, x_ticks=None, y_label=None, y_ticks=None,
                    cmap="Spectral", data_format='.2f', fontsize=11, linewidth=0.5, show_cbar=True, figsize=None,
                    show_null_value=True, annot_cell=True, insert_summary=True, use_mean=True, show_percent=False, ):
    '''
    print 2D matrix
    Args:
        data: 2d list or array
        columns:
        cmap: Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fmt:
        cbar:
        show_null_value:
        annot_cell (bool): annotate values in the cells
        show_percent: False: only show digit; True: only show percent;
    Returns:

    '''
    
    def config_cell_4_confusion_matrix(data, position, oText, facecolors, fontsize, data_format='%.2f',
                                       show_null_value=0):
        # TODO have not been implemented yet
        """
            config cell text and colors
                and return text elements to add and to del
        """
        text_add = []
        text_del = []
        column, row = list(map(int, position))
        cell_val = data[row][column]
        tot_all = data[-1][-1]
        per = (float(cell_val) / tot_all) * 100
        curr_column = data[:, column]
        ccl = len(curr_column)
        
        # last row  and/or last column
        if (column == (ccl - 1)) or (row == (ccl - 1)):
            # tots and percents
            if (cell_val != 0):
                if (column == ccl - 1) and (row == ccl - 1):
                    tot_rig = 0
                    for i in range(data.shape[0] - 1):
                        tot_rig += data[i][i]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif (column == ccl - 1):
                    tot_rig = data[row][row]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif (row == ccl - 1):
                    tot_rig = data[column][column]
                    per_ok = (float(tot_rig) / cell_val) * 100
                per_err = 100 - per_ok
            else:
                per_ok = per_err = 0
            
            per_ok_s = ['%.2f%%' % (per_ok), '100%'][int(per_ok == 100)]
            
            # text to DEL
            text_del.append(oText)
            
            # text to ADD
            font_prop = fm.FontProperties(weight='bold', size=fontsize)
            text_kwargs = dict(color='black', ha="center", va="center", gid='sum', fontproperties=font_prop)
            lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy()
            dic['color'] = 'g'
            lis_kwa.append(dic)
            dic = text_kwargs.copy()
            dic['color'] = 'r'
            lis_kwa.append(dic)
            lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
            for i in range(len(lis_txt)):
                newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
                # print 'row: %s, column: %s, newText: %s' %(row, column, newText)
                text_add.append(newText)
            # print '\n'
            
            # set background color for sum cells (last row and last column)
            carr = [0.27, 0.30, 0.27, 1.0]
            if (column == ccl - 1) and (row == ccl - 1):
                carr = [0.17, 0.20, 0.17, 1.0]
            facecolors[posi] = carr
        
        else:
            if (per > 0):
                txt = '%s\n%.2f%%' % (cell_val, per)
            else:
                if (show_null_value == 0):
                    txt = ''
                elif (show_null_value == 1):
                    txt = '0'
                else:
                    txt = '0\n0.0%'
            oText.set_text(txt)
            
            # main diagonal
            if (column == row):
                # set color of the textin the diagonal to white
                oText.set_color('black')
                # set background color in the diagonal to blue
                facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
            else:
                oText.set_color('r')
        
        return text_add, text_del
    
    def config_cell(data, position, oText, fontsize, data_format, show_null_value=False, show_percent=False):
        """
            config cell text and colors
                and return text elements to add and to del
        """
        text_add = []
        text_del = []
        column, row = list(map(int, position))
        cell_val = data[row][column]
        
        text_del.append(oText)
        
        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fontsize)
        text_kwargs = dict(color='black', ha="center", va="center", gid='sum', fontproperties=font_prop)
        
        if show_percent:
            text_ls = ['', ] if ((show_null_value == False) and np.allclose(cell_val, 0)) \
                else [str(data_format + '%%') % (cell_val * 100.)]
            kwargs_ls = [text_kwargs]
            posi_ls = [(oText._x, oText._y)]
        else:
            text_ls = ['', ] if ((show_null_value == False) and (cell_val == 0)) else [data_format % cell_val, ]
            kwargs_ls = [text_kwargs] * len(text_ls)
            posi_ls = [(oText._x, oText._y)]
        
        for i in range(len(text_ls)):
            newText = dict(x=posi_ls[i][0], y=posi_ls[i][1], text=text_ls[i], kw=kwargs_ls[i])
            text_add.append(newText)
        
        return text_add, text_del
    
    def insert_totals(data_df, use_mean=False):
        """ insert summary row and column """
        
        if use_mean:
            data_df['Mean'] = data_df.mean(axis=1)
            avg_line = data_df.mean(axis=0)
            avg_line.name = 'Mean'
            data_df = data_df.append(avg_line, ignore_index=False)
        else:
            data_df['Sum'] = data_df.sum(axis=1)
            sum_line = data_df.sum(axis=0)
            sum_line.name = 'Sum'
            data_df = data_df.append(sum_line, ignore_index=False)
        
        return data_df
    
    data = np.array(data)
    x_ticks = map(str, range(data.shape[1])) if (x_ticks is None) else x_ticks
    y_ticks = map(str, range(data.shape[0])) if (y_ticks is None) else y_ticks
    data_df = DataFrame(data, index=y_ticks, columns=x_ticks)
    
    if insert_summary:
        data_df = insert_totals(data_df, use_mean=use_mean)
        data = np.asarray(data_df)
    
    # create plot
    fig = plt.figure(figsize)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing plot
    
    # annot_kws = {
    #     "size": fontsize
    # }
    ax = sn.heatmap(data_df, annot=annot_cell, linewidths=linewidth, ax=ax, cbar=show_cbar,  # annot_kws=annot_kws,
                    cmap=cmap, linecolor='w', )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)
    
    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()
    
    # iter in text elements
    text_add = []
    text_del = []
    for idx, text in enumerate(ax.collections[0].axes.texts):
        posi = np.array(text.get_position()) - [0.5, 0.5]
        
        if is_confusion_matrix:
            txt_res = config_cell_4_confusion_matrix(data=data, position=posi, oText=text, fontsize=fontsize,
                                                     data_format=data_format, show_null_value=show_null_value,
                                                     facecolors=facecolors, )
        else:
            txt_res = config_cell(data=data, position=posi, oText=text, fontsize=fontsize, data_format=data_format,
                                  show_null_value=show_null_value, show_percent=show_percent, )
        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])
    
    for item in text_del:
        item.remove()
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])
    
    if title is not None:
        ax.set_title('Confusion matrix')
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    plt.tight_layout()  # set layout slim
    plt.show()


def plot_confusion_matrix_from_y(y, y_pred, x_label=None, x_ticks=None, y_label=None, y_ticks=None,
                                 cmap="Spectral", data_format='%.2f', fontsize=10, linewidth=0.5, show_cbar=True,
                                 figsize=None, show_null_value=False, annot_cell=True, insert_summary=True,
                                 use_mean=True):
    # TODO not implemented completely yet
    """
        plot confusion matrix function with y and y_pred whitout a confusion matrix
    """
    
    c_matrix = confusion_matrix(y, y_pred)
    
    plot_2d_heatmap(data=c_matrix, is_confusion_matrix=False, x_label=x_label, x_ticks=x_ticks, y_label=y_label,
                    y_ticks=y_ticks, show_percent=True,
                    cmap=cmap, data_format=data_format, fontsize=fontsize, linewidth=linewidth, show_cbar=show_cbar,
                    figsize=figsize, show_null_value=show_null_value, annot_cell=annot_cell,
                    insert_summary=insert_summary, use_mean=use_mean)


if __name__ == '__main__':
    """ test function with y_true and y_test """
    y_true = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ])
    y_test = np.array([1, 2, 4, 3, 5, 1, 2, 4, 3, 5, 1, 2, 3, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, ])
    
    plot_confusion_matrix_from_y(y_true, y_test, use_mean=True)
