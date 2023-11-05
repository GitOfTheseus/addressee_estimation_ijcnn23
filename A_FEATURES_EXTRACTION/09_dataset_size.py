import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bar_plot(data_list, slots_list):

    main_index = np.arange(10) + 1
    second_index = np.arange(-1, 2) / 5
    print(second_index)
    main_labels = slots_list
    second_labels = ['NAO', 'LEFT', 'RIGHT']
    colors = plt.cm.BuPu(np.linspace(0.2, 0.5, len(second_labels)))
    n_rows = len(second_labels)
    bar_width = 0.2
    cell_text = []

    for n, list in enumerate(data_list):
        print(n)
        print(second_index[n])
        print(list)
        plt.bar(main_index + second_index[n], list, bar_width, color=colors[n])
        cell_text.append(['{}'.format(x) for x in list])

    the_table = plt.table(cellText=cell_text,
                          rowLabels=second_labels,
                          rowColours=colors,
                          colLabels=main_labels,
                          loc='bottom')

    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.xticks([])

    plt.ylabel("Data for each class")
    plt.yticks()
    plt.title('Data in each slot')

    plt.show()


def count_data(csv_dir):
    print('read_data')
    df = pd.read_csv(csv_dir, sep="\t", index_col='index')
    n_nao = (len(df[df['ADDRESSEE'] == 'NAO'].index))
    n_pleft = (len(df[df['ADDRESSEE'] == 'PLEFT'].index))
    n_pright = (len(df[df['ADDRESSEE'] == 'PRIGHT'].index))
    tot = (len(df.index))

    return tot, n_nao, n_pleft, n_pright


def bar_plot_slots(dataset_path, slots_list):
    tot_list, nao_list, pleft_list, pright_list = np.array([]), np.array([]), np.array([]), np.array([])

    csv_file = 'lstm_label__.csv'
    for slot in slots_list:
        csv_dir = os.path.join(dataset_path, slot, csv_file)
        tot, n_nao, n_pleft, n_pright = count_data(csv_dir)
        tot_list = np.append(tot_list, tot)
        nao_list = np.append(nao_list, n_nao)
        pleft_list = np.append(pleft_list, n_pleft)
        pright_list = np.append(pright_list, n_pright)

    data_list = [nao_list, pleft_list, pright_list]
    bar_plot(data_list, slots_list)

    for n, tot in enumerate(tot_list):
        nao_list[n] = round(nao_list[n]/tot * 100, 1)
        pleft_list[n] = round(pleft_list[n] / tot * 100, 1)
        pright_list[n] = round(pright_list[n] / tot * 100, 1)

    data_list = [nao_list, pleft_list, pright_list]
    bar_plot(data_list, slots_list)

def bar_plot_training(dataset_path, slots_list):
    tot_list, nao_list, pleft_list, pright_list = np.array([]), np.array([]), np.array([]), np.array([])
    for slot in slots_list:
        csv_file = 'train_slot_{}_out.csv'.format(slot)
        csv_dir = os.path.join(dataset_path, csv_file)
        tot, n_nao, n_pleft, n_pright = count_data(csv_dir)
        tot_list = np.append(tot_list, tot)
        nao_list = np.append(nao_list, n_nao)
        pleft_list = np.append(pleft_list, n_pleft)
        pright_list = np.append(pright_list, n_pright)

    data_list = [nao_list, pleft_list, pright_list]
    bar_plot(data_list, slots_list)

    for n, tot in enumerate(tot_list):
        nao_list[n] = round(nao_list[n] / tot * 100, 1)
        pleft_list[n] = round(pleft_list[n] / tot * 100, 1)
        pright_list[n] = round(pright_list[n] / tot * 100, 1)

    data_list = [nao_list, pleft_list, pright_list]
    bar_plot(data_list, slots_list)


def main():
    dataset_path = '.../dataset_slots'
    slots_list = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    slots_list.sort()
    print(slots_list)
    bar_plot_slots(dataset_path, slots_list)


if __name__ == "__main__":
    main()
