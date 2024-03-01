import numpy as np
import pandas as pd
import random


graph_path = 'D:/codeFile/matFile/matOne/graph/network_1281_106.dat'
a = pd.read_csv(graph_path, header = None, sep = '\t')
a.columns = ['u', 'v']
for i in [1,2,3,4,5]:
    print(a.shape)
    print(a.head())
    sample_index = random.sample(list(a.index), int(a.shape[0] / 6.0))
    sample_df = a.iloc[sample_index]

    node_num = int(a.max().max())
    max_node_df = a[a['u'] == node_num]
    max_node_df.index = range(max_node_df.shape[0])
    max_sample_index = random.sample(list(max_node_df.index), max(1, int(max_node_df.shape[0] / 6.0)) )
    max_sample_df = max_node_df.iloc[max_sample_index]

    sample_df_unique = sample_df.append(max_sample_df)
    sample_df_unique_reverse = pd.DataFrame()
    sample_df_unique_reverse['u'] = sample_df_unique['v'].values
    sample_df_unique_reverse['v'] = sample_df_unique['u'].values
    sample_df_unique = sample_df_unique.append(sample_df_unique_reverse)
    print(sample_df_unique.shape)
    sample_df_unique = sample_df_unique.drop_duplicates()
    print(sample_df_unique.shape)

    # sample_df.index = range(sample_df.shape[0])
    # print(sample_df.head())
    # sample_values = list(set(sample_df['u'].values) | set(sample_df['v'].values))
    # print(sample_values[:10])
    # sample_df_u = []
    # sample_df_v = []
    # for si in range(sample_df.shape[0]):
    #     u_value = sample_df.loc[si, 'u']
    #     u_index = sample_values.index(u_value)
    #     v_value = sample_df.loc[si, 'v']
    #     v_index = sample_values.index(v_value)
    #     sample_df_u.append(u_index+1)
    #     sample_df_v.append(v_index+1)
    # sample_df_unique = pd.DataFrame()
    # sample_df_unique['u'] = sample_df_u
    # sample_df_unique['v'] = sample_df_v
    # print(sample_df_unique.head())

    sample_df_unique.to_csv('D:/codeFile/matFile/matOne/graph/network_1281_106_sample_' + str(i) + '.dat', sep = '\t', header = None, index=False)
    print('\n')





# for node_degree in [2,3,4,5,6]:
#     a = pd.read_csv('D:/codeFile/pyFile/Hetero/pind/network_2000_' + str(node_degree) + '_me.dat', sep = '\t', header = None)
#     a.to_csv('D:/codeFile/pyFile/Hetero/pind/network_2000_' + str(node_degree) + '_0.15_200_0.3.txt', index = False, header = None, sep = '\t')



# for degrees in [[4],[3,4],[3,4,5],[2,3,4,5],[2,3,4,5,6]]:
#     a = pd.DataFrame()
#     c = pd.DataFrame()
#     node_degree_str = ''
#     for node_degree in degrees:
#         node_degree_str = node_degree_str + str(node_degree) + '_'
#         ba = pd.read_csv('D:/codeFile/pyFile/Hetero/pind/record_states_network_2000_' + str(node_degree) + '_me_0.15_300_0.3.txt', sep = '\t', header = None)
#         if a.shape[0] < 1:
#             a = ba.copy()
#         else:
#             a = a.append(ba)
#         print('a.shape = ' + str(a.shape) + ', ba.shape = ' + str(ba.shape))
#
#         bc = pd.read_csv('D:/codeFile/pyFile/Hetero/pind/record_times_network_2000_' + str(node_degree) + '_me_0.15_300_0.3.txt', sep = '\t', header = None)
#         if c.shape[0] < 1:
#             c = bc.copy()
#         else:
#             c = c.append(bc)
#         print('c.shape = ' + str(c.shape) + ', bc.shape = ' + str(bc.shape))
#
#     a.to_csv('D:/codeFile/pyFile/Hetero/pind/record_states_2000_' + node_degree_str + '300.txt', index = False, header = None, sep = '\t')
#     c.to_csv('D:/codeFile/pyFile/Hetero/pind/record_times_2000_' + node_degree_str + '300.txt', index = False, header = None, sep = '\t')



# cascade_name = r"record_states_200_4_200.txt"
# cascade_name = r"record_states_200_3_4_200.txt"
# cascade_name = r"record_states_200_3_4_5_200.txt"
# cascade_name = r"record_states_200_2_3_4_5_200.txt"
# cascade_name = r"record_states_200_2_3_4_5_6_200.txt"