import random
import time

import networkx as nx
import numpy as np
from absl import app, flags
from scipy.stats import expon

# flags
FLAGS = flags.FLAGS
flags.DEFINE_string ('aug_strategy', 'AugSIM', "Augmentation strategy, 'AugSIM' or 'AugRWR'.")
flags.DEFINE_float  ('aug_strength', 0.1, 'Augmentation strength eta.')
flags.DEFINE_float  ('gamma', 3, 'Random walk with restart walking steps parameter gamma.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_float  ('restart_prob', 0.2, 'Random walk with restart probability.')
flags.DEFINE_integer('t_o', 3600, 'Observation time, default: 3600s for Weibo.')
flags.DEFINE_float  ('theta', 0.1, 'Weight parameter theta.')

# paths
flags.DEFINE_string ('input', './datasets/weibo/', 'Dataset path.')


def save_file(id, paths, file):
    # save a cascade into a line of a file
    file.write(str(id) + '\t' + '\t'.join(paths) + '\n')


def augmentor_rwr(input_file, output_file, restart_prob=.2, gamma=2):
    # a reference implementation of AugRWR

    # open two files, one for read and another for write
    with open(input_file, 'r') as f, \
            open(output_file, 'w') as save_f:
        for line in f:
            # create a nx graph
            g = nx.Graph()
            user_dict = dict()
            paths = line.strip().split('\t')
            # remove cascade label
            paths = paths[:-1]
            observation_path = [paths[1]]
            cascade_id = paths[0]
            root_path = paths[1]
            root_user = root_path.split(':')[0]
            if root_user == '-1':
                continue
            user_dict[root_user] = root_path
            num_ori_nodes = 0
            for path in paths[2:FLAGS.max_seq+2]:
                nodes = path.split(':')[0].split(',')
                user_dict[nodes[-1]] = path
                user_dict[nodes[-2]] = path
                g.add_edge(nodes[-1], nodes[-2])
                num_ori_nodes += 1
            num_steps = int(gamma * g.number_of_nodes())
            cur_node = root_user
            sampled_node_set = {cur_node}
            for step in range(num_steps):
                if len(sampled_node_set) == g.number_of_nodes():
                    break
                if random.random() > restart_prob:
                    cur_node = random.choices(list(g[cur_node]),
                                              weights=[degree for node, degree in nx.degree(g, g[cur_node])])[0]
                else:
                    cur_node = root_user
                sampled_node_set.add(cur_node)

            # save walked nodes into list
            for node in sampled_node_set:
                observation_path.append(user_dict[node])

            # sort nodes by their publish time/date
            observation_path.sort(key=lambda tup: int(tup.split(':')[1]))
            save_file(cascade_id, observation_path, save_f)
            num_cur_nodes = len(observation_path)

            # uncomment the following code to print augmentation details
            # print('{:7}, Number of nodes (now/ori): {:3}/{:3}'.format(cascade_id, num_cur_nodes, num_ori_nodes))


def augmentor_sim(cascade_file, save_aug_file):
    # a reference implementation of AugSIM

    def calculate_global_time(path):
        # calculate global_time among all adoption times/dates in a dataset
        all_ts = list()
        i = 0
        with open(path, 'r') as f:
            for line in f:
                i += 1
                last_t = 0
                paths = line.strip().split('\t')
                # remove cascade id and label
                paths = paths[2:-1]
                for path in paths:
                    t = int(path.split(':')[1])
                    reaction_t = t - last_t
                    last_t = t
                    all_ts.append(reaction_t)

        return np.mean(all_ts), i, expon.fit(all_ts)

    def calculate_local_time(line):
        # calculate local_time among adoption times/dates in one cascade
        last_t = 0
        cascade_t = list()
        paths = line.strip().split('\t')
        # remove cascade id and label
        paths = paths[2:-1]
        for path in paths:
            t = int(path.split(':')[1])
            reaction_t = t - last_t
            last_t = t
            cascade_t.append(reaction_t)

        return np.mean(cascade_t)

    def node_degree(line):
        # calculate node degrees and number of leaf nodes in a cascade graph

        g = nx.Graph()
        # remove cascade id and label
        paths = line.strip().split('\t')[2:-1]
        for path in paths:
            nodes = path.split(':')[0].split(',')
            g.add_edge(nodes[-1], nodes[-2])

        degree = {node: g.degree(node) for node in g.nodes()}
        all_degree = sum([g.degree(node) for node in g.nodes()])
        num_leaf_nodes = sum([1 for node, degree in g.degree if len(list(nx.neighbors(g, node))) == 1])

        return degree, all_degree, num_leaf_nodes

    def parent_node_degree(observation_path):
        # calculate parent node degrees and number of added leaf nodes

        g = nx.Graph()
        for path in observation_path[1:-1]:
            nodes = path.split(':')[0].split(',')
            g.add_edge(nodes[-1], nodes[-2])
        parent_degree = {node: nx.degree(g, list(nx.neighbors(g, node))[0])
                         for node, degree in g.degree if len(list(nx.neighbors(g, node))) == 1}
        all_parent_degree = sum([nx.degree(g, list(nx.neighbors(g, node))[0])
                                 for node, degree in g.degree if len(list(nx.neighbors(g, node))) == 1])
        num_added_leaf_nodes = sum([1 for node, degree in g.degree if len(list(nx.neighbors(g, node))) == 1])

        return parent_degree, all_parent_degree, num_added_leaf_nodes

    mean_global_t, num_samples, (loc, scale) = calculate_global_time(cascade_file)
    # print('Mean Global Time:', mean_global_t)

    # open two files, one for read and another for write
    with open(cascade_file, 'r') as f, \
            open(save_aug_file, 'w') as save_f:
        i = 0
        for line in f:
            mean_local_t = calculate_local_time(line)
            degree, all_degree, num_leaf_nodes = node_degree(line)
            added_nodes = list()
            # remove cascade label
            paths = line.strip().split('\t')[:-1][:FLAGS.max_seq+1]
            num_ori_nodes = len(paths)-1
            eta = FLAGS.aug_strength * num_ori_nodes
            cascade_id = int(paths[0])
            observation_path = [paths[1]]
            added_node_idx = 1

            # add nodes
            for path in paths[2:]:
                nodes = path.split(':')[0].split(',')
                cur_node = nodes[-1]
                observation_path.append(path)
                add_node_prob = eta * (degree[cur_node] / all_degree)
                if random.random() > add_node_prob:
                    continue
                t = int(path.split(':')[1]) + FLAGS.theta * mean_local_t + (1-FLAGS.theta) * expon.rvs(loc, scale, size=1)
                nodes.append('-' + str(added_node_idx))
                added_node_idx += 1
                if t > FLAGS.t_o:
                    t = FLAGS.t_o
                added_nodes.append(','.join(nodes) + ':' + str(int(t)))
            observation_path.extend(added_nodes)
            num_added_nodes = len(observation_path)

            # delete nodes
            parent_degree, all_parent_degree, num_added_leaf_nodes = parent_node_degree(observation_path)
            eta = FLAGS.aug_strength * num_added_leaf_nodes

            if num_leaf_nodes != 0:
                for path in observation_path[1:]:
                    nodes = path.split(':')[0].split(',')
                    cur_node = nodes[-1]
                    try:
                        del_node_prob = eta * (parent_degree[cur_node] / all_parent_degree)
                    except KeyError:
                        continue
                    # is_leaf = is_leaf_node(observation_path, nodes)
                    if random.random() < del_node_prob:
                        observation_path.remove(path)

            observation_path.sort(key=lambda tup: int(tup.split(':')[1]))
            num_del_nodes = len(observation_path)

            # write cascades into file
            save_file(cascade_id, observation_path, save_f)
            i += 1

            # uncomment the following code to print augmentation details
            # print('{}/{}, ori/add/del/minus: {:4}, {:4}, {:4}, {:4}.'.format(i, num_samples,
            #                                                                  num_ori_nodes,
            #                                                                  num_added_nodes-num_ori_nodes,
            #                                                                  num_added_nodes-num_del_nodes,
            #                                                                  num_del_nodes-num_ori_nodes))


def main(argv):
    time_start = time.time()
    print('Start to augment cascade data!')
    if FLAGS.aug_strategy == 'AugSIM':
        print('Augmentation strategy: AugSIM')
        augmentor_sim(FLAGS.input + 'train.txt', FLAGS.input + 'aug_1.txt')
        print('1/4')
        augmentor_sim(FLAGS.input + 'train.txt', FLAGS.input + 'aug_2.txt')
        print('2/4')
        augmentor_sim(FLAGS.input + 'unlabel.txt', FLAGS.input + 'unlabel_aug_1.txt')
        print('3/4')
        augmentor_sim(FLAGS.input + 'unlabel.txt', FLAGS.input + 'unlabel_aug_2.txt')
        print('4/4\nFinished!')
    elif FLAGS.aug_strategy == 'AugRWR':
        print('Augmentation strategy: AugRWR')
        augmentor_rwr(FLAGS.input + 'train.txt', FLAGS.input + 'aug_1.txt',
                      FLAGS.restart_prob, FLAGS.gamma)
        print('1/4')
        augmentor_rwr(FLAGS.input + 'train.txt', FLAGS.input + 'aug_2.txt',
                      FLAGS.restart_prob, FLAGS.gamma)
        print('2/4')
        augmentor_rwr(FLAGS.input + 'unlabel.txt', FLAGS.input + 'unlabel_aug_1.txt',
                      FLAGS.restart_prob, FLAGS.gamma)
        print('3/4')
        augmentor_rwr(FLAGS.input + 'unlabel.txt', FLAGS.input + 'unlabel_aug_2.txt',
                      FLAGS.restart_prob, FLAGS.gamma)
        print('4/4\nFinished!')
    else:
        print('Specified augmentation strategy doesn\'t exist.')

    time_end = time.time()
    print('Processing Time: {:.2f}s'.format(time_end - time_start))


if __name__ == "__main__":
    app.run(main)
