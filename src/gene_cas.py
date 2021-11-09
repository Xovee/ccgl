import random
import time

from absl import app, flags


# flags
FLAGS = flags.FLAGS
"""
Dataset | Observation Time           | Prediction Time               |
---------------------------------------------------------------------|
weibo   | 3600 (1 hour)              | 3600*24 (86400, 1 day)        |
twitter | 3600*24*2 (172800, 2 days) | 3600*24*32 (2764800, 32 days) |
acm     | 3 (years)                  | 10 (years)                    |
aps     | 365*3 (1095, 3 years)      | 365*20+5 (7305, 20 years)     |
dblp    | 5 (years)                  | 20 (years)                    |
"""
flags.DEFINE_integer('observation_time', 3600, 'Observation time.')
flags.DEFINE_integer('prediction_time', 3600*24, 'Prediction time.')
flags.DEFINE_boolean('unlabel', False, 'Generate unlabeled data.')

# paths
flags.DEFINE_string ('input', './datasets/weibo/', 'Dataset path.')


def generate_cascades(observation_time, prediction_time, unlabel,
                      filename, file_train, file_val, file_test, file_unlabel, seed='xovee'):

    # a list to save the cascades
    filtered_data = list()
    with open(filename) as file:

        cascades_type = dict()  # 0 for train, 1 for val, 2 for test
        cascades_time_dict = dict()
        cascade_total = 0
        cascade_valid_total = 0

        for line in file:
            # split the cascades into 5 parts
            # 1: cascade id
            # 2: user/item id
            # 3: publish date/time
            # 4: number of adoptions
            # 5: a list of adoptions
            cascade_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]

            # filter cascades by their publish date/time
            # if criterion satisfied, put cascades into labeled set, otherwise unlabeled set
            if not unlabel:
                if 'weibo' in FLAGS.input:
                    # timezone invariant
                    hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8
                    if hour < 8 or hour >= 18:
                        continue
                elif 'twitter' in FLAGS.input:
                    month = int(time.strftime('%m', time.localtime(float(parts[2]))))
                    day = int(time.strftime('%d', time.localtime(float(parts[2]))))
                    if month == 4 and day > 10:
                        continue
                elif 'acm' in FLAGS.input:
                    year = parts[2]
                    if year > '2006':
                        continue
                elif 'aps' in FLAGS.input:
                    date = parts[2]
                    if date > '1997-12':
                        continue
                elif 'dblp' in FLAGS.input:
                    year = parts[2]
                    if year > '1997':
                        continue
                else:
                    print('Wow, a new dataset!')
            else:
                if 'weibo' in FLAGS.input:
                    # timezone invariant
                    hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8
                    obt = observation_time // 3600
                    if hour < 18 or hour >= 24 - obt:
                        continue
                elif 'twitter' in FLAGS.input:
                    month = int(time.strftime('%m', time.localtime(float(parts[2]))))
                    day = int(time.strftime('%d', time.localtime(float(parts[2]))))
                    if month == 4 and day < 10:
                        continue
                    if month == 3:
                        continue
                elif 'acm' in FLAGS.input:
                    year = parts[2]
                    if year <= '2006':
                        continue
                elif 'aps' in FLAGS.input:
                    date = parts[2]
                    if date <= '1997-12':
                        continue
                elif 'dblp' in FLAGS.input:
                    year = parts[2]
                    if year <= '1997':
                        continue
                else:
                    print('Wow, a new dataset!')

            # a list of adoptions
            paths = parts[4].strip().split(' ')

            observation_path = list()
            # number of observed popularity
            p_o = 0
            for p in paths:
                # observed adoption/participant
                nodes = p.split(':')[0].split('/')
                time_now = int(p.split(':')[1])
                if time_now < observation_time:
                    p_o += 1
                # save observed adoption/participant into 'observation_path'
                observation_path.append((nodes, time_now))

            # filter cascades which observed popularity less than 5 or 10
            if 'dblp' in FLAGS.input:
                if p_o < 5:
                    continue
            else:
                if p_o < 10:
                    continue

            # sort list by their publish time/date
            observation_path.sort(key=lambda tup: tup[1])

            # for each cascade, save its publish time into a dict
            if 'aps' in FLAGS.input:
                cascades_time_dict[cascade_id] = int(0)
            else:
                cascades_time_dict[cascade_id] = int(parts[2])

            o_path = list()

            for i in range(len(observation_path)):
                nodes = observation_path[i][0]
                t = observation_path[i][1]
                o_path.append('/'.join(nodes) + ':' + str(t))

            # write data into the targeted file, if they are not excluded

            line = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' \
                   + parts[3] + '\t' + ' '.join(o_path) + '\n'
            filtered_data.append(line)
            cascade_valid_total += 1

    if not unlabel:
        # open three files to save train, val, and test set, respectively.
        with open(file_train, 'w') as data_train, \
                open(file_val, 'w') as data_val, \
                open(file_test, 'w') as data_test:

            def shuffle_cascades():
                # shuffle all cascades
                shuffled_time = list(cascades_time_dict.keys())
                random.seed(seed)
                random.shuffle(shuffled_time)

                count = 0
                # split datasets
                for key in shuffled_time:
                    if count < cascade_valid_total * .5:
                        cascades_type[key] = 0  # training set, 50%
                    elif count < cascade_valid_total * .6:
                        cascades_type[key] = 1  # validation set, 10%
                    else:
                        cascades_type[key] = 2  # test set, 40%
                    count += 1

            shuffle_cascades()

            # number of valid cascades
            print("Number of     labeled cascades: {}/{}".format(cascade_valid_total, cascade_total))

            # 3 list to save the filtered sets
            filtered_data_train = list()
            filtered_data_val = list()
            filtered_data_test = list()
            for line in filtered_data:
                cascade_id = line.split('\t')[0]
                if cascades_type[cascade_id] == 0:
                    filtered_data_train.append(line)
                elif cascades_type[cascade_id] == 1:
                    filtered_data_val.append(line)
                elif cascades_type[cascade_id] == 2:
                    filtered_data_test.append(line)
            print("Number of valid train cascades: {}".format(len(filtered_data_train)))
            print("Number of valid   val cascades: {}".format(len(filtered_data_val)))
            print("Number of valid  test cascades: {}".format(len(filtered_data_test)))

            # shuffle the train set again
            random.seed(seed)
            random.shuffle(filtered_data_train)

            def file_write(file_name):
                # write file, note that compared to the original 'dataset.txt', only cascade_id and each of the
                # observed adoptions are saved, plus label information at last
                file_name.write(cascade_id + '\t' + '\t'.join(observation_path) + '\t' + label + '\n')

            # write cascades into files
            for line in filtered_data_train + filtered_data_val + filtered_data_test:
                # split the cascades into 5 parts
                parts = line.split('\t')
                cascade_id = parts[0]
                observation_path = list()
                label = int()
                edges = set()
                paths = parts[4].split(' ')

                for p in paths:
                    nodes = p.split(':')[0].split('/')

                    time_now = int(p.split(":")[1])
                    if time_now < observation_time:
                        observation_path.append(",".join(nodes) + ":" + str(time_now))
                        for i in range(1, len(nodes)):
                            edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
                    # add label information depends on prediction_time, e.g., 24 hours for Weibo dataset
                    if time_now < prediction_time:
                        label += 1

                # calculate the incremental prediction
                label = str(label - len(observation_path))

                # write files by cascade type
                # 0 to train, 1 to validate, 2 to test
                if cascade_id in cascades_type and cascades_type[cascade_id] == 0:
                    file_write(data_train)
                elif cascade_id in cascades_type and cascades_type[cascade_id] == 1:
                    file_write(data_val)
                elif cascade_id in cascades_type and cascades_type[cascade_id] == 2:
                    file_write(data_test)
    else:
        # open a file to write unlabeled cascades
        with open(file_unlabel, 'w') as data_unlabel:
            print("Number of   unlabeled cascades: {}".format(len(filtered_data)))

            def unlabel_file_write(file_name):
                # write file, note that compared to the original 'dataset.txt', only cascade_id and each of the
                # observed adoptions are saved, since here we write unlabeled cascades, we don't have label information
                file_name.write(cascade_id + '\t' + '\t'.join(observation_path) + '\n')

            for line in filtered_data:
                # split the cascades into 5 parts
                parts = line.split('\t')
                cascade_id = parts[0]
                observation_path = list()
                edges = set()
                paths = parts[4].split(' ')

                for p in paths:
                    nodes = p.split(':')[0].split('/')

                    time_now = int(p.split(":")[1])
                    if time_now < observation_time:
                        observation_path.append(",".join(nodes) + ":" + str(time_now))
                        for i in range(1, len(nodes)):
                            edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

                # write unlabeled data into file
                unlabel_file_write(data_unlabel)


def main(argv):
    time_start = time.time()
    print('Start to generate cascades!\n')
    print('Dataset path: ', FLAGS.input)

    generate_cascades(FLAGS.observation_time,
                      FLAGS.prediction_time,
                      FLAGS.unlabel,
                      FLAGS.input + 'dataset.txt',
                      FLAGS.input + 'train.txt',
                      FLAGS.input + 'val.txt',
                      FLAGS.input + 'test.txt',
                      FLAGS.input + 'unlabel.txt',
                      'xovee',
                      # special caveat: because of some... historical reasons about the codes,
                      # for weibo, acm, and dblp datasets, the seed is set to 'xovee' (string),
                      # and for twitter and aps datasets, the seed is set to 0 (integer).
                      )

    time_end = time.time()
    print('Processing Time: {:.2f}s'.format(time_end - time_start))


if __name__ == "__main__":
    app.run(main)
