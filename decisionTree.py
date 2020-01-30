import sys
import math
import numpy as np

Debug = False

def gini_impurity(dataset):
    total = len(dataset)
    count_0, count_1 = 0., 0.
    value_0 = dataset[0][-1]

    for data in dataset:
        if data[-1] == value_0:
            count_0 += 1
        else:
            count_1 += 1

    return 1 - (count_0 / total) ** 2 - (count_1 / total) ** 2

def gini_gain(dataset, attri_idx):
    total = len(dataset)
    count_att0, count_att1 = 0, 0
    dataset_att0, dataset_att1 = [], []
    value_0 = dataset[0][-1]

    for data in dataset:
        if data[attri_idx] == value_0:
            count_att0 += 1
            dataset_att0.append(data)
        else:
            count_att1 += 1
            dataset_att1.append(data)

    p_att0 = count_att0 / total
    p_att1 = count_att1 / total
    gini_att0 = gini_impurity(dataset_att0)
    gini_att1 = gini_impurity(dataset_att1)
    gini_y = gini_impurity(dataset)

    return gini_y - p_att0 * gini_att0 - p_att1 * gini_att1

# return the y value which appears most times
def majority_vote(dataset):
    cnt = dict()
    for data in dataset:
        if data[-1] in cnt:
            cnt[data[-1]] += 1
        else:
            cnt[data[-1]] = 1
    return max(cnt, key = lambda x: cnt[x])

class tree_node(object):
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val

class decision_tree(object):
    def __init__(self, max_depth):
        self.root = None
        self.depth = 0
        self.max_depth = max_depth

    def train(self, train_file):
        self.value_left = None
        dataset_left, dataset_right = [], []

        with open(train_file, 'r') as f:
            idx = 0
            for line in f:
                if idx == 0:
                    idx += 1
                    continue
                split_line = line.strip().split('\t')
                # set the first value met as for left child
                if idx == 1:
                    self.value_left = split_line[self.split_idx]

                if split_line[self.split_idx] == self.value_left:
                    dataset_left.append(split_line)
                else:
                    dataset_right.append(split_line)
                idx += 1

        self.left_pred = majority_vote(dataset_left)
        self.right_pred = majority_vote(dataset_right)

        if Debug:
            print("Trainset: ", idx - 1)
            print("left chd:\n dataset: ", len(dataset_left), "pred: ", self.left_pred)
            print("right chd:\n dataset: ", len(dataset_right), "pred: ", self.right_pred)

    def prefict(self, ele):
        if ele[self.split_idx] == self.value_left:
            return self.left_pred
        else:
            return self.right_pred

    def evaluate(self, in_path, out_path):
        error = 0
        total = 0

        with open(in_path, 'r') as f_in:
            with open(out_path, 'w') as f_out:
                for line in f_in:
                    if total == 0:
                        total += 1
                        continue
                    split_line = line.strip().split('\t')

                    pred = self.prefict(split_line)
                    if pred != split_line[-1]:
                        error += 1
                    f_out.write(pred + "\n")
                    total += 1

        return error / (total - 1) # len(data)

    def print_tree(self):
        print("TODO\n")
        pass


if __name__ == '__main__':
    train_file = sys.argv[1]  # path to the training input .tsv file
    test_file = sys.argv[2] # path to the test input .tsv file
    max_depth = int(sys.argv[3])  # maximum depth to which the tree should be built
    train_out = sys.argv[4] # path of output .labels file to the predictions on the training data
    test_out = sys.argv[5]  # path of output .labels file to the predictions on the testing data
    metrics_out = sys.argv[6] # path of the output .txt file to metrics such as train and test error

    model = decision_tree(max_depth)

    # training: build the model
    model.train(train_file)

    # testing: evaluate and write labels to output files
    train_error = model.evaluate(train_file, train_out)
    test_error = model.evaluate(test_file, test_out)

    if Debug:
        print("train_error: ", train_error)
        print("test_error: ", test_error)

    # Output: Metrics File
    with open(metrics_out, 'w') as f_metrics:
        f_metrics.write("error(train): " + str(train_error) + "\n")
        f_metrics.write("error(test): " + str(test_error))

    # Output: Printing the Tree
    # Note: another way is to print it during tree generation
    model.print_tree()
