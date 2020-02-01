import sys
import math
import numpy as np

Debug = False

# return the y value which appears most times
def majority_vote(dataset):
    cnt = dict()
    for data in dataset:
        if data[-1] in cnt:
            cnt[data[-1]] += 1
        else:
            cnt[data[-1]] = 1
    
    # handle tied case
    cnt_key = []
    for key in cnt:
        cnt_key.append(key)
    # sort in the reversed lexicographical order
    cnt_key.sort(reverse=True)
    if len(cnt_key) == 2 and cnt[cnt_key[0]] == cnt[cnt_key[1]]:
        return cnt_key[0]
    return max(cnt, key = lambda x: cnt[x])

def gini_impurity(dataset):
    total = len(dataset)
    count_0, count_1 = 0., 0.
    value_0, value_1 = dataset[0][-1], None

    for data in dataset:
        if data[-1] == value_0:
            count_0 += 1
        else:
            if value_1 == None:
                value_1 = data[-1]
            count_1 += 1

    gini = 1 - (count_0 / total) ** 2 - (count_1 / total) ** 2
    label = {value_0: int(count_0), value_1: int(count_1)}

    return gini, label

def gini_gain(dataset, attri_idx):
    total = len(dataset)
    count_att0, count_att1 = 0, 0
    dataset_att0, dataset_att1 = [], []
    value_0, value_1 = dataset[0][attri_idx], None

    for data in dataset:
        if data[attri_idx] == value_0:
            count_att0 += 1
            dataset_att0.append(data)
        else:
            if value_1 == None:
                value_1 = data[attri_idx]
            count_att1 += 1
            dataset_att1.append(data)

    # If one attri has constant value, then it's unmeaningful to split. Just return 0
    if len(dataset_att0) == 0 or len(dataset_att1) == 0:
        return 0, None

    if Debug:
        print(dataset)
        print(dataset_att0)
        print(dataset_att1)

    p_att0 = count_att0 / total
    p_att1 = count_att1 / total
    gini_att0, _ = gini_impurity(dataset_att0)
    gini_att1, _ = gini_impurity(dataset_att1)
    gini_y, label = gini_impurity(dataset)
    gg = gini_y - p_att0 * gini_att0 - p_att1 * gini_att1

    node_info = {"label": label, "left_value": value_0, "right_value": value_1, \
                                 "left_ds": dataset_att0, "right_ds": dataset_att1}

    return gg, node_info


class tree_node(object):
    def __init__(self, val, isleaf = False):
        self.left = None
        self.right = None
        self.val = val  # store split_idx if isn't a leaf, otherwise store predicted value
        self.isleaf = isleaf
        self.split_info = None # if not a leaf, store split-related information


class decision_tree(object):
    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth
        # The following attributes are for printing tree
        self.attriName = None
        self.labelName = set()
        self.dataset = None

    def build_tree(self, train_file):
        dataset = []

        # read from dataset
        with open(train_file, 'r') as f:
            idx = 0
            for line in f:
                split_line = line.strip().split('\t')
                if idx == 0:
                    self.attriName = split_line
                else:
                    dataset.append(split_line)
                    if split_line[-1] not in self.labelName:
                        self.labelName.add(split_line[-1])
                idx += 1

        self.dataset = dataset

        # use length of first data line to generate available attributes # set
        self.root = self.train_stump(dataset, set(range(len(dataset[0]) - 1)))

    def train_stump(self, dataset, available_nodes, depth = 0):
        # special stopping rules
        pred = majority_vote(dataset)
        _, label = gini_impurity(dataset)
        if (len(available_nodes) == 0) or (depth >= self.max_depth):
            if Debug:
                print("stoped: special rules", available_nodes, depth)
            return self.make_leaf(pred, label)

        gg_max, split_idx = 0, -1
        dataset_0, dataset_1 = None, None
        for idx in available_nodes:
            gg_cur, node_info = gini_gain(dataset, idx)
            # Is there any possibility to have two identical gini from two attris?
            if (gg_cur > gg_max):
                gg_max = gg_cur
                split_idx = idx
                split_info = node_info

        if split_idx != -1:
            # split and create a node
            node = tree_node(split_idx)
            node.split_info = split_info
            dataset_0, dataset_1 = node.split_info["left_ds"], node.split_info["right_ds"]
            
            # must make a new copy in order to don't affect other sub-trees
            unused_nodes = available_nodes.copy()
            unused_nodes.remove(split_idx)
            next_depth = depth
            next_depth += 1

            if Debug:
                print("left chd:\n dataset: ", len(dataset_0))
                print("right chd:\n dataset: ", len(dataset_1))

            # build sub trees
            left_chd = self.train_stump(dataset_0, unused_nodes, next_depth)
            if left_chd:
                node.left = left_chd
            else:
                node.left = tree_node(majority_vote(dataset_0), True)

            if Debug:
                print("from here is the right tree")
            right_chd = self.train_stump(dataset_1, unused_nodes, next_depth)
            if right_chd:
                node.right = right_chd
            else:
                node.right = tree_node(majority_vote(dataset_1), True)

            return node
        else:
            # touch stoping rule, no split
            if Debug:
                print("stoped: regular rules")
            return self.make_leaf(pred, label)

    # return a new leaf with given label info (for printing)
    def make_leaf(self, pred, label):
        newleaf = tree_node(pred, True)
        newleaf.split_info = {"label": label}
        return newleaf

    # Use decision tree to predict y for a single data line
    def predict(self, node, ele):
        if node.isleaf:
            return node.val
        elif node.split_info["left_value"] == ele[node.val]:
            return self.predict(node.left, ele)
        elif node.split_info["right_value"] == ele[node.val]:
            return self.predict(node.right, ele)
        else:
            print("Error! Unknown value " + ele[node.val] + "for attribute " + self.attriName[node.val])
            exit(-1)

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

                    pred = self.predict(self.root, split_line)
                    if pred != split_line[-1]:
                        error += 1
                    f_out.write(pred + "\n")
                    total += 1

        return error / (total - 1) # len(data)

    def print_tree(self, node, layers):
        # TODO: does blanks matters?
        '''
        split_info = {"label": label, "left_value": value_0, "right_value": value_1,
                                      "left_ds": dataset_att0, "right_ds": dataset_att1}
        label = {value_0: count_0, value_1: count_1}
        '''
        # this is to ensure "/" be printed only in first iteration
        first_iter = True
        print("[", end = "")
        for y_name in self.labelName:
            if y_name in node.split_info["label"]:
                print(str(node.split_info["label"][y_name]) + " " + y_name, end = " ")
            else:
                print("0 " + y_name, end = "")
            if first_iter:
                print(" /", end = "")
            first_iter = False
        print("]")
        if not node.isleaf:
            print("| " * layers + self.attriName[node.val] + " = " + node.split_info["left_value"] + ": ", end = '')
            self.print_tree(node.left, layers + 1)
            print("| " * layers + self.attriName[node.val] + " = " + node.split_info["right_value"] + ": ", end = '')
            self.print_tree(node.right, layers + 1)


if __name__ == '__main__':
    train_file = sys.argv[1]  # path to the training input .tsv file
    test_file = sys.argv[2] # path to the test input .tsv file
    max_depth = int(sys.argv[3])  # maximum depth to which the tree should be built
    train_out = sys.argv[4] # path of output .labels file to the predictions on the training data
    test_out = sys.argv[5]  # path of output .labels file to the predictions on the testing data
    metrics_out = sys.argv[6] # path of the output .txt file to metrics such as train and test error

    model = decision_tree(max_depth)

    # training: build the decison tree model
    model.build_tree(train_file)

    # testing: evaluate and write labels to output files
    train_error = model.evaluate(train_file, train_out)
    test_error = model.evaluate(test_file, test_out)

    print("train_error: ", train_error)
    print("test_error: ", test_error)

    # Output: Metrics File
    with open(metrics_out, 'w') as f_metrics:
        f_metrics.write("error(train): " + str(train_error) + "\n")
        f_metrics.write("error(test): " + str(test_error))

    # Output: Printing the Tree
    # Note: another way is to print it during tree generation
    model.print_tree(model.root, 1)
