import sys
import math
import numpy as np

Debug = True

# return the y value which appears most times
def majority_vote(dataset):
    cnt = dict()
    for data in dataset:
        if data[-1] in cnt:
            cnt[data[-1]] += 1
        else:
            cnt[data[-1]] = 1
    return max(cnt, key = lambda x: cnt[x])

def gini_impurity(dataset):
    total = len(dataset)
    # TODO: not sure if this's correct (do I need to handle it pre?)
    # Unmeaningful to split, so return 0
    if total == 0:
        return 0

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
    label_count = [value_0, count_0, value_1, count_1]

    return gini, label_count

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

    node_info = {"label": label, "left_value": value_0, "right_vlaue": value_1, \
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
        self.depth = 0
        self.max_depth = max_depth
        self.unused_nodes = None
        self.isMarVote = False
        # The following attris are for printing tree
        self.attriName = None

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
                idx += 1

        # len(0, 1, 2, 3, 4) = 5
        self.unused_nodes = set(range(len(dataset[0]) - 1))
        self.root = self.train_stump(dataset)
        if self.root == None:
            self.isMarVote = True

    def train_stump(self, dataset):
        # additional stopping rules
        if (len(self.unused_nodes) == 0) and (self.depth >= self.max_depth):
            return None

        gg_max, split_idx = 0, -1
        dataset_0, dataset_1 = None, None
        for idx in self.unused_nodes:
            gg_cur, node_info = gini_gain(dataset, idx)
            # Is there any possibility to have two identical gini from two attris?
            if (gg_cur > gg_max):
                gg_max = gg_cur
                split_idx = idx

        if split_idx != -1:
            # split and create a node
            node = tree_node(split_idx)
            node.split_info = node_info
            dataset_0, dataset_1 = node.split_info[0]["dataset"], node.split_info[1]["dataset"]
            self.depth += 1

            if Debug:
                print("left chd:\n dataset: ", len(dataset_0))
                print("right chd:\n dataset: ", len(dataset_1))

            # build sub trees
            left_chd = self.train_stump(dataset_0)
            if left_chd:
                node.left = left_chd
            else:
                node.left = tree_node(majority_vote(dataset_0), True)

            right_chd = self.train_stump(dataset_1)
            if right_chd:
                node.right = right_chd
            else:
                node.right = tree_node(majority_vote(dataset_1), True)

            return node
        else:
            # touch stoping rule, no split
            return None

    # Use decision tree to predict y for a single data line
    # this func is used to packaging self.isMarVote
    def prefict(self, ele):
        if self.isMarVote:
            return self.root
        return self.predict_node(self.root, ele)

    def predict_node(self, node, ele):
        if node.isleaf:
            return node.val
        elif node.split_info["left_value"] == ele[node.val]:
            return self.predict_node(node.left, ele)
        elif node.split_info["right_value"] == ele[node.val]:
            return self.predict_node(node.right, ele)
        else:
            print("Error! Unknown value " + ele[node.val] + "for attribute " + self.attriName[node.val])

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

    def print_tree(self, node, layers):
        # TODO
        '''
        split_info = {"label": label, "left_value": value_0, "right_vlaue": value_1,
                                      "left_ds": dataset_att0, "right_ds": dataset_att1}
        label = [value_0, count_0, value_1, count_1]
        '''
        print("[" + node.split_info["label"][1] + " " + node.split_info["label"][0] \
            + "/" + node.split_info["label"][3] + " " + node.split_info["label"][2] + "]")
        if not node.isleaf:
            print("| " * layers + self.attriName[node.val] + " = " + node.split_info["left_value"] + ": ", end="")
            self.print_tree(node.left, layers + 1)
            print("| " * layers + self.attriName[node.val] + " = " + node.split_info["right_value"] + ": ", end="")
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

    if Debug:
        print("train_error: ", train_error)
        print("test_error: ", test_error)

    # Output: Metrics File
    with open(metrics_out, 'w') as f_metrics:
        f_metrics.write("error(train): " + str(train_error) + "\n")
        f_metrics.write("error(test): " + str(test_error))

    # Output: Printing the Tree
    # Note: another way is to print it during tree generation
    model.print_tree(model.root, 1)
