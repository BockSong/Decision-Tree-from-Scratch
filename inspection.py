import sys
import numpy as np

Debug = True

def inspect(input_file):
    value_left = None
    left_count, right_count = 0., 0.

    with open(input_file, 'r') as f:
        idx = -1
        for line in f:
            if idx == -1:
                idx += 1
                continue
            split_line = line.strip().split('\t')
            # set the first y value met as for left edge
            if idx == 0:
                value_left = split_line[-1]

            if split_line[-1] == value_left:
                left_count += 1
            else:
                right_count += 1
            idx += 1

    gini = 1 - (left_count / idx) ** 2 - (right_count / idx) ** 2
    error = min(left_count, right_count) / idx

    if Debug:
        print("Dataset size: ", idx)
        print("For value ", value_left, ": ", left_count)
        print("For the other value: ", right_count)
        print("gini_impurity: " + str(gini))
        print("error: " + str(error))

    return gini, error


if __name__ == '__main__':
    input_file = sys.argv[1]  # path to the training input .tsv file
    output_file = sys.argv[2] # path to the test input .tsv file

    gini, error = inspect(input_file)

    with open(output_file, 'w') as f_out:
        f_out.write("gini_impurity: " + str(gini) + "\n")
        f_out.write("error: " + str(error))
