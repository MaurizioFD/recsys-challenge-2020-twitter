#!/bin/bash

# train first model
python3 nn_dual_output.py like retweet 1
python3 nn_single_output.py reply 1
python3 nn_single_output.py comment 1

# predict validation set
python3 nn_test_dual_output.py like retweet cherry_val 1
python3 nn_test_single_output.py reply cherry_val 1
python3 nn_test_single_output.py comment cherry_val 1

# predict test set
python3 nn_test_dual_output.py like retweet new_test 1
python3 nn_test_single_output.py reply new_test 1
python3 nn_test_single_output.py comment new_test 1

# predict final test set
python3 nn_test_dual_output.py like retweet last_test 1
python3 nn_test_single_output.py reply last_test 1
python3 nn_test_single_output.py comment last_test 1

#------------------------------------------------------------

# train second model
python3 nn_dual_output.py like retweet 2
python3 nn_single_output.py reply 2
python3 nn_single_output.py comment 2

# predict validation set
python3 nn_test_dual_output.py like retweet cherry_val 2
python3 nn_test_single_output.py reply cherry_val 2
python3 nn_test_single_output.py comment cherry_val 2

# predict test set
python3 nn_test_dual_output.py like retweet new_test 2
python3 nn_test_single_output.py reply new_test 2
python3 nn_test_single_output.py comment new_test 2

# predict final test set
python3 nn_test_dual_output.py like retweet last_test 2
python3 nn_test_single_output.py reply last_test 2
python3 nn_test_single_output.py comment last_test 2
