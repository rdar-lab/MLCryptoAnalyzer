import numpy as np

from utils.file_utils import get_project_root

"""
The location of the English dict in the project
To get the file:
1. Download it from here: https://www.kaggle.com/watts2/glove6b50dtxt
2. Save it to the /resources sub-folder
"""
ENGLISH_DICT_PATH = get_project_root() + "/resources/glove.6B.50d.txt"


def read_dict_file(file):
    """
    Reads a dict file and returns the 'words' and 'word_to_vec_map'
    :param file: The location of the file
    :return: A tuple of 'words', 'word_to_vec_map'
    """
    with open(file, 'r', encoding="UTF-8") as f:
        words = []
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.append(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map

