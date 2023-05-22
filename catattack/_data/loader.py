import os

def _load_pytorch_tutorial_text(path):
    r"""Loads the pytorch translation dataset

    Assumption:
        * There must be a "eng-fra.txt" file under the path

    """
    eng_fra_path = os.path.join(path, 'eng-fra.txt')
    with open(eng_fra_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [line.split('\t') for line in lines]
