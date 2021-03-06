import json
import os
import logging
from itertools import chain


def generate_dialogs(file, count):
    """Extract dialogs from the original dataset and store it under directory dialogs."""
    with open(file) as f:
        data = json.load(f)
    dialogs = {}
    idx = 0
    for dialogue in data:
        dialog = []
        for turns in dialogue["turns"]:
            dialog.append(turns["utterance"])
        name = "dialog_" + str(idx)
        dialogs[name] = dialog
        idx += 1

    f_name = "dialogs/" + "dialogs_" + str(count) + ".json"
    with open(f_name, 'w') as d_f:
        json.dump(dialogs, d_f, indent=4)


def generate_dialogs4all():
    """Generate dialog files for all the original training datasets."""
    directory = "multiwoz"
    count = 1
    for filename in os.listdir(directory):
        print(f"generating file for {filename} in {directory} ...")
        f = os.path.join(directory, filename)
        generate_dialogs(f, count)
        count += 1


def build_training_input(history, target):
    """Build input for training, input begins with <bos>"""
    with open("../config.json") as config:
        data = json.load(config)
    bos, eos, customer, assistant, pad, _ = data["special_tokens"]
    count = 0
    sequence = []

    # print("history (input) -> ", history)
    # print("target (input) -> ", target)
    for sen in history:
        # print("sen is: ", sen)
        # Add dialog turns into the sequence with the token of the speaker
        pre = customer if count % 2 == 0 else assistant
        string = pre + ' ' + sen if sequence \
            else bos + ' ' + pre + ' ' + sen
        # print("sentence ", count, ": ", string)
        sequence.append(string)
        count += 1

    if target:
        target = customer + ' ' + target \
            if count % 2 == 0 else assistant + ' ' + target
        sequence.append(target)

    # print("sequence (input) -> ", sequence)
    return sequence


def build_training_label(history, target):
    """Build label for training, output ends with <eos>"""
    with open("../config.json") as config:
        data = json.load(config)
    bos, eos, customer, assistant, pad, _ = data["special_tokens"]
    count = 0
    sequence = []
    # print("history (label) -> ", history)
    # print("target (label) -> ", target)
    for sen in history:
        # Add dialog turns into the sequence with the token of the speaker
        pre = customer if count % 2 == 0 else assistant
        string = pre + ' ' + sen
        sequence.append(string)
        count += 1
    if target:
        target = customer + ' ' + target + ' ' + eos \
            if count % 2 == 0 else assistant + ' ' + target + ' ' + eos
        sequence.append(target)
    # print("sequence (label) -> ", sequence)
    return sequence


def build_input(history, target):
    """
    Build input for language model.
    history: a list of strings
    reply: a string
    """
    with open("../config.json") as config:
        data = json.load(config)
    bos, eos, customer, assistant, pad, _ = data["special_tokens"]
    count = 0
    sequence = []
    sequence_no_eos = []
    for sen in history:
        # print("sen is: ", sen)
        # Add dialog turns into the sequence with the token of the speaker
        pre = customer if count % 2 == 0 else assistant
        string_no_bos = pre + ' ' + sen
        string = pre + ' ' + sen if sequence \
            else bos + ' ' + pre + ' ' + sen
        # print("sentence ", count, ": ", string)
        sequence.append(string)
        sequence_no_eos.append(string_no_bos)
        count += 1
    # If target do not exist, then no eos in the end of the sentence.
    if target:
        target_no_eos = customer + ' ' + target \
            if count % 2 == 0 else assistant + ' ' + target
        target = customer + ' ' + target + ' ' + eos \
            if count % 2 == 0 else assistant + ' ' + target + ' ' + eos
        sequence.append(target)
    '''
    # Build our word, segments and position inputs from the sequence
    words = list(chain(*sequence))
    segments = [customer if i % 2 == 0 else assistant for i, s in enumerate(sequence) for _ in s]

    position = list(range(len(words)))

    assert len(words) == len(segments) == len(position)
    '''
    return sequence, target, sequence_no_eos


def get_history_reply_pairs(data_usage: str):
    start, end = 0, 0
    if data_usage == 'train':
        start, end = get_train_files()
    elif data_usage == 'val':
        start, end = get_val_files()
    elif data_usage == 'test':
        start, end = get_test_files()
    history, reply = [], []
    for file_num in range(start, end + 1):
        file_name = "../dialogs/" + "dialogs_" + str(file_num) + ".json"
        with open(file_name) as f:
            data = json.load(f)
        dialog_num = len(data)
        for ith in range(dialog_num):
            dialog_name = 'dialog_' + str(ith)
            dialog = data[dialog_name]
            turns = len(dialog)
            for i in range(2, turns):
                history.append(dialog[:i - 1])
                reply.append(dialog[i - 1])
        assert len(history) == len(reply)
    return history, reply


def get_train_files():
    with open("../config.json", 'r') as f:
        data = json.load(f)
    start, end = data["train_file_id"]
    return start, end


def get_val_files():
    with open("../config.json", 'r') as f:
        data = json.load(f)
    start, end = data["val_file_id"]
    return start, end


def get_test_files():
    with open("../config.json", 'r') as f:
        data = json.load(f)
    start, end = data["test_file_id"]
    return start, end

def choose_distractor(file="dialogs/distractor.json"):
    """Choose a random distracotr sentence from the file."""
    with open(file) as f:
        data = json.load(f)
    from random import randint
    num = randint(0, len(data)-1)
    return data[str(num)]