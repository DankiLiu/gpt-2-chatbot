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
        print(f)
        generate_dialogs(f, count)
        count += 1


def build_input(history, target):
    """
    Build input for language model.
    history: a list of strings
    reply: a string
    """
    with open("config.json") as config:
        data = json.load(config)
    bos, eos, customer, assistant, pad, br = data["special_tokens"]
    count = 0
    sequence = []
    for sen in history:
        print("sen is: ", sen)
        # Add dialog turns into the sequence with the token of the speaker
        pre = customer if count % 2 == 0 else assistant
        string = pre + ' ' + sen + ' ' + br if sequence \
            else bos + ' ' + pre + ' ' + sen + ' ' + br
        print("sentence ", count, ": ", string)
        sequence.append(string.split())
        count += 1
    target = customer + ' ' + target + ' ' + eos \
        if count % 2 == 0 else assistant + ' ' + target + ' ' + eos
    sequence.append(target.split())
    print("add all the history sentences \n", sequence)

    # Build our word, segments and position inputs from the sequence
    words = list(chain(*sequence))
    # print(f"words with length {len(words)}\n{words}")
    segments = [customer if i % 2 == 0 else assistant for i, s in enumerate(sequence) for _ in s]
    # print(f"segments with length {len(segments)}\n{segments}")
    position = list(range(len(words)))
    # print(f"position with length {len(position)}\n{position}")

    assert len(words) == len(segments) == len(position)
    return words, segments, position, sequence


def get_history_reply_pairs():
    file_id, _ = read_from_json() # how many files are gonna be read
    history, reply = [], []
    for file_num in range(1, file_id+1):
        file_name = "dialogs/" + "dialogs_" + str(file_num) + ".json"
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
            print(f"history: {history}")
            print(f"reply: {reply}")
        assert len(history) == len(reply)
    return history, reply


def read_from_json():
    with open("config.json", 'r') as f:
        data = json.load(f)
    file_id = data["file_id"]
    dialog_id = data["dialog_id"]
    return file_id, dialog_id


def write_to_json(file_id, dialog_id):
    with open("config.json", 'r') as f:
        data = json.load(f)
    data["file_id"] = file_id
    data["dialog_id"] = dialog_id
    with open("config.json", 'w') as f:
        json.dump(data, "config.json")


def choose_distractor(file="dialogs/distractor.json"):
    """Choose a random distracotr sentence from the file."""
    with open(file) as f:
        data = json.load(f)
    from random import randint
    num = randint(0, len(data)-1)
    return data[str(num)]


def print_input(words, segments, position, sequence):
    print("-------------input---------------")
    print(words)
    print(segments)
    print(position)
    print(sequence)
    print("----------------------------------")


distractor = choose_distractor()
history, reply = get_history_reply_pairs()

print(distractor)
print("-------------")
print(history)
print("-------------")
print(reply)

