import json

from data_processing import get_history_reply_pairs
from data_processing import build_training_input, build_training_label

"""Store the data in json file"""
# history is a list of strings
# reply is a string
histories, replies = get_history_reply_pairs()
# print three examples of history, reply pairs
for i in range(3):
    print("history: ", histories[i])
    print("reply: ", replies[i])

# add special tokens
count = 0
assert len(histories) == len(replies)
examples = []

for i in range(len(histories)):
    model_input = build_training_input(histories[i], replies[i])
    model_label = build_training_label(histories[i], replies[i])
    """model_input and model_label are a list of strings with special token.
    concatenate the strings into one string."""
    model_input = ' '.join([model_input[i] for i in range(len(model_input))])
    model_label = ' '.join([model_label[i] for i in range(len(model_label))])
    example = {
        "id": count,
        "model_input": model_input,
        "model_label": model_label
    }
    examples.append(example)
    count += 1

with open("examples_test.json", 'a') as f:
    json.dump(examples, f, indent=4)

"""
def token_ids(sequence):
    sentences = []
    token_ids = []
    ids = []
    for sentence in sequence:
        words = tokenizer.tokenize(sentence)
        sentences.append(words)
        # build token_ids
        if '<customer>' in words:
            segments = '<customer> ' * len(words)
            # token_ids = [*token_ids, *tokenizer.convert_tokens_to_ids(tokenizer.tokenize(segments))]
            token_ids = [*token_ids, *len(tokenizer.tokenize(segments)) * [0]]
        elif '<assistant>' in words:
            segments = '<assistant> ' * len(words)
            # token_ids = [*token_ids, *tokenizer.convert_tokens_to_ids(tokenizer.tokenize(segments))]
            token_ids = [*token_ids, *len(tokenizer.tokenize(segments)) * [1]]
        ids = [*ids, *tokenizer.convert_tokens_to_ids(words)]
        assert len(token_ids) == len(ids)
    return ids, token_ids, sentences
"""