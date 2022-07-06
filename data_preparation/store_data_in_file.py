import json
from data_processing import build_training_input, build_training_label, get_history_reply_pairs

from model_lightning import define_tokenizer


def get_token_ids(sequence, tokenizer):
    token_ids = []
    for sentence in sequence:
        words = tokenizer.tokenize(sentence)
        # build token_ids
        if '<customer>' in words:
            segments = '<customer> ' * len(words)
            # token_ids = [*token_ids, *tokenizer.convert_tokens_to_ids(tokenizer.tokenize(segments))]
            token_ids = [*token_ids, *len(tokenizer.tokenize(segments)) * [0]]
        elif '<assistant>' in words:
            segments = '<assistant> ' * len(words)
            # token_ids = [*token_ids, *tokenizer.convert_tokens_to_ids(tokenizer.tokenize(segments))]
            token_ids = [*token_ids, *len(tokenizer.tokenize(segments)) * [1]]
    return token_ids


def store_data_in_file(tokenizer, usage):
    """Store the data in json file"""
    # history is a list of strings
    # reply is a string
    histories, replies = get_history_reply_pairs(data_usage=usage)
    count = 0
    assert len(histories) == len(replies)
    examples = []
    for i in range(len(histories)):
        print(f"adding {count}th example")
        model_input = build_training_input(histories[i], replies[i])
        model_label = build_training_label(histories[i], replies[i])

        token_ids = get_token_ids(model_label, tokenizer)
        model_input = ' '.join([model_input[i] for i in range(len(model_input))])
        model_label = ' '.join([model_label[i] for i in range(len(model_label))])
        example = {
            "id": count,
            "model_input": model_input,
            "model_label": model_label,
            'token_ids': token_ids
        }
        examples.append(example)
        count += 1

    examples = {"example" : 1}

    name = '../data_preparation/examples_' + usage + '.json'
    print("storing")
    with open(name, 'a') as f:
        json.dump(examples, f, indent=4)


if __name__ == '__main__':
    tokenizer = define_tokenizer()
    # store_data_in_file(tokenizer, "train")
    # print("train data stored")
    store_data_in_file(tokenizer, "val")
    print("val data stored")
    store_data_in_file(tokenizer, "test")
    print("test data stored")