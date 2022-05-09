import logging
import torch
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
from data_processing import *

history, reply = get_history_reply_pairs()
examples_len = len(history)
spilt_len = int(0.8 * examples_len)

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

special_tokens_dict = {'bos_token': '<bos>',
                       'eos_token': '<eos>',
                       'customer_token': '<customer>',
                       'assistant_token': '<assistant>',
                       'pad_token': '<pad>',
                       'br_token': '<br>'}
special_tokens_dict1 = {'additional_special_tokens':
                            ['<bos>', '<eos>', '<customer>', '<assistant>', '<pad>', '<br>']}
tokenizer.add_special_tokens(special_tokens_dict1)
model.resize_token_embeddings(len(tokenizer))
optimizer = AdamW(model.parameters(), lr=0.01, correct_bias=True)


def build_training_data(history, reply):
    """
    Build training or testing data for training
    return input and target
    """
    sequence, target = build_input(history, reply)
    ids = []
    token_ids = []
    sentences = []
    for sentence in sequence:
        words = tokenizer.tokenize(sentence)
        sentences.append(words)
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

    # Language model losses
    len_ignored = sum(len(s) for s in sentences[:-1])
    lm_targets = [-100] * len_ignored + tokenizer.convert_tokens_to_ids(sentences[-1])
    # print(lm_targets)
    attention_mask = ([0] * sum(len(s) for s in sentences[:-1])) + [1] * (len(sentences[-1]))
    assert len(token_ids) == len(ids) == len(lm_targets) == len(attention_mask)
    # print(tokenizer.convert_ids_to_tokens(ids))
    """
    print(f"words len({len(ids)})                  {ids}")
    print(f"segments len({len(token_ids)})               {token_ids}")
    print(f"lm target len({len(lm_targets)})              {lm_targets}")
    """
    # Convert ids into Tensors
    # words tokens
    input_ids = torch.tensor([[*ids]], dtype=torch.long)
    # print(f"input_ids        shape({input_ids.size()})")
    # segment tokens
    token_type_ids = torch.tensor([[*token_ids]], dtype=torch.long)
    # print(f"tokens_ids      shape({token_type_ids.size()})")
    # Positions tokens can be automatically created by the model as (0, 1, ..., N)

    # Language modeling labels
    lm_labels = torch.tensor([[*lm_targets]], dtype=torch.long)
    # print(f"lm_labels        shape({lm_labels.size()})")

    return input_ids, token_type_ids, lm_labels


def train():
    from random import shuffle
    # Generate a random list of index
    indexes = [i for i in range(examples_len)]
    shuffle(indexes)
    train_indexes = indexes[:spilt_len]
    test_indexes = indexes[spilt_len:]
    logging.info("Preparing training and testing data ...")

    epochs = 5
    sample_num = 0
    for epoch in range(epochs):
        logging.info(f"Training epoch {epoch}")
        logging.info("*" * epoch)
        for train_index in train_indexes:
            ids, token_ids, lm_targets = \
                build_training_data(history[train_index], reply[train_index])
            # Training
            model.train()
            output = model(input_ids=ids,
                           token_type_ids=token_ids,
                           labels=lm_targets)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sample_num += 1
            if sample_num % 100000:
                # Validate
                model.eval()
                from random import choice
                test_index = choice(test_indexes)

                ids, token_ids, lm_targets = \
                    build_training_data(history[test_index], reply[test_index])
                val_loss = model(input_ids=ids,
                                 token_type_ids=token_ids,
                                 labels=lm_targets)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': output.loss,
                    'val_loss': val_loss.loss
                }, "saved_models/model" + str(sample_num % 1000))
                print("loss      - ", output.loss)
                print("val_loss  - ", val_loss.loss)
                logging.info("Saving model ...")
                logging.info(f"Training loss - {output.loss}")
                logging.info(f"Validation loss - {val_loss}")


def load_model(file="saved_models/model8"):
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    return model


def decode(history, model):
    input_ids, token_type_ids, _ = build_training_data(history, "")
    print("history is ", tokenizer.convert_ids_to_tokens(input_ids))
    output_ids = model.generate(input_ids)
    print(output_ids)
    """
    beam_outputs = model.generate(input_ids,
                                  max_length=50,
                                  num_beams=5,
                                  no_repeat_ngram_size=2,
                                  num_return_sequences=5,
                                  early_stopping=True
                                  )
    print("Output:\n" + 100 * '-')
    print(beam_outputs)
    for i, beam_output in enumerate(beam_outputs):
        output_ids = tokenizer.convert_ids_to_tokens(beam_output)
        output = " ".join(output_ids)
        print("{}: {}".format(i, output))
    num = input("Please select a response from the above sentences: ")
    if int(num) not in range(len(beam_outputs)+1):
        num = input("Please select a response from the above sentences: ")
    """

    response_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    print(response_tokens)
    response = " ".join(output_ids)
    print(response)
    return response


def evaluate_model():
    history = []
    model = load_model()
    while True:
        value = input("Please enter a sentence (input q to quit): ")
        if value == 'q':
            break
        history.append(value)
        selected_response = decode(history, model)
        history.append(selected_response)


if __name__ == '__main__':
    evaluate_model()
