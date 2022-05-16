import logging
import torch
import json
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from data_processing import *

history, reply = get_history_reply_pairs()
examples_len = len(history)
spilt_len = int(0.8 * examples_len)


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


def init_model_optimizer(tokenizer, cuda=True):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    if cuda:
        model = model.cuda()
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=0.01, correct_bias=True)
    print(f"original max length is {model.config.max_length}")
    model.config.max_length = 1020
    return model, optimizer


def build_training_data(history, reply):
    """
    Build training or testing data for training
    return input and target
    """
    sequence, target = build_input(history, reply)
    ids = []
    token_ids = []
    sentences = []
    print(history)
    print(reply)
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

    if target:
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
        # print(f"transfered to ids: \n{ids}\n{token_ids}")
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
    else:
        input_ids = torch.tensor([[*ids]], dtype=torch.long)
        token_type_ids = torch.tensor([[*token_ids]], dtype=torch.long)
        return input_ids, token_type_ids, None


def train(from_checkpoint=False, cuda=True):
    # If trained model exists, then load trained model, otherwise load pre-trained model.
    model, optimizer = load_model(from_checkpoint, cuda)
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
            if cuda:
                ids = ids.cuda()
                token_ids = token_ids.cuda()
                lm_targets = lm_targets.cuda()
            output = model(input_ids=ids,
                           token_type_ids=token_ids,
                           labels=lm_targets)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sample_num += 1
            if sample_num % 1000000:
                # Validate after every 1000000 examples
                model.eval()
                from random import choice
                test_index = choice(test_indexes)

                ids, token_ids, _ = \
                    build_training_data(history[test_index], None)
                if cuda:
                    ids = ids.cuda()
                    # token_ids = token_ids.cuda()
                    # lm_targets = lm_targets.cuda()
                """
                val_loss = model(input_ids=ids,
                                 token_type_ids=token_ids,
                                 labels=lm_targets)              
                """
                responses = model.generate(input_ids=ids,
                                           max_length=50)

                print(f"Model evaluation: with \ninput: {history[test_index]}"
                      f"\nmodel response: {tokenizer.convert_ids_to_tokens(responses[0])}")
                print(f"model decode: {tokenizer.decode(responses[0])}")
                # sentences = []
                """
                for res in responses:
                    sentences.append(tokenizer.decode(res))
                """
                from datetime import datetime
                training_info = {
                    "time": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "epoch": epoch,
                    "sample_size": sample_num,
                    "loss": str(output.loss)
                }
                outfile = open('training_info.json', 'a')
                json.dump(training_info, outfile, indent=6)
                outfile.close()
                print("epoch     - ", epoch)
                print("loss      - ", output.loss)
                # print("val_loss  - ", val_loss.loss)

        # Validate after one epoch
        model.eval()
        from random import choice
        test_index = choice(test_indexes)

        ids, token_ids, _ = \
            build_training_data(history[test_index], None)
        val_loss = model(input_ids=ids.cuda(),
                         token_type_ids=token_ids.cuda())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss.loss
        }, "saved_models/model" + str(sample_num % 1000))
        print("epoch     - ", epoch)
        print("val_loss  - ", val_loss.loss)
        logging.info("Saving model ...")
        logging.info(f"Validation loss - {val_loss}")


def load_model(from_checkpoint=False, cuda=True):
    import glob
    import os
    model_files = glob.glob('saved_models/*')
    model, optimizer = init_model_optimizer(tokenizer, cuda)
    if not model_files or not from_checkpoint:
        return model, optimizer
    last_model_file = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(last_model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"load model with epoch {epoch} and loss {loss}")
    return model, optimizer


def decode(history, model):
    input_ids, token_type_ids, _ = build_training_data(history, "")
    print("history is ", tokenizer.convert_ids_to_tokens(input_ids[0]))
    output_ids = model.generate(input_ids)
    print("model output is: ", output_ids)
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
    response_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
    response = " ".join(response_tokens)
    print("model output in string: ", response)
    return response


def evaluate_model(from_checkpoint, cuda):
    history = []
    model, optimizer = load_model(from_checkpoint, cuda)
    model.eval()
    while True:
        value = input("Please enter a sentence (input q to quit): ")
        if value == 'q':
            break
        history.append(value)
        selected_response = decode(history, model)
        history.append(selected_response)


if __name__ == '__main__':
    # evaluate_model()
    """
    is_cuda = input("train with cuda? Please type 'y'")
    if 'Y' or 'y':
        is_cuda = True
    else:
        is_cuda = False
    """
    train(from_checkpoint=False, cuda=True)