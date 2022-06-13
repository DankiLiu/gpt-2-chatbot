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


def init_model_optimizer(tokenizer, cuda):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    if cuda:
        model = model.cuda()
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=0.01, correct_bias=True)
    # print(f"original max length is {model.config.max_length}")
    model.config.max_length = 1020
    return model, optimizer


def build_training_data(history, reply):
    """
    Build training or testing data for training
    return input and target
    """
    input_seq = build_training_input(history, reply)
    label_seq = build_training_label(history, reply)
    print(f"input sequence {input_seq}")
    print(f"label sequence {label_seq}")
    assert len(input_seq) == len(label_seq)
    # print(f"input sequence: {input_seq}\nlabel sequence: {label_seq}")
    # sequence, target, sequence_no_nos, target_no_eos = build_input(history, reply)
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

    input_ids, input_token_ids, input_sentences = token_ids(input_seq)
    label_ids, label_token_ids, label_sentences = token_ids(label_seq)
    # print("input ids: ", input_ids)
    # print("label ids", label_ids)
    """add padding here"""
    len_ignored = sum(len(s) for s in label_sentences[:-1])
    lm_targets = [-100] * len_ignored + tokenizer.convert_tokens_to_ids(label_sentences[-1])
    # print(lm_targets)
    # attention_mask = ([0] * sum(len(s) for s in sentences[:-1])) + [1] * (len(sentences[-1]))
    # assert len(input_token_ids) == len(input_ids) == len(lm_targets)
    # Convert ids into Tensors
    # words tokens
    input_ids = torch.tensor([[*input_ids]], dtype=torch.long)
    print(f"input_ids        shape({input_ids.size()})")
    # segment tokens
    token_type_ids = torch.tensor([[*input_token_ids]], dtype=torch.long)
    print(f"tokens_ids      shape({token_type_ids.size()})")
    # Language modeling labels
    lm_labels = torch.tensor([[*lm_targets]], dtype=torch.long)
    print(f"lm_labels        shape({lm_labels.size()})")
    return input_ids, token_type_ids, lm_labels


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
                    token_ids = token_ids.cuda()
                    # token_ids = token_ids.cuda()
                    # lm_targets = lm_targets.cuda()
                """
                val_loss = model(input_ids=ids,
                                 token_type_ids=token_ids,
                                 labels=lm_targets)              
                """
                responses = model.generate(input_ids=ids,
                                           max_length=50)
                output = model(input_ids=ids,
                               token_type_ids=token_ids)
                print(f"Model evaluation: \ninput: {history[test_index]}")
                print(f"model decode: {tokenizer.decode(responses[0])}")
                # print(f"model output: {tokenizer.decode(output)}")
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
        if cuda:
            ids = ids.cuda()
            token_ids = token_ids.cuda()
        val_loss = model(input_ids=ids,
                         token_type_ids=token_ids)
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
    loss = checkpoint['val_loss']
    print(f"load model with epoch {epoch} and loss {loss}")
    if cuda:
        model.cuda()
    return model, optimizer


def model_evaluation():
    """Input a sentence and model makes prediction."""
    cuda = torch.cuda.is_available()
    print("---------Model Evaluation---------")
    # load recent model
    model, optimizer = load_model(True, cuda)
    if cuda:
        model.cuda()

    model.eval()
    # user input a sentence and model predict response
    sentence = "I want to book a cheap hotel."
    print(f"request: {sentence}")
    input_ids, _, _ = build_training_data([sentence], None)
    while True:
        if cuda:
            input_ids.cuda()
        print(f"model on cuda? {next(model.parameters()).is_cuda}\n"
              f"input_ids is on device {input_ids.get_device()}")
        model.cpu()
        input_ids.cpu()
        responses = model.generate(input_ids=input_ids,
                                   max_length=50)
        res_sen = tokenizer.decode(responses[0],
                                   skip_special_tokens=True)
        print(f"input sentence: {sentence}")
        print(f"model response: {res_sen}")
        if res_sen.split(' ')[-1] == "<br>" or res_sen.spilt(' ')[-1] == "<eos>":
            break
        input_ids = responses


if __name__ == '__main__':
    # evaluate_model()
    #cuda = torch.cuda.is_available()
    #train(from_checkpoint=False, cuda=cuda)
    model_evaluation()