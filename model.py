from asyncore import dispatcher
import logging
import torch
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from transformers import AdamW
from data_processing import *

history, reply = get_history_reply_pairs()
distractor = choose_distractor()

model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


def build_trainging_data_batch(history, reply):
    words, words_distractor, segments, segments_distractor, \
    lm_targets, lm_distractor, last_token, last_token_distractor = [], [], [], [], [], [], [], []

    for i in range(len(history)):
        distractor = choose_distractor()
        wrds, wrds_dis, seg, seg_dis, lm_trgs, lm_dis, las, las_dis = build_training_data(history[i], reply[i],
                                                                                          distractor)
        words.append(wrds)
        words_distractor.append(wrds_dis)
        segments.append(seg)
        segments_distractor.append(seg_dis)
        lm_targets.append(lm_trgs)
        lm_distractor.append(lm_dis)
        last_token.append(las)
        last_token_distractor.append(las_dis)
    # And gather reply and distractor inputs to build the input tensors:
    # words tokens
    input_ids = torch.tensor([[words, words_distractor]], dtype=torch.long)
    # segment tokens
    token_type_ids = torch.tensor([[segments, segments_distractor]], dtype=torch.long)
    # Positions tokens can be automatically created by the model as (0, 1, ..., N)
    # Last tokens location
    mc_token_ids = torch.tensor([[last_token, last_token_distractor]], dtype=torch.long)
    # Language modeling labels
    lm_labels = torch.tensor([[lm_targets, lm_distractor]], dtype=torch.long)
    # Next-sentence prediction labels
    mc_labels = torch.tensor([0], dtype=torch.long)  # Gold reply is 1st (index 0)
    return input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids


def build_training_data(history, reply, distractor):
    """
    Build training or testing data for training
    return input and target
    """
    words, segments, position, sequence = build_input(history, reply)
    words_distractor, segments_distractor, _, _ = build_input(history, distractor)

    words = tokenizer.convert_tokens_to_ids(words)
    segments = tokenizer.convert_tokens_to_ids(segments)
    words_distractor = tokenizer.convert_tokens_to_ids(words_distractor)
    segments_distractor = tokenizer.convert_tokens_to_ids(segments_distractor)
    # Language model losses
    lm_targets = ([-1] * sum(len(s) for s in sequence[:-1])) \
                 + [-1] + tokenizer.convert_tokens_to_ids(sequence[-1][1:])
    lm_distractor = [-1] * len(words_distractor)
    # Store the posistion of the last tokens for the next sentence prediction loss
    last_token = len(words) - 1
    last_token_distractor = len(words_distractor) - 1

    padding_length = max(len(words), len(words_distractor))

    def pad(x, padding):
        return x + [padding] * (padding_length - len(x))

    (words, words_distractor,
     segments, segments_distractor) = [pad(x, padding=tokenizer.convert_tokens_to_ids('<pad>'))
                                       for x in (words, words_distractor, segments, segments_distractor)]
    (lm_targets, lm_distractor) = [pad(x, padding=tokenizer.convert_tokens_to_ids('<pad>'))
                                   for x in (lm_targets, lm_distractor)]
    return words, words_distractor, segments, segments_distractor, lm_targets, lm_distractor, last_token, last_token_distractor


def forward(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids):
    # Forward pass
    lm_loss, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)
    # Total loss as a weighted sum
    lm_coef = 2.0
    mc_coef = 1.0
    total_loss = lm_loss * lm_coef + mc_loss * mc_coef
    return total_loss


def train():
    optimizer = AdamW(model.parameters(), lr=0.01, correct_bias=True)
    input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = \
        build_trainging_data_batch(history, reply)
    from random import shuffle
    spilt_len = 0.8 * len(input_ids)
    logging.info("Preparing training and testing data ...")
    logging.info(f"Training dataset has {spilt_len} data samples in total.")


    input_ids_train, input_ids_test = shuffle(input_ids[:spilt_len]), shuffle(input_ids[spilt_len:])
    mc_token_ids_train, mc_token_ids_test = shuffle(mc_token_ids[:spilt_len]), shuffle(mc_token_ids[spilt_len:])
    lm_labels_train, lm_labels_test = shuffle(lm_labels[:spilt_len]), shuffle(lm_labels[spilt_len:])
    mc_labels_train, mc_labels_test = shuffle(mc_labels[:spilt_len]), shuffle(mc_labels[spilt_len:])
    token_type_ids_train, token_type_ids_test = shuffle(token_type_ids[:spilt_len]), shuffle(token_type_ids[spilt_len:])

    epochs = 5
    sample_num = 0
    for epoch in range(epochs):
        logging.info(f"Training epoch {epoch}")
        logging.info("*" * epoch)
        for ith in range(len(input_ids_train)):
            logging.info(f"Training {ith}th examples")
            logging.info("#" * sample_num)
            # Training
            model.train()
            loss = forward(input_ids_train[ith], mc_token_ids_train[ith], lm_labels_train[ith],
                           mc_labels_train[ith], token_type_ids_train[ith])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sample_num += 1
            if sample_num % 1000:
                # Validate
                model.eval()
                from random import randint
                index = randint(len(input_ids_test))
                val_loss = forward(input_ids_test[index], mc_token_ids_test[index], lm_labels_test[index],
                                   mc_labels_test[index], token_type_ids_train[index])
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'val_loss': val_loss
                }, "saved_models/model" + str(sample_num % 1000))
                logging.info("Saving model ...")
                logging.info(f"Training loss - {loss}")
                logging.info(f"Validation loss - {val_loss}")


if __name__ == '__main__':
    train()