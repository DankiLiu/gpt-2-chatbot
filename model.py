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


def build_trainging_data_batch(batch_size, history, reply):
    words, words_distractor, segments, segments_distractor, \
    lm_targets, lm_distractor, last_token, last_token_distractor = [], [], [], [], [], [], [], []

    for i in range(batch_size):
        distractor = choose_distractor()
        wrds, wrds_dis, seg, seg_dis, lm_trgs, lm_dis, las, las_dis = build_training_data(history[i], reply[i], distractor)
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
    def update():
        optimizer.zero_grad()
        model.train()
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = build_trainging_data_batch(batch_size=200)
        loss = forward(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)
        loss.backward()
        optimizer.step()
