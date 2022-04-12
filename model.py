from asyncore import dispatcher
import logging
import torch
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from transformers import AdamW
from data_processing import *

history, reply = get_history_reply_pairs()
examples_len = len(history)
spilt_len = 0.8 * examples_len

model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

SPECIAL_TOKENS = [
  "<bos>",
  "<eos>",
  "<customer>",
  "<assistant>",
  "<pad>",
  "<br>"
  ]
tokenizer.set_special_tokens(SPECIAL_TOKENS)
model.set_num_special_tokens(len(SPECIAL_TOKENS))


def build_trainging_tensor_data(history, reply, distractor):
    words, words_distractor, segments, segments_distractor, \
    lm_targets, lm_distractor, last_token, last_token_distractor = \
        build_training_data(history, reply, distractor)

    # And gather reply and distractor inputs to build the input tensors:
    # words tokens
    input_ids = torch.tensor([*words, *words_distractor], dtype=torch.long)
    print(f"input_ids        shape({input_ids.size()})")
    # segment tokens
    token_type_ids = torch.tensor([*segments, *segments_distractor], dtype=torch.long)
    print(f"segment tokens   shape({token_type_ids.size()})")
    # Positions tokens can be automatically created by the model as (0, 1, ..., N)
    # Last tokens location
    mc_token_ids = torch.tensor([last_token, last_token_distractor], dtype=torch.long)
    print(f"mc_token_ids     shape({mc_token_ids.size()})")
    # Language modeling labels
    lm_labels = torch.tensor([*lm_targets, *lm_distractor], dtype=torch.long)
    print(f"lm_labels        shape({lm_labels.size()})")
    # Next-sentence prediction labels
    mc_labels = torch.tensor([0], dtype=torch.long)  # Gold reply is 1st (index 0)
    print(f"mc_labels        shape({mc_labels.size()})")
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
    """
    print(f"words len({len(words)})                  {words}")
    print(f"words distractor len({len(words_distractor)})       {words_distractor}")
    print(f"segments len({len(segments)})               {segments}")
    print(f"segments distractor len({len(segments_distractor)})    {segments_distractor}")
    print(f"lm target len({len(lm_targets)})              {lm_targets}")
    print(f"lm distractor len({len(lm_distractor)})          {lm_distractor}")
    print(f"last token type({type(last_token)})             {last_token}")
    print(f"last token distractor len({type(last_token_distractor)})  {last_token_distractor}")
    """
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
    from random import shuffle
    # Generate a random list of index
    indexes = shuffle(i for i in range(0, examples_len))
    train_indexes = indexes[...:spilt_len]
    test_indexes = indexes[spilt_len:...]
    logging.info("Preparing training and testing data ...")

    epochs = 5
    sample_num = 0
    for epoch in range(epochs):
        logging.info(f"Training epoch {epoch}")
        logging.info("*" * epoch)
        for train_index in train_indexes:
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = \
                build_trainging_tensor_data(history[train_index], reply[train_index], choose_distractor())
            # Training
            model.train()
            loss = forward(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sample_num += 1
            if sample_num % 1000:
                # Validate
                model.eval()
                from random import choice
                test_index = choice(test_indexes)
                input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = \
                    build_trainging_tensor_data(history[test_index], reply[test_index], choose_distractor())
                val_loss = forward(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)
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