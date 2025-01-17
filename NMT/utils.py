from typing import Tuple
import torch
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datasets import load_dataset # type: ignore
from transformers import AutoTokenizer  # type: ignore

#Data loading methods

def encode_trans(examples, input_tokenizer, target_tokenizer):
  examples = examples["translation"]
  ens = []
  des = []
  for ex in examples:
    # possible filter short sentences so no padding is needed
      ens.append(ex['en'])
      des.append(ex['de'])
  inputs = input_tokenizer(ens, padding='longest', truncation=True, max_length=40)
  targets = target_tokenizer(des, padding='longest', truncation=True, max_length=40)
  return {'input': inputs["input_ids"], "target": targets["input_ids"]}

def collate_custom(batch):
  inputs = batch[0]["input"]
  targets = batch[0]["target"]
  return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def get_dataloader(split:str, input_tokenizer, target_tokenizer, batch_size:int) -> DataLoader:
    """
    Returns a streaming version of the wmt16 dataset.
    params:
    split: str indicates what split of the dataset to return
    > get_dataloader("train")
    """
    dataset_stream = load_dataset("wmt16", "de-en", streaming=True, split=split, trust_remote_code=True)
    dataset_batched = dataset_stream.batch(batch_size=batch_size)
    dataset_m = dataset_batched.map(lambda x: encode_trans(x, input_tokenizer, target_tokenizer),remove_columns="translation")
    return DataLoader(dataset_m, collate_fn=collate_custom)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#Training and evaluating methods

def train_epoch(dataloader, modelNMT, model_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        model_optimizer.zero_grad()
        
        output = modelNMT(input_tensor)
        
        loss = criterion(output, target_tensor)
        loss.backward()

        model_optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, modelNMT, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    model_optimizer = optim.Adam(modelNMT.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, modelNMT, model_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    #TODO plot?
    #showPlot(plot_losses)
    
def translate(nmtModel, sentence, input_tokenizer, target_tokenizer):
    with torch.no_grad():
        input_tensor = input_tokenizer(sentence)

        output_tensor = nmtModel(input_tensor)

        decoded_words = target_tokenizer.decode(output_tensor)
    return decoded_words

def run_evaluation(nmtModel:nn.Module, val_dataloader, input_lang, output_lang, input_tokenizer, tgt_tokenizer, print_sentences=False):
    #TODO incorporate input/target tokenizers
    #(encoder:nn.Module, decoder:nn.Module, pairs:Tuple[str,str], input_lang:Lang, output_lang:Lang, tokenizers=None, print_sentences=False):
    #TODO huggingface has gleu and bleu, remove hfeval ¿and nltk.translate?
    #import hfeval
    #from nltk.translate import gleu_score
    print("Evaluating model on test set:")
    predictions = []
    predictions_tokenized = []
    references = []
    references_tokenized = []
    for pair in val_dataloader:
        output_words = translate(nmtModel, pair[0], input_tokenizer, tgt_tokenizer)
        references.append(pair[1])
        references_tokenized.append(pair[1].split(" "))
        output_sentence = ' '.join(output_words)
        predictions.append(output_sentence)
        predictions_tokenized.append(output_words)
        if print_sentences:
          print('>', pair[0])
          print('=', pair[1])
          print(f"<{output_sentence}\n")

    #bleu = hfeval.load("bleu")
    #bleu = hfeval.load("bleu", smoothing=True) # same as sacrebleu with default params
    #sacrebleu = hfeval.load("sacrebleu") # default smooth_method="exp"

    #bleu_score = bleu.compute(predictions=predictions, references=references)
    #sacrebleu_score = sacrebleu.compute(predictions=predictions, references=references)
    #gleu_number = gleu_score.corpus_gleu(list_of_references = references_tokenized, hypotheses= predictions_tokenized, min_len=1, max_len=4)

    print("Metric scores:")
    #print(f"Corpus BLEU Score on test set: {bleu_score}")
    #print(f"Corpus SacreBLEU Score on test set: {sacrebleu_score}")
    #print(f"Corpus GLEU Score on test set: {gleu_number}")


