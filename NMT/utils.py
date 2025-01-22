from typing import Dict, Tuple
import torch
import time
import math
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datasets import load_dataset # type: ignore
from transformers import AutoTokenizer  # type: ignore
from tokenizers import decoders # type: ignore
import evaluate as hfeval # type: ignore

#####################
#Data loading methods

def encode_trans(examples:Dict, input_tokenizer, target_tokenizer, sentence_length:int)-> Dict:
  examples = examples["translation"]
  ens = []
  des = []
  for ex in examples:
      ens.append(ex['en'])
      des.append(ex['de'])
  inputs = input_tokenizer(ens, padding='longest', truncation=True, max_length=sentence_length)
  targets = target_tokenizer(des, padding='longest', truncation=True, max_length=sentence_length)
  return {'input': inputs["input_ids"], "target": targets["input_ids"]}

def collate_custom(batch) -> Tuple[torch.Tensor, torch.Tensor]:
  inputs, targets = [] ,[]
  for batchi in batch:
    inputs.append(batchi["input"])
    targets.append(batchi["target"])
  return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def get_dataloader(split:str, input_tokenizer, target_tokenizer, batch_size:int, sentence_length:int, streaming: bool=False, shard:int|None=None) -> DataLoader:
    """
    Returns the wmt16 dataset.
    params:
    split: str indicates what split of the dataset to return
    > get_dataloader("train")
    """
    #TODO stream is too slow?
    if streaming:
        dataset_origin = load_dataset("wmt16", "de-en", streaming=streaming, split=split, trust_remote_code=True)
    else:
        dataset_origin = load_dataset("wmt16", "de-en", split=split)
        if shard is not None:
            dataset_origin = dataset_origin.shard(shard,1)
    dataset_m = dataset_origin.map(lambda x: encode_trans(x, input_tokenizer, target_tokenizer, sentence_length= sentence_length), remove_columns="translation", batched=True, batch_size=batch_size)
    return DataLoader(dataset_m, collate_fn=collate_custom, batch_size=batch_size)

######
# Logging methods

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

###############################
#Training and evaluating methods

def train_epoch(dataloader, modelNMT, model_optimizer, criterion, device, print_every:int|None=100):
    total_loss = 0
    start = time.time()
    dataset_size = len(dataloader)
    for batch_num, data in enumerate(dataloader):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        model_optimizer.zero_grad()
        output = modelNMT((input_tensor, target_tensor))
        loss = criterion(output.view(-1, output.shape[-1]), target_tensor.view(-1))
        loss.backward()

        model_optimizer.step()
        
        total_loss += loss.item()
        if print_every is not None:
            if batch_num % print_every == 0:
                print_loss_avg = total_loss / (batch_num+1)
                print(f"Time {timeSince(start, (batch_num+1) / dataset_size)}, on batch {(batch_num+1)},\
                progress: {(batch_num+1) / dataset_size * 100}%, accumulated loss: {total_loss}, avg loss{print_loss_avg}, last loss: {loss.item()}")

    return total_loss / len(dataloader)

def evaluate(dataloader, model, criterion, device, print_every:int|None=100)-> float:
    """_summary_

    Args:
        dataloader (_type_): validation / test dataloader
        model (_type_): _description_
        criterion (_type_): _description_
        device (_type_): _description_
        print_every (int | None, optional): _description_. Defaults to 100.

    Returns:
        float: average loss
    """
    total_loss = 0
    start = time.time()
    dataset_size = len(dataloader)
    for batch_num, data in enumerate(dataloader):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        output = model((input_tensor, target_tensor))
        loss = criterion(output.view(-1, output.shape[-1]), target_tensor.view(-1))    
        total_loss += loss.item()
        #TODO count correct tokens (?)
        if print_every is not None:
            if batch_num % print_every == 0:
                print_loss_avg = total_loss / (batch_num+1)
                print(f"Time {timeSince(start, (batch_num+1) / dataset_size)}, on batch {(batch_num+1)},\
                progress: {(batch_num+1) / dataset_size * 100}%, accumulated loss: {total_loss}, avg loss{print_loss_avg}, last loss: {loss.item()}")

    return total_loss / len(dataloader)

def train(train_dataloader, modelNMT, n_epochs:int,
                valid_dataloader,
                learning_rate:float=0.001,
                print_every:int=100, plot_every:int=100,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    model_optimizer = optim.Adam(modelNMT.parameters(), lr=learning_rate)
    # TODO add different optimizers as parameter for easier set up
    criterion = nn.NLLLoss()
    # TODO look at NLLLoss ignore_index= param for adjusting the weight of token classes like <EOS> etc.

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, modelNMT, model_optimizer, criterion, device)
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
            
         # evaluate on the validation set
        average_loss = evaluate(valid_dataloader, modelNMT, criterion, device)
        #valid_ppl = math.exp(min(valid_loss, 100))
        print(f"Validation loss: {average_loss}")       
        ### TODO save checkpoints
        
    #TODO plot?
    #showPlot(plot_losses)
    
def translate(nmtModel, input, target, target_tokenizer) -> str:
    with torch.no_grad():
        output_tensor = nmtModel((input,target))
        #[batch, sentence_length, dict_size]
        output_sentence_ids = torch.topk(output_tensor,k=1)
        # target_tokenizer.decoder = decoders.WordPiece()
        decoded_words = target_tokenizer.decode(output_sentence_ids)
        
    return decoded_words

def run_test(nmtModel:nn.Module, test_dataloader, input_tokenizer, tgt_tokenizer, print_sentences=False):
    print("Evaluating model on test set:")
    predictions = []
    references = []
    for input, target in test_dataloader:
        #TODO iterate over batch
        output_sentence = translate(nmtModel, input, target, tgt_tokenizer)
        #TODO output_words works, accumulate, 
        references.append(input)
        predictions.append(output_sentence)
        if print_sentences:
          print('>', input)
          print('=', target)
          print(f"<{output_sentence}\n")

    bleu = hfeval.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print("Metric scores:")
    print(f"Corpus BLEU Score on test set: {bleu_score}")


