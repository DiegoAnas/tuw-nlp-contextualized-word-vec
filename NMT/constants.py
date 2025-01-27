PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

MAX_SEQ_LEN = 256  # Maximum sequence length for tokenized input
DEFAULT_BATCH_SIZE = 32  # Default batch size for DataLoader
LEARNING_RATE = 0.001  # Default learning rate for training
DROPOUT = 0.2  # Dropout ratio applied in models

DEFAULT_VOCAB_SIZE = 50000  # Default vocabulary size for tokenizers
MIN_FREQ = 1  # Minimum frequency for a word to be included in the vocabulary

# Constants for BLEU evaluation
BLEU_NGRAM_ORDER = 4  # Maximum n-gram order for BLEU score calculation
