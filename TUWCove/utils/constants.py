PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

MAX_SEQ_LEN = 40
MAX_VOCAB_SIZE = 50000
EMBEDDING_DIM = 300
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'

#Training

# Paths
GLOVE_PATH = "./data/glove.6B.300d.txt"
VOCAB_PATH = "./data/prep/vocab.json"
EMBEDDINGS_PATH = "./data/prep/combined_embeddings.pth"
TRAIN_SRC_PATH = "./data/prep/train.tok.en"
TRAIN_TGT_PATH = "./data/prep/train.tok.de"
VAL_SRC_PATH = "./data/prep/val.tok.en"
VAL_TGT_PATH = "./data/prep/val.tok.de"
TEST_SRC_PATH = "./data/prep/test.tok.en"
TEST_TGT_PATH = "./data/prep/test.tok.de"
MODEL_SAVE_PATH = "./checkpoints/"

# Hyperparameters
EMBEDDING_DIM = 300
RNN_SIZE = 300
DROPOUT = 0.2
NUM_LAYERS = 2
BIDIRECTIONAL = True
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

#TODO properly define