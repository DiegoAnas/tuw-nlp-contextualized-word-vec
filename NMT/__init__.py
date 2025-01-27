# Import core modules
import NMT.constants
import NMT.models
import NMT.modules

# Import additional components
from NMT.encoder import Encoder
from NMT.decoder import Decoder
from NMT.maxout import Maxout
from NMT.utils import (
    encode_trans,
    collate_custom,
    get_dataloader,
    train,
    evaluate,
    train_epoch,
    translate,
    run_test,
    timeSince,
    asMinutes
)

# Expose primary classes and functions
__all__ = [
    # Constants
    "constants",
    
    # Core Models
    "Encoder",
    "Decoder",
    "NMTModel",
    "Maxout",
    
    # Dropout
    "nn.Dropout",  # Expose the dropout functionality explicitly if needed

    # Utilities
    "encode_trans",
    "collate_custom",
    "get_dataloader",
    "train",
    "evaluate",
    "train_epoch",
    "translate",
    "run_test",
    "timeSince",
    "asMinutes"
]
