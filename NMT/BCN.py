from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from NMT.maxout import Maxout
from NMT.models import NMTModel
from NMT import constants 

class BCN(nn.Module):
    """Implementation of Biattentive Classification Network in
    Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    for text classification"""

    def __init__(self, config: dict, converter:nn.Module, num_labels:int, dict_length:int,
                 embedding_type:str="random", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(config, dict), "config must be a dictionary"
        self.word_vec_size = config.get('word_vec_size', 300)
        self.mtlstm_hidden_size = config.get("nmt_hidden_size", 300)
        self.fc_hidden_size = config.get('fc_hidden_size', 128)
        self.bilstm_encoder_size = config.get('bilstm_encoder_size', 256)
        self.bilstm_integrator_size = config.get('bilstm_integrator_size', 256)
        self.dropout = config.get('dropout', 0.1)
        self.pool_size = config.get("maxout_channels", 4)
        self.device = config.get('device', 'cpu')
        self.embedding_type = embedding_type
        if embedding_type == "glove":
            # Converter is matrix embedding
            self.encoder = converter
            self.encoded_size = self.word_vec_size
        elif embedding_type == "cove":
            # Converter is nmtModel
            self.encoder = converter.encoder
            self.encoded_size = self.word_vec_size + self.mtlstm_hidden_size
        else:
            # assuming random
            self.encoded_size = self.mtlstm_hidden_size
            self.encoder = nn.Embedding(num_embeddings=dict_length,
                                        embedding_dim=self.encoded_size,
                                        padding_idx=constants.PAD)  
            
        self.fc = nn.Linear(self.encoded_size, self.fc_hidden_size)

        self.bilstm_encoder = nn.LSTM(self.fc_hidden_size,
                                      self.bilstm_encoder_size // 2,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=True,
                                      dropout=config['dropout'])

        self.bilstm_integrator = nn.LSTM(self.bilstm_encoder_size * 3,
                                         self.bilstm_integrator_size // 2,
                                         num_layers=1,
                                         batch_first=True,
                                         bidirectional=True,
                                         dropout=config['dropout'])
        # TODO 2 sentences case requires 2 integrators:
        """
        self.bilstm_integrator_2 = nn.LSTM(self.bilstm_encoder_size * 3,
                                         self.bilstm_integrator_size // 2,
                                         num_layers=1,
                                         batch_first=True,
                                         bidirectional=True,
                                         dropout=config['dropout'])
        """
        self.attentive_pooling_proj = nn.Linear(self.bilstm_integrator_size,
                                                1)
        self.pool_size = config["maxout_channels"]
        self.dropout_pool = nn.Dropout(config['dropout'])
        self.bn1 = nn.BatchNorm1d(self.bilstm_integrator_size * 4)
        self.maxout1 = Maxout(self.bilstm_integrator_size * 4, self.bilstm_integrator_size * 4 // 4, self.pool_size)
        self.dropout_m1 = nn.Dropout(config['dropout'])
        self.maxout2 = Maxout(self.bilstm_integrator_size * 4 // 4,
                              self.bilstm_integrator_size * 4 // 4 // 4, self.pool_size)
        self.bn2 = nn.BatchNorm1d((self.bilstm_integrator_size * 4) // 4)
        self.dropout_m2 = nn.Dropout(config['dropout']) 
        
        #TODO add an extra //4 for the last layer output size and to the classifier input size
        #self.dropout_m3 = nn.Dropout(config['dropout']) 
        #self.bn3 = nn.BatchNorm1d((self.bilstm_integrator_size * 4) // 4)
        #self.maxout3 = Maxout(self.bilstm_integrator_size * 4 // 4,
        #                      self.bilstm_integrator_size * 4 // 4 // 4, self.pool_size)
        self.classifier = nn.Linear((self.bilstm_integrator_size * 4) // 4 // 4, num_labels)
        
    def get_pad_mask(self, input:torch.Tensor, hidden_size:int, pad_token_id:int=constants.PAD)-> torch.Tensor:
        """Returns a tensor with 1 where there are words and 0 where there are Padding ids
        Args:
            input (torch.Tensor): _description_
            hidden_size (int): _description_
            pad_token_id (int, optional): _description_. Defaults to constants.PAD.

        Returns:
            torch.Tensor: Tensor same shape as input or [input, hidden_size]
        """
        pad_mask = input.clone().detach()
        pad_mask = (~(pad_mask == pad_token_id)).to(torch.uint8)
        if hidden_size == 1:
            pad_mask = pad_mask
        else:
            pad_mask = pad_mask.unsqueeze(2).expand(pad_mask.size(0),
                                                  pad_mask.size(1),
                                                  hidden_size)
        return pad_mask

    def forward(self, input_sentences:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_
        Args:
            input_sentence (Tensor): sentences to be classified
        Returns:
            torch.Tensor: class logits [num_classes]
            torch.Tensor: probability distribution over the number the of classes [num_classes]
        """
        reshaped: bool = False
        if len(input_sentences.shape)<2: #if not batched
          input_sentences = input_sentences.unsqueeze(0)
          reshaped = True

        # MTLSTM vectors
        if self.embedding_type == "cove":
            with torch.no_grad():
                glove = self.encoder.embedding(input_sentences)
                dropped1 = self.encoder.dropout_l(glove)
                cove, _ = self.encoder.LSTM(dropped1)
                cove = self.encoder.dropout_l2(cove)
                encoded = torch.cat([dropped1, cove], dim=-1)
        elif self.embedding_type == "glove":
            with torch.no_grad():
                encoded = self.encoder(input_sentences)
        else:
            encoded = self.encoder(input_sentences)
            

        #Task specific f: ff + ReLU Network
        task_specific_reps = F.relu(self.fc(encoded))
        #Encoder (equation 7,8)
        encoded_tokens_XY, _ = self.bilstm_encoder(task_specific_reps) # [batch, x,input_encoder_size]
    
        # Affinity Matrix. This is a special case since the inputs are the same.
        attention_logits_A = encoded_tokens_XY.bmm(encoded_tokens_XY.permute(0, 2, 1).contiguous())

        # Extract Attention weights. Equation 9
        # masked-softmax: turn small values into very negative ones (-Inf) so that they can be zeroed during softmax
        attention_mask1 = attention_logits_A.clone().detach()
        attention_mask1 = -1e32 * (attention_mask1 <= 1e-7).float()
        masked_attention_logits = attention_logits_A + attention_mask1  # mask logits that are near zero
        masked_Ax = F.softmax(masked_attention_logits, dim=1)  # prerform column-wise softmax
        masked_Ay = F.softmax(masked_attention_logits.permute(0, 2, 1), dim=-1)
        attention_mask2 = attention_logits_A.clone().detach()
        attention_mask2 = (attention_mask2 >= 1e-7).float()
        
        Ax = masked_Ax * attention_mask2
        Ay = masked_Ay * attention_mask2
        # Context summary Cx Cy (equation 10)
        Cx = torch.bmm(Ax.permute(0, 2, 1), encoded_tokens_XY)  # batch_size * max_len * bilstm_encoder_size
        Cy = torch.bmm(Ay.permute(0, 2, 1), encoded_tokens_XY)

        # Integrate (equation 11, 12)
        integrator_input = torch.cat([encoded_tokens_XY,
                                      encoded_tokens_XY - Cy,
                                      encoded_tokens_XY * Cy], 2)
        outputs_Xy_Yx, _ = self.bilstm_integrator(integrator_input)  # batch_size * max_len * bilstm_integrator_size
        # TODO 2 sentences case:
        """integrator_Y_input = torch.cat([encoded_tokens_XY,
                                      encoded_tokens_XY - Cx,
                                      encoded_tokens_XY * Cx], 2)
        outputs_Yx, _ = self.bilstm_integrator_2(integrator_input_Y)  # batch_size * max_len * bilstm_integrator_size"""
        # Simple Pooling layers
        # Replace masked values so they don't interfere with pooling values
        pad_mask = self.get_pad_mask(input_sentences, self.bilstm_integrator_size)
        max_masked_Xy = outputs_Xy_Yx + -1e7 * (1 - pad_mask)
        max_pool = torch.max(max_masked_Xy, 1)[0]
        min_masked_Xy = outputs_Xy_Yx + 1e7 * (1 - pad_mask)
        min_pool = torch.min(min_masked_Xy, 1)[0]
        mean_pool = torch.sum(outputs_Xy_Yx, 1) / torch.sum(self.get_pad_mask(input_sentences, 1),
                                                 1,
                                                 keepdim=True)

        # Self-attentive pooling layer (Equation 13)
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.attentive_pooling_proj(outputs_Xy_Yx)
        self_attentive_logits = torch.squeeze(self_attentive_logits) \
                                + -1e32 * (1 - self.get_pad_mask(input_sentences, 1)) #masked softmax
        self_weights = F.softmax(self_attentive_logits, dim=-1)
        # Equation 14
        self_attentive_pool = self_weights.unsqueeze(1).bmm(outputs_Xy_Yx).squeeze(1)
        # Equation 15, 16
        pooled_representations = torch.cat([max_pool,
                                            min_pool,
                                            mean_pool,
                                            self_attentive_pool], 1)
        pooled_representations_dropped = self.dropout_pool(pooled_representations)
        
        # Batch Normalized MaxOut (x2)
        bn_pooled = self.bn1(pooled_representations_dropped)
        max_out1 = self.maxout1(bn_pooled)
        max_out1_dropped = self.dropout_m1(max_out1)
        
        bn_max_out1 = self.bn2(max_out1_dropped)
        max_out2 = self.maxout2(bn_max_out1)
        max_out2_dropped = self.dropout_m2(max_out2)
        
        #bn_max_out2 = self.bn3(max_out2_dropped)
        #max_out3 = self.maxout3(bn_max_out2)
        #max_out3_dropped = self.dropout_m3(max_out3)

        logits = self.classifier(max_out2_dropped)
        class_probabilities = F.softmax(logits, dim=-1)
        return logits, class_probabilities
