from transformers import BertForMaskedLM
from transformers import RobertaForMaskedLM
from transformers import XLMRobertaForMaskedLM

from transformers import XLMRobertaConfig
from transformers import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from torch.nn import CrossEntropyLoss
import copy
import numpy as np

import config as c

class BertForDynamicMLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, batch, eval = False):
        #all_input_ids, all_masked_input_ids all_attention_masks, all_token_type_ids
        if not eval:
            masked_input = copy.deepcopy(batch[0])
            for i in range(len(masked_input)):    
                for j in range(1, len(batch[0][i]) - 1):
                    if masked_input[i][j] == c.sep_token_id:
                        break
                    p = np.random.rand()
                    if p < c.mask_probability:
                        masked_input[i][j] = c.mask_token_id

        masked_input_ids = batch[1] if eval else masked_input
        
        outputs = super().forward(masked_input_ids, batch[2], batch[3])
        mask = masked_input_ids.eq(c.mask_token_id)
        
        labels = batch[0][mask]
        preds = outputs[0][mask]

        loss = self.loss_fn(preds, labels)

        outputs = (loss, ) + outputs
        return outputs

class RobertaForDynamicMLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, batch, eval = False):
        #all_input_ids, all_masked_input_ids all_attention_masks, all_token_type_ids
        if not eval:
            masked_input = copy.deepcopy(batch[0])
            for i in range(len(masked_input)):    
                for j in range(1, len(batch[0][i]) - 1):
                    if masked_input[i][j] == c.sep_token_id:
                        break
                    p = np.random.rand()
                    if p < c.mask_probability:
                        masked_input[i][j] = c.mask_token_id

        masked_input_ids = batch[1] if eval else masked_input
        
        outputs = super().forward(masked_input_ids, batch[2])
        mask = masked_input_ids.eq(c.mask_token_id)
        
        labels = batch[0][mask]
        preds = outputs[0][mask]

        loss = self.loss_fn(preds, labels)

        outputs = (loss, ) + outputs
        return outputs

class XLMRobertaForDynamicMLM(RobertaForDynamicMLM):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP