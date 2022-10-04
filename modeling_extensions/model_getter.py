from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import XLMRobertaTokenizer

from transformers import BertConfig
from transformers import RobertaConfig
from transformers import XLMRobertaConfig

from .modeling_biaffine import BertForBiaffineParsing
from .modeling_biaffine import RobertaForBiaffineParsing
from .modeling_biaffine import XLMRobertaForBiaffineParsing

from .modeling_adapters import BottleneckAdapterBertConfig
from .modeling_adapters import BottleneckAdapterRobertaConfig
from .modeling_adapters import BottleneckAdapterXLMRobertaConfig

from .modeling_adapters import AdapterBertForBiaffineParsing
from .modeling_adapters import AdapterRobertaForBiaffineParsing
from .modeling_adapters import AdapterXLMRobertaForBiaffineParsing

from transformers import BertForSequenceClassification
from transformers import RobertaForSequenceClassification
from transformers import XLMRobertaForSequenceClassification

from .modeling_adapters import AdapterBertForSequenceClassification
from .modeling_adapters import AdapterRobertaForSequenceClassification
from .modeling_adapters import AdapterXLMRobertaForSequenceClassification

from .modeling_mcqa import BertForMultichoiceQA
from .modeling_mcqa import RobertaForMultichoiceQA
from .modeling_mcqa import XLMRobertaForMultichoiceQA

from .modeling_adapters import AdapterBertForMultichoiceQA
from .modeling_adapters import AdapterRobertaForMultichoiceQA
from .modeling_adapters import AdapterXLMRobertaForMultichoiceQA

from .modeling_mlm import BertForDynamicMLM
from .modeling_mlm import RobertaForDynamicMLM
from .modeling_mlm import XLMRobertaForDynamicMLM

from .modeling_adapters import AdapterBertForDynamicMLM
from .modeling_adapters import AdapterRobertaForDynamicMLM
from .modeling_adapters import AdapterXLMRobertaForDynamicMLM

from .modeling_mlm import BertForDynamicMLM

import config as c

def get_tokenizer():
    if c.original_transformer.startswith("bert-"):
        return BertTokenizer.from_pretrained(c.original_transformer)

    elif c.original_transformer.startswith("roberta-"):
        return RobertaTokenizer.from_pretrained(c.original_transformer)

    elif c.original_transformer.startswith("xlm-roberta-"):
        return XLMRobertaTokenizer.from_pretrained(c.original_transformer)
    else:
        raise ValueError("Not supported!")

def get_task_type_model(path, config):
    if c.original_transformer.startswith("bert-"):
        if c.adapter:
            if c.task_type == "parsing":
                return AdapterBertForBiaffineParsing.from_pretrained(path, config = config)
            elif c.task_type == "seq_class":
                return AdapterBertForSequenceClassification.from_pretrained(path, config = config)
            elif c.task_type == "mcqa":
                return AdapterBertForMultichoiceQA.from_pretrained(path, config = config)
            elif c.task_type == "mlm":
                return AdapterBertForDynamicMLM.from_pretrained(path, config = config)
        else:
            if c.task_type == "parsing":
                return BertForBiaffineParsing.from_pretrained(path, config = config)
            elif c.task_type == "seq_class":
                return BertForSequenceClassification.from_pretrained(path, config = config)
            elif c.task_type == "mcqa":
                return BertForMultichoiceQA.from_pretrained(path, config = config)
            elif c.task_type == "mlm":
                return BertForDynamicMLM.from_pretrained(path, config = config)

    elif c.original_transformer.startswith("roberta-"):
        if c.adapter:
            if c.task_type == "parsing":
                return AdapterRobertaForBiaffineParsing.from_pretrained(path, config = config)
            elif c.task_type == "seq_class":
                return AdapterRobertaForSequenceClassification.from_pretrained(path, config = config)
            elif c.task_type == "mcqa":
                return AdapterRobertaForMultichoiceQA.from_pretrained(path, config = config)
            elif c.task_type == "mlm":
                return AdapterRobertaForDynamicMLM.from_pretrained(path, config = config)
        else:
            if c.task_type == "parsing":
                return RobertaForBiaffineParsing.from_pretrained(path, config = config)
            elif c.task_type == "seq_class":
                return RobertaForSequenceClassification.from_pretrained(path, config = config)
            elif c.task_type == "mcqa":
                return RobertaForMultichoiceQA.from_pretrained(path, config = config)
            elif c.task_type == "mlm":
                return RobertaForDynamicMLM.from_pretrained(path, config = config)

    elif c.original_transformer.startswith("xlm-roberta-"):
        if c.adapter:
            if c.task_type == "parsing":
                return AdapterXLMRobertaForBiaffineParsing.from_pretrained(path, config = config)
            elif c.task_type == "seq_class":
                return AdapterXLMRobertaForSequenceClassification.from_pretrained(path, config = config)
            elif c.task_type == "mcqa":
                return AdapterXLMRobertaForMultichoiceQA.from_pretrained(path, config = config)
            elif c.task_type == "mlm":
                return AdapterXLMRobertaForDynamicMLM.from_pretrained(path, config = config)
        else:
            if c.task_type == "parsing":
                return XLMRobertaForBiaffineParsing.from_pretrained(path, config = config)
            elif c.task_type == "seq_class":
                return XLMRobertaForSequenceClassification.from_pretrained(path, config = config)
            elif c.task_type == "mcqa":
                return XLMRobertaForMultichoiceQA.from_pretrained(path, config = config)
            elif c.task_type == "mlm":
                return XLMRobertaForDynamicMLM.from_pretrained(path, config = config)
    else:
        raise ValueError("Not supported!")

def get_task_type_config(path):
    if c.original_transformer.startswith("bert-"):
        if c.adapter:
            return BottleneckAdapterBertConfig.from_pretrained(path)
        else:
            return BertConfig.from_pretrained(path)

    elif c.original_transformer.startswith("roberta-"):
        if c.adapter:
            return BottleneckAdapterRobertaConfig.from_pretrained(path)
        else:
            return RobertaConfig.from_pretrained(path)

    elif c.original_transformer.startswith("xlm-roberta-"):
        if c.adapter:
            return BottleneckAdapterXLMRobertaConfig.from_pretrained(path)
        else:
            return XLMRobertaConfig.from_pretrained(path)
    else:
        raise ValueError("Not supported!")