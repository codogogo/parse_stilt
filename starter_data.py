import data_provider
import os
import config as c
import helper
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import XLMRobertaTokenizer
import random

if c.original_transformer.startswith("bert-"):
    tokenizer = BertTokenizer.from_pretrained(c.pretrained_transformer)
if c.original_transformer.startswith("roberta-"):
    tokenizer = RobertaTokenizer.from_pretrained(c.pretrained_transformer)
if c.original_transformer.startswith("xlm-roberta"):
    tokenizer = XLMRobertaTokenizer.from_pretrained(c.pretrained_transformer)


### NLI-HANS ###
# examples = data_provider.load_nli_dataset("/work/gglavas/data/app-specific/hans-nli/heuristics_evaluation_set.txt")
# #random.shuffle(examples)

# #train = examples[:28000]
# #dev = examples[28000:]

# #data_provider.featurize_serialize_nli(train, "/work/gglavas/data/app-specific/hans-nli/serialized/hans.train.bert.ser", tokenizer)
# #data_provider.featurize_serialize_nli(dev, "/work/gglavas/data/app-specific/hans-nli/serialized/hans.dev.bert.ser", tokenizer)
# data_provider.featurize_serialize_nli(examples, "/work/gglavas/data/app-specific/hans-nli/serialized/hans.test.roberta.ser", tokenizer)

#### PARSING / MLM ######
if not c.mlm:
    deps_dict = None
    if os.path.exists(os.path.join(c.base_path, c.deps_dict_path)):
        deps_dict = helper.deserialize(os.path.join(c.base_path, c.deps_dict_path)) 

sentences, arc_labels, rel_labels, deps_dict = data_provider.load_ud_treebank(os.path.join(c.base_path, c.in_file), None if c.mlm else deps_dict, c.max_word_len)
print("Num. sentences: " + str(len(sentences)))
print("Num. dependency relations: " + str(len(deps_dict)))

if c.mlm:
    sentences = [" ".join(s) for s in sentences]

data_provider.featurize_serialize_mlm(sentences, os.path.join(c.base_path, c.out_file), tokenizer)
data_provider.featurize_serialize_ud(os.path.join(c.base_path, c.out_file), sentences, arc_labels, rel_labels, tokenizer, is_roberta = "roberta" in c.pretrained_transformer)

#if not os.path.exists(os.path.join(c.base_path, c.deps_dict_path)):
if not c.mlm:
    helper.serialize(deps_dict, os.path.join(c.base_path, c.deps_dict_path))

###### PAWS-X ######
#examples = data_provider.load_paws(os.path.join(c.base_path, c.in_file))
#print("Num. sentences: " + str(len(examples)))

# data_provider.feature_serialize_paws(examples, os.path.join(c.base_path, c.out_file), tokenizer)

# ###### XHATE ######
#examples = data_provider.load_textclass_dataset(os.path.join(c.base_path, c.in_file), skip_first = True)
#data_provider.feature_serialize_textclass_dataset(examples, os.path.join(c.base_path, c.out_file), tokenizer)

# ##### XHate MLM
# sentences = data_provider.load_xhate_mlm_corpus(os.path.join(c.base_path, c.in_file), filt = False)
# random.shuffle(sentences)
# if len(sentences) > 202000:
#     sentences = sentences[:202000]
# dev_size = 2000
# train_sents = sentences[:len(sentences)-dev_size]
# dev_sents = sentences[len(sentences)-dev_size:]
# print("Num. sentences: " + str(len(sentences)))
# data_provider.featurize_serialize_mlm(train_sents, os.path.join(c.base_path, c.out_file_train), tokenizer)
# data_provider.featurize_serialize_mlm(dev_sents, os.path.join(c.base_path, c.out_file_dev), tokenizer)
