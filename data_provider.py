import codecs
from numpy import random
import transformers
from transformers.data.processors import InputFeatures
import torch
from torch.utils.data import TensorDataset
import helper
import numpy as np
import copy
import config as c
import re
from sys import stdin

class InputFeaturesMLM(InputFeatures):
    """
    A single set of features for a MCQA data point: [CLS] premise + question [SEP] answer_i [SEP], for each answer "i".
    """
    def __init__(self, input_ids, masked_input_ids, attention_mask=None, token_type_ids=None, word_start_positions = None, real_len = None, label=None):
        super().__init__(input_ids, attention_mask, token_type_ids, label)
        self.masked_input_ids = masked_input_ids


class InputFeaturesWordsMap(InputFeatures):
    """
    A single set of features for a MCQA data point: [CLS] premise + question [SEP] answer_i [SEP], for each answer "i".
    """
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, word_start_positions = None, real_len = None, label=None):
        super().__init__(input_ids, attention_mask, token_type_ids, label)
        self.word_start_positions = word_start_positions
        self.real_len = real_len 


class InputFeaturesDepParseInstance(object):
    """
    A single set of features for a MCQA data point: [CLS] premise + question [SEP] answer_i [SEP], for each answer "i".
    """
    def __init__(self, input_feats, labs_arc, labs_rel):
        self.input_features = input_feats
        self.labels_arc = labs_arc
        self.labels_rel = labs_rel

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def add_example(sentences, sent_heads, sent_relations, sent, deps):
    sentences.append(sent)

    heads = [c.pad_value] * (c.max_word_len)
    rels = [c.pad_value] * (c.max_word_len)
    for k in deps:
        heads[k] = deps[k]["head"]
        rels[k] = deps[k]["rel"]

    sent_heads.append(heads)
    sent_relations.append(rels)

def get_dep_rels(path, dep_rels_dict):
    lines = [l.split("\t") for l in helper.load_lines(path)]

    for i in range(len(lines)):
        line = lines[i]
        if is_int(line[0]):
            if line[7] not in dep_rels_dict:
                dep_rels_dict[line[7]] = len(dep_rels_dict)


def load_ud_treebank(path, deps_dict, max_word_len, dep_rel_augment = False):
    if not deps_dict:
        deps_dict = {}

    sentences = []
    sent_heads = []
    sent_rels = []

    lines = [l.split("\t") for l in helper.load_lines(path)]
    
    sent = []
    deps = {}
            
    for i in range(len(lines)):
        line = lines[i]
        if line[0].startswith("#"):
            if len(sent) > 0: 
                if len(sent) < max_word_len:
                    add_example(sentences, sent_heads, sent_rels, sent, deps)
                else:
                    print("Long sentence: " + str(len(sent)))
                sent = []
                deps = {}

        elif is_int(line[0]):
            if dep_rel_augment:
                if line[7] not in deps_dict:
                    deps_dict[line[7]] = len(deps_dict)

            sent.append(line[1])
            deps[int(line[0]) - 1] = { "head" : int(line[6]), "rel" : (deps_dict[line[7]] if dep_rel_augment else (deps_dict[line[7]] if line[7] in deps_dict else deps_dict["<unk>"]))}

    if len(sent) > 0 and len(sent) <= max_word_len:
       add_example(sentences, sent_heads, sent_rels, sent, deps)
            
    return sentences, sent_heads, sent_rels, deps_dict

def language_specific_preprocessing(lang, sent):
    replace_dict = {"º" : "o", "ª" : "a", "²" : "2", "³" : "3", "¹" : "1", "\u200b" : "", "\u200d" : "", "…" : "...", "µ" : "μ", "r̝" : "r", "ˢ" : "s", 
                    "½" : "1/2", "´" : "'", "Ã¯" : "Ã", "11 3" : "113", "˝" : "\"", "ﬂ" : "fl", "？" : "?", "！" : "!", "。" : ".", "，" : ",", 
                    "）" : ")", "（" : "(", "：" : ":", "Ｂ" : "B", "ＯＫ" : "OK", '＋' : '+', 'Ｄ' : "D", "№" : "No", "™" : "TM", "\ufeff" : "", "¾" : "3/4", 
                    "Ǩusṫa" : "Kusṫa", "₂" : "2", '；' : ";", "\u200e" : "", "อำ" : "อํา", "สำ" : "สํา", "คำ" : "คํา", "จำ" : "จํา", "กำ" : "กํา", "ร่ำ" : "ร่ํา", "ทำ" : "ทํา", 
                    "น้ำ" : "น้ํา", "ตำ" : "ตํา", "ดำ" : "ดํา", "งำ" : "งํา", "นำ" :  "นํา", "ต่ำ" : "ต่ํา", "ซ้ำ" : "ซ้ํา", "ย้ำ" : "ย้ํา", "ว่ำ" : "ว่ํา", "ม่ำ" : "ม่ํา", "ลำ" : "ลํา", 
                    "ยำ" : "ยํา", "ย่ำ" : "ย่ํา", "รำ" : "รํา", "ชำ" : "ชํา", "ล่ำ" : "ล่ํา", "ค่ำ" : "ค่ํา", "ค้ำ" : "ค้ํา", ": )" : ":)", "ㄷ" : "ᄃ", "⸢" : "Γ", "⸣" : "Γ", "ḫ" : "h", 
                    "₄" : "4", "₅" : "5", "₁" : "1", "Ḫ" : "H", "₆" : "6", "ᾧ" : "ω", "ὧ" : "ω", "ᾷ" : "α", "ἣ" : "η", "ἳ" : "ι", "ὦ" : "ω", "Ἴ" : "I", "ἲ" : "ι", 
                    "ᾖ" : "η", "Ὑ" : "Y", "ὣ" : "ω", "Ἵ" : "I", "ῄ" : "η", "ῴ" : "ω", "ὤ" : "ω", "ᾐ" : "η", "ὓ" : "ν", "ᾔ" : "η", "ἃ" : "α", "ᾗ" : "η", "Ἤ" : "H",
                    "ᾅ" : "α", "Ὡ" : "Ω", "ὢ" : "ω", "Ῥ" : "P", "ἆ" : "α", "ᾄ" : "α", "ᾠ" : "ω", "Ἥ" : "H", "Ὄ" : "O", "ὒ" : "ν", "Ὕ" : "Y", "Ἲ" : "I", "Ἶ" : "I", 
                    "ῒ" : "ι", "Ἦ" : "H", "Ὠ" : "Ω", "ῂ" : "η", "Ἦ" : "H", "ᾑ" : "η", "Ἢ" : "H", "ῢ" : "ν", "Ὥ" : "Ω", "ὂ" : "ο", "ᾴ" : "α", "Ὦ" : "Ω", "％" : "%", 
                    "Ⅲ" : "III", "℃" : "°C", "և" : "եւ", "\u200c" : "", "ǹ" : "n", "Ǹ" : "N", "\xa0" : "", "㎞" : "km"}
    
    nospacelangs = ["zh"]

    for i in range(len(sent)):
        for k in replace_dict:
            if k in sent[i]:
                sent[i] = sent[i].replace(k, replace_dict[k])
                
            m = re.search('([0-9]\s+[0-9])', sent[i])
            if m:
                orig = m.group(0)
                rep = orig.replace(" ", "")
                sent[i] = sent[i].replace(orig, rep)

        if lang in ["vi", "sv", "kk", "lt", "kmr", "br"] and " " in sent[i]:
            sent[i] = sent[i].replace(" ", "")

    if lang in nospacelangs:
        return ("").join(sent)
    else:
        return (" ").join(sent)
    

def featurize_serialize_ud(path, sentences, arc_labels, rel_labels, tokenizer, lang, is_roberta = False, on_the_fly = False):
    featurized_examples = []
    examples = list(zip(sentences, arc_labels, rel_labels))
    counter = 0
    for sent, al, rl in examples:
        #xlm_roberta_zh = "xlm-roberta" in c.pretrained_transformer and c.is_zh

        instance_text = featurize_text_parsing(language_specific_preprocessing(lang, sent), sent, tokenizer, c.max_length, c.add_special_tokens, has_toktype_ids = not is_roberta)
        if not instance_text:
            counter += 1
            continue

        instance = InputFeaturesDepParseInstance(instance_text, al, rl)
        featurized_examples.append(instance)
    
    all_input_ids = torch.tensor([fe.input_features.input_ids for fe in featurized_examples], dtype=torch.long)
    if not is_roberta:
        all_token_type_ids = torch.tensor([fe.input_features.token_type_ids for fe in featurized_examples], dtype=torch.long)
    all_attention_masks = torch.tensor([fe.input_features.attention_mask for fe in featurized_examples], dtype=torch.long)
    all_word_start_positions = torch.tensor([fe.input_features.word_start_positions for fe in featurized_examples], dtype=torch.long)
    all_lengths = torch.tensor([fe.input_features.real_len for fe in featurized_examples], dtype=torch.long)
    all_labels_arcs = torch.tensor([fe.labels_arc for fe in featurized_examples], dtype=torch.long)
    all_labels_rels = torch.tensor([fe.labels_rel for fe in featurized_examples], dtype=torch.long)

    if is_roberta:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_word_start_positions, all_lengths, all_labels_arcs, all_labels_rels) 
    else:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_word_start_positions, all_lengths, all_labels_arcs, all_labels_rels) 
    
    print("Instances added: " + str(len(all_input_ids)))
    print("Instances skipped: " + str(counter))

    if on_the_fly: 
        return dataset
    else:
        torch.save(dataset, path)
    #if counter > 0:
    #    stdin.readline()


def get_pure_subword_string(subword_string):
    if c.pretrained_transformer.startswith("bert-"):
        return (subword_string[2:] if subword_string.startswith("##") else subword_string)
    if c.pretrained_transformer.startswith("roberta-"):
        return (subword_string[1:] if subword_string.startswith("Ġ") else subword_string)
    if c.pretrained_transformer.startswith("xlm-roberta-"):
        return (subword_string[1:] if subword_string.startswith("▁") else subword_string)

def get_word_start_positions(word_tokens, subword_tokens):
    if 'การเปลี่ยน' in word_tokens:
        z = 7

    positions = []
    
    index_subwords = 1
    index_words = 0
    
    string_subword = ""
    string_word = ""

    starter_subwords = 1

    prolong_words = False
    prolong_subwords = False
    
    while True:
        if (index_subwords >= len(subword_tokens) - 1) or (index_words >= len(word_tokens)):
            break
        
        if subword_tokens[index_subwords] == '▁':
            if starter_subwords == index_subwords:
                starter_subwords += 1
            index_subwords += 1
            continue

        string_subword += get_pure_subword_string(subword_tokens[index_subwords])
        string_word += word_tokens[index_words]

        if string_word == string_subword:
            prolong_subwords = False
            prolong_words = False

            positions.append(starter_subwords)
            
            index_subwords += 1
            starter_subwords = index_subwords

            index_words += 1
            string_subword = ""
            string_word = ""

        # main approach, for most languages
        elif string_word.startswith(string_subword):
            # handling horrible cases like: ['餘下', '的'] vs. ['餘', '下的'] 
            if prolong_words:
                w = word_tokens[index_words]
                sw = subword_tokens[index_subwords]
                for i in range(len(sw)):
                    if sw[i] == w[0]:
                        string_subword = sw[i:]
                        break
                starter_subwords = index_subwords + 1

            index_subwords += 1
            string_word = ""
    
            prolong_subwords = True
            prolong_words = False

        # for Chinese-like problem, if subword token contains more than one word token
        elif string_subword.startswith(string_word):      
            positions.append(starter_subwords)

            # handling horrible cases like: ['餘', '下的'] vs. ['餘下', '的']
            if prolong_subwords:
                w = word_tokens[index_words]
                sw = subword_tokens[index_subwords]
                for i in range(len(w)):
                    if w[i] == sw[0]:
                        string_word = w[i:]
                        break
                starter_subwords = index_subwords
            
            string_subword = ""
            index_words += 1

            prolong_subwords = False
            prolong_words = True
            
        else:
            print("Non-matching strings between accumulations of subword-level and word-level tokens")
            return None, -1
            #raise ValueError("Non-matching strings between accumulations of subword-level and word-level tokens")
   
    real_len = len(positions) 

    if real_len != len(word_tokens):
        return None, -1
        #raise ValueError("Tokenization mismatch")

    positions.append(len(subword_tokens) - 1)
    extension = [-1] * (c.max_word_len + 1 - len(positions))
    positions.extend(extension)
    return positions, real_len

def get_word_start_positions_old(word_tokens, subword_tokens):
    xlmr = "xlm-roberta" in c.pretrained_transformer

    positions = []
    running_position_start = 0
    running_string = ""
    running_word_index = 0

    for i in range(1, len(subword_tokens) - 1):
        if running_string == "":
            running_position_start = i

        running_string += get_pure_subword_string(subword_tokens[i])
        if running_string.strip() == (word_tokens[running_word_index]).strip() or (xlmr and c.is_zh and running_string == "," and word_tokens[running_word_index] == "，"):
            positions.append(running_position_start)

            running_word_index += 1
            running_string = ""
    
    real_len = len(positions) 

    if real_len != len(word_tokens):
        raise ValueError("Tokenization mismatch")

    positions.append(len(subword_tokens) - 1)
    extension = [-1] * (c.max_word_len + 1 - len(positions))
    positions.extend(extension)
    return positions, real_len

def featurize_text_parsing(text, word_tokens, tokenizer, max_length = 510, add_special_tokens = True, label = None, has_toktype_ids = True):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length)
    subword_strings = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    word_start_positions, real_len = get_word_start_positions(word_tokens, subword_strings) 
    if not word_start_positions:
        return None

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if has_toktype_ids:
        token_type_ids = inputs["token_type_ids"]

    # Zero-pad up to the sequence length.
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    if has_toktype_ids:
        token_type_ids = token_type_ids + ([0] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)

    if has_toktype_ids:
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    return InputFeaturesWordsMap(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids if has_toktype_ids else None, word_start_positions=word_start_positions, real_len = real_len, label = label)


### PAWS

def load_paws(path):
    lines = helper.load_lines(path)[1:]
    lines = [l.split("\t") for l in lines]

    examples = []
    for l in lines:
        ex = {}
        ex["sent1"] = l[0].lower() if c.preprocessing_lowercase else l[1]
        ex["sent2"] = l[1].lower() if c.preprocessing_lowercase else l[2]
        ex["label"] = int(l[3])
        examples.append(ex)

    return examples

def feature_serialize_paws(examples, path, tokenizer):
    featurized_examples = []
    is_roberta = "roberta" in c.pretrained_transformer
    cnt = 0
    for ex in examples:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        instance = featurize_text_simple((ex["sent1"], ex["sent2"]), tokenizer, label = ex["label"], is_text_pair = True, has_toktype_ids=not is_roberta)
        featurized_examples.append(instance)
    
    print("Max length: " + str(max(lengths_dataset)))
    all_input_ids = torch.tensor([fe.input_ids for fe in featurized_examples], dtype=torch.long)
    if not is_roberta:
        all_token_type_ids = torch.tensor([fe.token_type_ids for fe in featurized_examples], dtype=torch.long)
    all_attention_masks = torch.tensor([fe.attention_mask for fe in featurized_examples], dtype=torch.long)
    all_labels = torch.tensor([fe.label for fe in featurized_examples], dtype=torch.long)

    if is_roberta:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels) 
    else:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels) 
    torch.save(dataset, path)

### NLI (HANS)

def load_nli_dataset(path):
    lines = helper.load_lines(path)[1:]
    lines = [l.split("\t") for l in lines]

    examples = []
    for l in lines:
        ex = {}
        ex["sent1"] = l[5].lower() if c.preprocessing_lowercase else l[5]
        ex["sent2"] = l[6].lower() if c.preprocessing_lowercase else l[6]
        ex["label"] = l[0]
        examples.append(ex)
    return examples

def featurize_serialize_nli(examples, path_out, tokenizer):
    featurized_examples = []
    is_roberta = "roberta" in c.pretrained_transformer

    for ex in examples:
        premise = ex["sent1"]
        hypothesis = ex ["sent2"]
        label = 0 if (ex["label"] == "neutral" or ex["label"] == "non-entailment") else (1 if ex["label"] == "entailment" else 2)

        instance = featurize_text_simple((premise, hypothesis), tokenizer, label = label, is_text_pair = True, has_toktype_ids = not is_roberta)
        featurized_examples.append(instance)
    
    all_input_ids = torch.tensor([fe.input_ids for fe in featurized_examples], dtype=torch.long)
    if not is_roberta:
        all_token_type_ids = torch.tensor([fe.token_type_ids for fe in featurized_examples], dtype=torch.long)
    all_attention_masks = torch.tensor([fe.attention_mask for fe in featurized_examples], dtype=torch.long)
    all_labels = torch.tensor([fe.label for fe in featurized_examples], dtype=torch.long)

    if is_roberta:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels) 
    else:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels) 
    torch.save(dataset, path_out)

### GENERAL

lengths_dataset = []
def featurize_text_simple(text, tokenizer, label = None, is_text_pair = False, has_toktype_ids = True):
    if is_text_pair:
        t1, t2 = text
        inputs = tokenizer.encode_plus(t1, t2, add_special_tokens=c.add_special_tokens, max_length=c.max_length)
    else:
        inputs = tokenizer.encode_plus(text, add_special_tokens=c.add_special_tokens, max_length=c.max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if has_toktype_ids:
        token_type_ids = inputs["token_type_ids"]

    lengths_dataset.append(len(input_ids))
        
    # Zero-pad up to the sequence length.
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_length = c.max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    if has_toktype_ids:
        token_type_ids = token_type_ids + ([0] * padding_length)

    assert len(input_ids) == c.max_length, "Error with input length {} vs {}".format(len(input_ids), c.max_length)
    assert len(attention_mask) == c.max_length, "Error with input length {} vs {}".format(len(attention_mask), c.max_length)

    if has_toktype_ids:
        assert len(token_type_ids) == c.max_length, "Error with input length {} vs {}".format(len(token_type_ids), c.max_length)

    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids if has_toktype_ids else None, label = label)

### MLM

def featurize_serialize_mlm(sentences, path, tokenizer):
    featurized_examples = []
    is_roberta = "roberta" in c.pretrained_transformer
    cnt = 0
    for sent in sentences:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        instance = featurize_text_simple(sent, tokenizer, is_text_pair = False, has_toktype_ids=not is_roberta)
        masked_input_ids = copy.deepcopy(instance.input_ids)

        # masking randomly with probability given with c.mask_probability
        for i in range(1, len(instance.input_ids) - 1):
            if instance.input_ids[i] == tokenizer.convert_tokens_to_ids(tokenizer.sep_token):
                break
            p = np.random.rand()
            if p < c.mask_probability:
                mask_tok_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                masked_input_ids[i] = mask_tok_id
        
        real_inst = InputFeaturesMLM(instance.input_ids, masked_input_ids, instance.attention_mask, None if is_roberta else instance.token_type_ids)
        featurized_examples.append(real_inst)
    
    print("Max length: " + str(max(lengths_dataset)))

    all_input_ids = torch.tensor([fe.input_ids for fe in featurized_examples], dtype=torch.long)
    all_masked_input_ids = torch.tensor([fe.masked_input_ids for fe in featurized_examples], dtype=torch.long)
    if not is_roberta:
        all_token_type_ids = torch.tensor([fe.token_type_ids for fe in featurized_examples], dtype=torch.long)
    all_attention_masks = torch.tensor([fe.attention_mask for fe in featurized_examples], dtype=torch.long)
    
    if is_roberta:
        dataset = TensorDataset(all_input_ids, all_masked_input_ids, all_attention_masks) 
    else:
        dataset = TensorDataset(all_input_ids, all_masked_input_ids, all_attention_masks, all_token_type_ids)
    torch.save(dataset, path)

### General single text classification

def load_textclass_dataset(path, skip_first = False, delimiter = "\t", text_index = 0, label_index = 1, lab_map = None):
    lines = helper.load_lines(path)
    if skip_first:
        lines = lines[1:]
    lines = [l.split(delimiter) for l in lines]

    examples = []
    for l in lines:
        ex = {}
        ex["text"] = l[text_index].lower() if c.preprocessing_lowercase else l[text_index]
        ex["label"] = int(l[label_index]) if not lab_map else lab_map[l[label_index]]
        examples.append(ex)

    return examples

def feature_serialize_textclass_dataset(examples, path, tokenizer):
    featurized_examples = []
    is_roberta = "roberta" in c.pretrained_transformer
    cnt = 0
    for ex in examples:
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
        instance = featurize_text_simple(ex["text"], tokenizer, label = ex["label"], is_text_pair = False, has_toktype_ids=not is_roberta)
        featurized_examples.append(instance)
    
    print("Max length: " + str(max(lengths_dataset)))
    all_input_ids = torch.tensor([fe.input_ids for fe in featurized_examples], dtype=torch.long)
    if not is_roberta:
        all_token_type_ids = torch.tensor([fe.token_type_ids for fe in featurized_examples], dtype=torch.long)
    all_attention_masks = torch.tensor([fe.attention_mask for fe in featurized_examples], dtype=torch.long)
    all_labels = torch.tensor([fe.label for fe in featurized_examples], dtype=torch.long)

    if is_roberta:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels) 
    else:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels) 
    torch.save(dataset, path)

    ### XHate

def load_xhate_mlm_corpus(path, filt = True):
    lines = helper.load_lines(path)
    if filt:
        sents = [l.split(";;;")[1] for l in lines]
    else:
        sents = [l for l in lines if len(l) > 10]
    return sents


