###########################################################
# DATA PREPROCESSING
## These are picked up by starter_data.py
########################################################### 

# which pretrained transformer you're preparing the data for
# primarily needed to pick the tokenizer
original_transformer = "bert-base-cased" 

# padding and default values, don't change 
pad_value = -2
def_value = -1

# Max sequence length in terms of the subword tokens
max_length = 128

# for parsing (where we need to reconstruct words from subwords): max. sentence length in terms of words, 
# longer than this will be eliminated from the corpus
max_word_len = 105

# Should in principle not be changed (adds special tokens [CLS] and [SEP] to the encoded sequences)
add_special_tokens = True

# whether to lowercase the texts before processing (e.g., if you plan to use the transfotrmer pretrained on lowercased data)
preprocessing_lowercase = False

# if you're preparing a dataset for MLM-ing, needs to be set to True
mlm = True

# probability of subword token masking
mask_probability = 0.15

# you can (but don't have to) set the maximal number of tokens that can be masked in a single sentence 
#max_masked_per_sent = 20

# the size of the hidden layer of the transformer you intend to use
hidden_size = 768

# path to the folder where the input data is and in which the serialized output data will be stored
base_path = "" # E.g., #"/work/gglavas/data/ud-treebanks-v2.5/UD_English-EWT"

# Name of the input file to be processed, within the base_path (relative to base_path)
in_file = "" # E.g., en_ewt-ud-train.conllu

# Name of the output file to be generated, in which the serialization of the data (to be consumed by the model in training/inference) will be stored
# also relative to the base_path
out_file = "" # e.g., "serialized/en_ewt-ud-train.bert_cased.ser"

# For parsing only: name of the file in which to save the (pickled) dictionary of all dependency relation that appear in treebank
deps_dict_path  = "" # e.g., "deps_dict.pkl"



###################################################################################################################
###################################################################################################################
###################################################################################################################



###########################################################
# MODELING
## These are picked up by starter.py and trainer.py
########################################################### 
task_type = "parsing" # "seq_class" for classification tasks like "nli" or "paws"; "mlm" or "parsing"
task = "parsing" # "nli", "mlm", "parsing", "copa", "paws", "csve", "xhate", "hans", ...
# (depending on the task and type of task, different things/model instances are created in starter.py and different training/eval protocols ran in trainer.py)

# adapter-based model (True) or full fine-tuning (False)
adapter = True

original_transformer = "bert-base-cased" # for multilingual stuff, commonly used 'xlm-roberta-base'
# indicates which original transformer was your model based on
pretrained_transformer = "bert-base-cased" # if your training procedure starts from a vanilla pretrained transformer, then it should be the same as "original_transformer" above
# However, if you want to continue training from some checkpoint saved from some other (intermediate) training, then you should set it to the local path of your checkpoint, e.g.,    
# "/work/gglavas/models/xhate/base/enbert_cased/trac/1/best"

suffix = ("enbert_cased" if original_transformer == "bert-base-cased" else 
         "mbert_cased" if original_transformer == "bert-base-multilingual-cased" else 
         "roberta" if original_transformer == "roberta-base" else 
         "xlmr" if original_transformer == "xlm-roberta-base" else "")


###########################################################
# DATA (for training and/or inference)
###########################################################

base_path = "" # Path to the folder in which you've serialized your data (train, dev, and test) portions, 
# e.g., "/work/gglavas/data/app-specific/hans-nli/serialized/" or "/work/gglavas/data/app-specific/xnli/"

# Indicates whether you're running the trainer.py for training a model (set to "train") or just for inference (set "test")
train_or_test = "test"

# names of the train, validation, and test portions within your base_path folder
# in starter.py the code combines these two paths, e.g., os.path.join(c.base_path, c.train_set)
train_set = "" # E.g., "hans.train.roberta.ser"
val_set = "" # E.g., hans.dev.roberta.ser" 
test_set = "" # "hans.test.bert.ser"

###########################################################
# Optimization details
###########################################################

# on which device to run, "cpu" or "cude:X"
device = "cuda:0"

# maximal number of training epochs
num_train_epochs = 100

# early stopping if no improvement after this many consecutive evaluations on the development set
num_evals_early_stop = 10

# gradient accumulation (if you cannot fit big batches on your GPU; if you can fit decent size batches, keep the value to 1)
gradient_accumulation_steps = 1

# Evaluation metric for measuring the performance on the development set for early stopping 
# for parsing, set to "UAS" or "LAS"
eval_stop_metric = "UAS" # for classification task, set to "Accuracy"

# Whether the metric is an increasing (higher values mean better performance) or decreasing (lower values indicate better performance) one 
eval_metric_increasing = True

# batch size for validation, typically can be significantly larger than in training, as there's only the forward pass, no gradient computation
eval_batch_size = 256

# learning rate
learning_rate = 1e-5 

# dropout in training; at inference it is automatically set with model.eval(), no need to change here if you're doing only inference
drpt = 0.1

# batch size in training
train_batch_size = 32

# random seed, to be changed if multiple experiments that account for random initalization effects are to be considered
seed = 42

# If you want to run a grid search over the parameters: grid_search = True must be set in that case
grid_search = True

# Output folder in which training abnd validation runs of different hyperparameter configurations will be stored 
# E.g. "/work/gglavas/models/hans/" + "original" + "/" + "roberta/"
# In this folder, the subfolders with individual configurations will be iteratively created: "1", "2", ...
grid_search_base_output_directory = "" 

# after how many update steps to checkpoint the model and measure the validation performance
grid_base_logging_steps = 100

# base batch size for the training within the grid
grid_base_batch_size = 32

# the grid itself: for each hypeparam, add the values you wanna examine
grid = { "lr" : [1e-5, 2e-5, 5e-6], 
         "drpt" : [0.1, 0.2], 
         "batch" : [16, 32],
         "seed" : [42, 84, 21]}

# other lower-level optimization details, typically no need to change

weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
max_steps = -1
warmup_steps = 0