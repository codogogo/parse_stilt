import logging
import torch
import itertools
import os
import helper
import torch
import sys
import data_provider
import trainer
from modeling_extensions import model_getter
import config as c

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if c.grid_search:
    hyp_configs = list(itertools.product(c.grid["lr"], c.grid["drpt"], c.grid["batch"], c.grid["seed"]))

best_eval_perf = -1000000 if c.eval_metric_increasing else 1000000
best_hyp_config = None
best_hyp_config_number = None

if c.train_or_test == "train":
    if not isinstance(c.train_set, list):
        train_data = torch.load(os.path.join(c.base_path, c.train_set))
    else:
        tr_sets = []
        for p in c.train_set:
            tr = torch.load(os.path.join(c.base_path, p))
            tr_sets.append(tr)
        train_data = torch.utils.data.dataset.ConcatDataset(tr_sets)

    if not isinstance(c.val_set, list):
        eval_data = torch.load(os.path.join(c.base_path, c.val_set))
    else:
        v_sets = []
        for p in c.val_set:
            val = torch.load(os.path.join(c.base_path, p))
            v_sets.append(val)
        eval_data = torch.utils.data.dataset.ConcatDataset(v_sets)

elif c.train_or_test == "test":
    test_data = torch.load(os.path.join(c.base_path, c.test_set))
else:
    raise ValueError("Unknown mode")

if c.task == "parsing":
    deps_dict = helper.deserialize(os.path.join(c.base_path, c.deps_dict_path)) 
    c.num_labels = len(deps_dict)
elif c.task == "nli" or c.task == "siqa":
    c.num_labels = 3
elif c.task == "copa" or c.task == "paws" or c.task == "csve" or c.task == "xhate" or c.task == "hans":
    c.num_labels = 2
elif c.task == "mlm":
    c.num_labels = 0
else:
    raise ValueError("Not supported!")

hyper_conf_counter = 0
for hc in hyp_configs:
    hyper_conf_counter += 1
    print("##########################################################################")
    print("Hyperparameter configuration " + str(hyper_conf_counter) + ": " + str(hc))
    print("##########################################################################")

    c.learning_rate, c.drpt, c.train_batch_size, c.seed = hc
    c.output_dir = os.path.join(c.grid_search_base_output_directory, str(hyper_conf_counter))

    c.logging_steps = int(c.grid_base_logging_steps * (c.grid_base_batch_size / c.train_batch_size))
    print(c.logging_steps)

    if c.train_or_test == "train":
        if not os.path.exists(c.output_dir):
            os.makedirs(c.output_dir)
        helper.write_list(os.path.join(c.output_dir, "config.txt"), [str(s) for s in list(hc)])
        # Set seed
        trainer.set_seed(c.seed)
    elif c.train_or_test == "test":
        c.output_dir = c.pretrained_transformer

    # Loading tokenizer, configuration, and model
    tokenizer = model_getter.get_tokenizer()
    if c.task == "mlm":
        c.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        c.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    match = [f for f in os.listdir(c.output_dir) if f.startswith("best")]
    if len(match) > 0:
        c.model_name_or_path = os.path.join(c.output_dir, 'best')
    else:
        c.model_name_or_path = c.pretrained_transformer

    continuing = c.model_name_or_path != c.pretrained_transformer
    config = model_getter.get_task_type_config(c.model_name_or_path)
    if not continuing:        
        config.num_labels = c.num_labels
        config.last_layer_dropout = c.drpt

    model = model_getter.get_task_type_model(c.model_name_or_path, config)
    #model.half()
    if not hasattr(c, "large_model_gpu_split_layer"):
        model.to(c.device)

    #model.classifier.to("cuda:0")

    if c.task_type != "parsing" and c.task_type != "mlm" and c.adapter:
        print("Making all params trainable...")
        for p in model.parameters():
            p.requires_grad = True

    # important for not continuing from the checkpoint of model trained on the previous task
    if c.train_or_test == "train":
        c.model_name_or_path = os.path.join(c.output_dir, 'best')

    logger.info("Training/evaluation starts...")
    if c.train_or_test == "train":
        print(model)
        _, _, eval_perf = trainer.train(train_data, eval_data, model)
    else:
        results = trainer.evaluate(test_data, model)
        print(results)
        #preds = results["Preds"]
        #helper.write_list(c.preds_out_path, preds)
        exit()

    if (c.eval_metric_increasing and eval_perf > best_eval_perf) or (not c.eval_metric_increasing and eval_perf < best_eval_perf):
        best_eval_perf = eval_perf
        best_hyp_config = hc
        best_hyp_config_number = hyper_conf_counter

        print("New best hyperparameter configuration: " + str(hc))
        print("Eval performance: " + str(best_eval_perf))
    
    del model
    del tokenizer
    torch.cuda.empty_cache()

print("Best hyperparameter configuration (" + str(best_hyp_config_number) + ") : " + str(best_hyp_config))
print("Eval performance: " + str(best_eval_perf))