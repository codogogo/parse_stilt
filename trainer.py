import argparse
import glob
import json
import logging
import os
import random
import helper
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from metric import Metric
import metric
import config as c
import math

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_optimizer_and_scheduler(model, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": c.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=c.learning_rate, eps=c.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=c.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(c.model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(c.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(c.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(c.model_name_or_path, "scheduler.pt")))

    return optimizer, scheduler

def save_model(model, optimizer, scheduler, step, dev_scores, additional_eval = False):
    # Save model checkpoint

    output_dir = os.path.join(c.output_dir, "best") if not additional_eval else os.path.join(c.output_dir, "addit_best") 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = (model.module if hasattr(model, "module") else model)  
    model_to_save.save_pretrained(output_dir)

    #torch.save(c, os.path.join(output_dir, "training_c.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    global_step_path = os.path.join(output_dir, "global_step.txt")
    helper.write_list(global_step_path, [str(step)])

    if dev_scores:
        eval_output_dir = os.path.join(c.output_dir, "eval")
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        output_eval_file = os.path.join(eval_output_dir, "eval_result.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(c.task + " " + c.val_set))
            for key in sorted(dev_scores.keys()):
                logger.info("  %s = %s", key, str(dev_scores[key]))
                writer.write("%s = %s\n" % (key, str(dev_scores[key])))

    logger.info("Saving optimizer and scheduler states to %s", output_dir) 

def train(train_dataset, eval_dataset, model, additional_eval = None):
    """ Train the model """
    
    helper.touch(os.path.join(c.output_dir, "train_log.txt"))

    # enabling TensorBoard
    tb_writer = SummaryWriter(c.output_dir)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=c.train_batch_size)

    if c.max_steps > 0:
        t_total = c.max_steps
        c.num_train_epochs = c.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * c.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = prepare_optimizer_and_scheduler(model, t_total)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", c.num_train_epochs)
    logger.info("  Train batch size = %d", c.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(c.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(helper.load_lines(os.path.join(c.model_name_or_path, "global_step.txt"))[0].strip())
        except ValueError:
            global_step = 0
        epochs_trained = global_step // len(train_dataloader)
        steps_trained_in_current_epoch = global_step % len(train_dataloader)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(epochs_trained, int(c.num_train_epochs), desc="Epoch", disable=False)

    set_seed(c.seed)  # Added here for reproductibility, every new training starts from the same seed, i.e., same parameter initialization

    eval_steps_no_improvement = 0
    stop_training = False
    best_eval_res = -1000000 if c.eval_metric_increasing else 1000000
    if additional_eval:
        best_addit_res = -1000000 if c.eval_metric_increasing else 1000000

    if (global_step > 0):
        eval_res_path = os.path.join(c.model_name_or_path.replace("best", "eval"), "eval_result.txt")
        if os.path.exists(eval_res_path):
            res = {x.split("=")[0].strip() : float(x.split("=")[1].strip()) for x in helper.load_lines(eval_res_path)}
            best_eval_res = res[c.eval_stop_metric]

    print("To start training...")    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        real_batch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            #print("Global step: " + str(global_step))
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train() 
            batch = tuple(t.to(c.device) for t in batch)

            if c.task_type == "mlm":
                outputs = model(batch, eval = False)

            elif c.task_type != "seq_class":
                outputs = model(batch)
        
            else:
                if c.original_transformer.startswith("bert-"):
                    outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])
                elif "roberta-" in c.original_transformer:
                    outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
                    
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tr_loss += loss.item()
            real_batch_loss += loss.item() / c.gradient_accumulation_steps
            
            # parameter updates
            if c.gradient_accumulation_steps > 1:
                loss = loss / c.gradient_accumulation_steps
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)

            if (global_step + 1) % c.gradient_accumulation_steps == 0:
                #print("Global step: " + str(global_step) + ", accumulated gradient update")
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                #print(scheduler.get_last_lr())    
                model.zero_grad() # zeroing gradients afer update

                #print("Batch loss: " + str(real_batch_loss))
                real_batch_loss = 0
            

            global_step += 1

            if not hasattr(c, "perform_training_evals") or c.perform_training_evals:
                if c.logging_steps > 0 and global_step % c.logging_steps == 0:
                    print("Global step: " + str(global_step))
                    logs = {}
                    results = evaluate(eval_dataset, model)
                    
                    if additional_eval:
                        results_additional = evaluate(additional_eval, model)
                        print("Additional results")
                        print(results_additional)

                        for k in list(results.keys()):
                            results[k + "_add"] = results_additional[k]


                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / c.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    lines_to_log = ["Step: " + str(global_step)]
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                        lines_to_log.append(key + ": " + str(value))
                    
                    lines_to_log.append(" ")
                    helper.write_list(os.path.join(c.output_dir, "train_log.txt"), lines_to_log, append = True)

                    print(json.dumps({**logs, **{"step": global_step}}))

                    eval_res = results[c.eval_stop_metric]
                    if (c.eval_metric_increasing and eval_res < best_eval_res) or (not c.eval_metric_increasing and eval_res > best_eval_res):
                        eval_steps_no_improvement += 1
                    else:
                        eval_steps_no_improvement = 0
                    
                    if additional_eval:
                        addit_res = results[c.eval_stop_metric + "_add"]
                        if (c.eval_metric_increasing and addit_res > best_addit_res) or (not c.eval_metric_increasing and addit_res < best_addit_res):
                            save_model(model, optimizer, scheduler, global_step, None, additional_eval=True)
                            best_addit_res = addit_res
                        

                    if eval_steps_no_improvement == c.num_evals_early_stop:
                        print("Early stopping training. ")
                        stop_training = True
                        break
                        
                    if eval_steps_no_improvement == 0:
                        best_eval_res = eval_res
                        print("New best eval " + c.eval_stop_metric + ": " + str(best_eval_res))
                        print("Saving best model...")
                        save_model(model, optimizer, scheduler, global_step, results)
                        print("New best model saved!")
                    else:
                        print("No improvement for " + str(eval_steps_no_improvement) + " steps!")
                        print("Current Eval " + c.eval_stop_metric + ": " + str(eval_res))
                        print("Best Eval " + c.eval_stop_metric + " so far: " + str(best_eval_res))
                    

            if c.max_steps > 0 and global_step > c.max_steps:
                epoch_iterator.close()
                break
        
        if (c.max_steps > 0 and global_step > c.max_steps) or stop_training:
            train_iterator.close()
            break
    
    if hasattr(c, "perform_training_evals") and not c.perform_training_evals:
        save_model(model, optimizer, scheduler, global_step, {})

    lines_to_log = [" ", "Best eval perf: " + str(best_eval_res)]
    if additional_eval:
        lines_to_log.append("Best addit perf: " + str(best_addit_res))
    helper.write_list(os.path.join(c.output_dir, "train_log.txt"), lines_to_log, append = True)

    tb_writer.close()

    return global_step, tr_loss / global_step, best_eval_res

def evaluate(eval_dataset, model):
    results = {}

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=c.eval_batch_size)
    
    # Eval!
    logger.info("***** Running evaluation {} *****".format("XCOPA VAL"))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", c.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    golds = None
    uaslas = None
         
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(c.device) for t in batch)
    
        with torch.no_grad():
            if c.task_type == "mlm":
                outputs = model(batch, eval = True)
            elif c.task_type != "seq_class":
                outputs = model(batch)
            else:
                if c.original_transformer.startswith("bert-"):
                    outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])
                elif "roberta-" in c.original_transformer:
                    outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])

            loss = outputs[0]

            if c.task_type == "parsing":
                labels_arc = batch[5] if c.original_transformer.startswith("bert-") else batch[4]
                labels_rels = batch[6] if c.original_transformer.startswith("bert-") else batch[5]
                mask = labels_arc.ne(c.pad_value)            
                
                arc_scores = outputs[-1]
                if len(arc_scores.shape) == 2:
                    arc_scores = arc_scores.unsqueeze(0)

                rel_scores = outputs[1]
                
                arc_preds, rel_preds = metric.decode(arc_scores, rel_scores, mask)

                if uaslas is None:
                    uaslas = Metric()
                uaslas(arc_preds, rel_preds, labels_arc, labels_rels, mask)

            elif c.task_type == "seq_class" or c.task_type == "mcqa":
                logits = outputs[1]
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    golds = batch[-1].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    golds = np.append(golds, batch[-1].detach().cpu().numpy())

            eval_loss += loss.mean().item()
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    results["Loss"] = eval_loss

    if c.task_type == "parsing":
        results["UAS"] = uaslas.uas
        results["LAS"] = uaslas.las

    elif c.task_type == "seq_class" or c.task_type == "mcqa":
        preds = np.argmax(preds, axis=1)

        if hasattr(c, "subtask") and c.subtask == "hans":
            preds = [(0 if x == 2 else x) for x in preds]

        #results["Preds"] = preds
        results["Accuracy"] = len([i for i in range(len(preds)) if preds[i] == golds[i]]) / len(preds)

        if c.task == "xhate" or c.subtask == "hans":
            tps = len([i for i in range(len(preds)) if preds[i] == 1 and golds[i] == 1])
            fps = len([i for i in range(len(preds)) if preds[i] == 1 and golds[i] == 0])
            fns = len([i for i in range(len(preds)) if preds[i] == 0 and golds[i] == 1])

            if tps + fps == 0:
                prec = 0
            else:
                prec = tps / (tps + fps)
            
            if tps + fns == 0:
                rec = 0
            else:
                rec = tps / (tps + fns)

            if prec + rec == 0:
                f = -1
            else:
                f = (2 * prec * rec) / (prec + rec)
                
            results["Precision"] = prec
            results["Recall"] = rec
            results["F1Bin"] = f
        
    return results