import codecs
import pickle
import os
import torch
from langvecs import hierarchy
import numpy as np
import math

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def serialize(item, path):
	pickle.dump(item, open(path, "wb" ))

def deserialize(path):
	return pickle.load(open(path, "rb" ))

def load_file(filepath):
	return (codecs.open(filepath, 'r', encoding = 'utf8', errors = 'replace')).read()

def load_lines(filepath):
	return [l.strip() for l in list(codecs.open(filepath, "r", encoding = 'utf8', errors = 'replace').readlines())]

def write_list(path, list, append = False):
	f = codecs.open(path,'w' if not append else 'a',encoding='utf8')
	for l in list:
		f.write(str(l) + "\n")
	f.close()

def get_subdirectories(path, recursive = False):
	if recursive:
		return [x[0] for x in list(os.walk(path))[1:]]
	else:
		return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

def get_files(path, recursive = False):
	if recursive:
		files = []
		for subdir in list(os.walk(path)):
			files.extend([os.path.join(subdir, x) for x in subdir[2]])
	else:
		return [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]



### traverse results for treebanks and get performance at checkpoints
def treebank_to_langcode_mapping(lang_hier):
	tbtolang = {}
	for lang in lang_hier:
		for test in lang_hier[lang]['test']:
			treebank = test.split("/")[-2]
			tbtolang[treebank] = lang
	return tbtolang

def get_train_information(run):
	base_path = "/".join(run.split("/")[:-2])

	train_info_dict = {} 
	train_set = torch.load(os.path.join(run, "train.ser")) 
	train_info_dict["size"] = len(train_set)
	
	lines_config = load_lines(os.path.join(run, "training_setup.txt"))
	for i in range(len(lines_config)):
		if lines_config[i].startswith("Train:"):
			train_treebanks = lines_config[i+1:]
			break

	train_info_dict["train_sets"] = []
	for ttb in train_treebanks:
		ttb_path = ttb.split("/")[-2]

		if os.path.exists(os.path.join(base_path, ttb_path, "1", "train.ser")):
			train_dataset = torch.load(os.path.join(base_path, ttb_path, "1", "train.ser"))
			train_info_dict["train_sets"].append((ttb_path, len(train_dataset)))
		else:
			train_file = [f for f in os.listdir(os.path.join("/work/gglavas/data/ud-treebanks-v2.5", ttb_path)) if "ud-train.txt" in f]
			if len(train_file) == 1: 
				size = len(load_lines(os.path.join("/work/gglavas/data/ud-treebanks-v2.5", ttb_path, train_file[0])))
				train_info_dict["train_sets"].append((ttb_path, size))
			else:
				train_info_dict["train_sets"].append((ttb_path, -1))

	return train_info_dict

def get_performance_information(run):
	run_perf = []
	content = load_lines(os.path.join(run, "train_log.txt")) 
	segments = [l for l in content if l.startswith("Step:")]
	for l in segments:
		i = content.index(l)
		step = int(l.split(":")[1].strip())
		uas_dev = float(content[i+2].split(":")[1].strip())
		las_dev = float(content[i+3].split(":")[1].strip())
		uas = float(content[i+5].split(":")[1].strip())
		las = float(content[i+6].split(":")[1].strip())
		run_perf.append((step, uas, las, uas_dev, las_dev))
	return run_perf


def get_all_checkpoint_performances(root_path, lang_rels_path):
	results = {}  

	treebank_dirs = get_subdirectories(root_path)
	for td in treebank_dirs:
		print(td + ": " + str(treebank_dirs.index(td) + 1))
		results[td] = {}
		results[td]['has_own_dev'] = os.path.exists(os.path.join(td, "has_own_dev.txt"))
		results[td]['runs'] = {}
		if results[td]['has_own_dev']:
			if os.path.exists(os.path.join(td, "best_setup.txt")):
				results[td]['best_setup'] = load_file(os.path.join(td, "best_setup.txt")).strip()
			else:
				print("Treebank not yet finished, skipping...")
				results.pop(td, None)
				continue

		run_dirs = get_subdirectories(td)
		for run in run_dirs:
			results[td]['runs'][run] = {}
			results[td]['runs'][run]['train'] = get_train_information(run)
			results[td]['runs'][run]['performance'] = get_performance_information(run)
			
	return results

def train_sample_target_weighted_sim(train_sets, treebank, tbtolang, lang_sims):
	weights = []
	sims = []
	for ts in train_sets:
		lang_target = tbtolang[treebank]
		lang_train = tbtolang[ts[0]]

		weights.append(ts[1])
		if lang_train == lang_target:
			sims.append(1)
		else:
			sims.append(lang_sims[lang_train][lang_target])

	weights = np.array(weights)
	weights = weights / np.sum(weights)
	return np.dot(weights, sims)


def select_run(results, treebank, tbtolang, lang_sims, dummy = False):
	runs = results[treebank]['runs']

	if dummy: 
		thold = 500
		for i in range(len(runs)):
			r = [x for x in list(runs.keys()) if x.endswith(str(i+1))][0]
			size = runs[r]['train']['size']	
			if size > thold:
				return r

	
	for run in runs:
		train_size = runs[run]['train']['size']
		train_sets = runs[run]['train']['train_sets']

		runs[run]['train']['sim'] = train_sample_target_weighted_sim(train_sets, treebank, tbtolang, lang_sims)

	# relative drop in the weighted averaged similarity has to be smaller or equal to the relative increase in the overall train set size
	prev_sim = 1
	prev_size = 1
	best_run = -1
	for i in range(len(runs)):
		r = [x for x in list(runs.keys()) if x.endswith(str(i+1))][0]
		sim_r = runs[r]['train']['sim']
		size_r = runs[r]['train']['size']

		if (1 - sim_r/prev_sim) < ((size_r/prev_size - 1)): #/ (1.2**(i+1))):
			prev_sim = sim_r
			prev_size = size_r
			best_run = r
		else:
			break

	# score = lambda x: runs[x]['train']['sim'] * math.log(runs[x]['train']['size'])

	# max_score = max([score(x) for x in runs])
	# best_run = [x for x in runs if score(x) == max_score][0]
	return best_run

def find_bestscore_run(results, treebank, uas_or_las = 'uas'):
	best_score = 0
	best_run = ""
	for r in results[treebank]['runs']:
		best_run_score = max([(x[1] if uas_or_las == 'uas' else x[2]) for x in results[treebank]['runs'][r]['performance']])
		if best_run_score > best_score:
			best_score = best_run_score
			best_run = r

	return best_run, best_score

def weighted_sim_train_sets(train_sets, tbtolang, lang_sims):
	if len(train_sets) == 1:
		return 1

	sims = []
	weights = [] 
	for ts1 in train_sets:
		for ts2 in train_sets:
			if ts1 != ts2:
				lang1 = tbtolang[ts1[0]]
				lang2 = tbtolang[ts2[0]]
				if lang1 == lang2:
					sims.append(1)
				else:
					sims.append(lang_sims[lang1][lang2])
				
				weights.append(min(ts1[1], ts2[1]))
	
	weights = np.array(weights)
	weights = weights / np.sum(weights)
	return np.dot(weights, sims)


def select_early_stop(results, treebank, run, tbtolang, lang_sims, uas_or_las = 'uas'):
	train_sets = results[treebank]['runs'][run]['train']['train_sets']

	sim_train_trg = train_sample_target_weighted_sim(train_sets, treebank, tbtolang, lang_sims)
	#print("Sim: " + str(sim_train_trg))

	sim_trains = weighted_sim_train_sets(train_sets, tbtolang, lang_sims)

	#print(sim_train_trg)
	#print(sim_trans)

	x = 0.2978
	y = 0.9803

	portion = (sim_train_trg - x) / (y - x) # * (1 - sim_trains + y)
	if portion < 0: 
		portion = 0
	#print(portion)

	checkpoints = results[treebank]['runs'][run]['performance']
	best_run_score = max([(x[1] if uas_or_las == 'uas' else x[2]) for x in checkpoints])
	best_ckpnt = [x for x in checkpoints if (x[1] if uas_or_las == 'uas' else x[2]) == best_run_score][0]
	best_ckpnt_ind = checkpoints.index(best_ckpnt)
	#print("Best: " + str(best_ckpnt_ind))
	#print(best_ckpnt)

	ind = int(portion * len(checkpoints))
	if ind >= len(checkpoints):
		ind = len(checkpoints) - 1

	sel = ind
	if ind > 0 and checkpoints[ind-1][2] > checkpoints[sel][2]:
		sel = ind-1

	if ind < (len(checkpoints) - 1) and checkpoints[ind+1][2] > checkpoints[sel][2]:
		sel = ind+1

	selected_checkpoint = checkpoints[sel]
	#print("Selected: " + str(sel))
	#print(selected_checkpoint)

	# dummy
	return selected_checkpoint[1], selected_checkpoint[2] 



def select_treebank_performance(treebank, results, tbtolang, lang_sims, uas_or_las = 'uas', base_path = "/work-ceph/gglavas/data/multiparse/models_third", dummy_run = False, dummy_early = False):
	if results[treebank]['has_own_dev']:
		best_run = results[treebank]['best_setup']
		best_run_str = os.path.join(base_path, treebank, best_run)
		run_results = results[treebank]['runs'][best_run_str]['performance']
		
		ind = 3 if uas_or_las == 'uas' else 4
		max_score = max([x[ind] for x in run_results])
		max_entry = [x for x in run_results if x[ind] == max_score][0]
		return max_entry[1], max_entry[2]

		# just check for the best dev performance and select test performance
	else:
		# selection strategies
		pred_best_run = select_run(results, treebank, tbtolang, lang_sims, dummy = dummy_run)
		true_best_run, true_best_score = find_bestscore_run(results, treebank)


		# if pred_best_run == true_best_run:
		# 	print(treebank + ": CORRECT run selected")
		# 	print(pred_best_run.split("/")[-1], true_best_run.split("/")[-1])
		# else:
		# 	print(treebank + ": WRONG RUN selected")
		# 	print(pred_best_run.split("/")[-1], true_best_run.split("/")[-1])

		if dummy_early: 
			run_results = results[treebank]['runs'][pred_best_run]['performance']
		
			ind = 3 if uas_or_las == 'uas' else 4
			max_score = max([x[ind] for x in run_results])
			max_entry = [x for x in run_results if x[ind] == max_score][0]
			return max_entry[1], max_entry[2]

		else:
			res = select_early_stop(results, treebank, pred_best_run, tbtolang, lang_sims)

		# dummy
		return res #int(pred_best_run.split("/")[-1]), int(true_best_run.split("/")[-1])


def get_treebank_performances(treebanks, all_results, tbtolang, lang_sims, uas_or_las = 'uas', dummy_run = False, dummy_early = False):
	#correct = 0
	#diffs = 0
	final_results = {}
	for t in treebanks:
		final_results[t] = select_treebank_performance(t, all_results, tbtolang, lang_sims, uas_or_las, dummy_run = dummy_run, dummy_early = dummy_early) 
		# pred, real = final_results[t]
		# if pred == real:
		# 	correct += 1
		# else: 
		# 	diffs += abs(pred-real)

	#print("Total correct: " + str(correct))
	#print("Sum diffs: " + str(diffs))

	return final_results

def compare(treebanks, results1, results2):
	uas_1 = []; uas_2 = []; las_1 = []; las_2 = []
	comparison_treebanks = []
	for t in treebanks:
		if t not in results1: 
			print("Treebank " + t + " not in results #1")
			continue
		if t not in results2:
			print("Treebank " + t + " not in results #2")
			continue

		uas_1.append(results1[t][0])
		las_1.append(results1[t][1])
		uas_2.append(results2[t][0])
		las_2.append(results2[t][1])
		comparison_treebanks.append(t)

	
	for i in range(len(comparison_treebanks)):
		print(comparison_treebanks[i])
		print(100*uas_1[i], 100*las_1[i])
		print(uas_2[i], las_2[i])

	print("Avg UAS #1: " + str(sum(uas_1) / len(uas_1)))
	print("Avg LAS #1: " + str(sum(las_1) / len(las_1)))
	print()
	print("Avg UAS #2: " + str(sum(uas_2) / len(uas_2)))
	print("Avg LAS #2: " + str(sum(las_2) / len(las_2)))

	first_better_uas = [i for i in range(len(uas_1)) if 100*uas_1[i] > uas_2[i]]
	first_better_las = [i for i in range(len(las_1)) if 100*las_1[i] > las_2[i]]

	print("First better, UAS: " + str(len(first_better_uas)))
	for i in first_better_uas:
		print(comparison_treebanks[i], uas_1[i], uas_2[i])

	print("First better, LAS: " + str(len(first_better_las)))
	for i in first_better_las:
		print(comparison_treebanks[i], las_1[i], las_2[i])

def get_udify_udpipe(path):
	udify_dict = {}
	udpipe_dict = {}

	content = load_file(path)
	blocks = content.strip().split("\n\n")

	for b in blocks:
		lines = b.split("\n")
		tbsplit = lines[0].strip().split()
		if len(tbsplit) != 2:
			raise ValueError("Unexpected!")
		treebank = "UD_" + tbsplit[0] + "-" + tbsplit[1]
		
		uas, las = float(lines[1].split()[4]), float(lines[1].split()[5])
		if len(lines) == 2:
			udify_dict[treebank] = (uas, las)
		elif len(lines) == 3:
			udpipe_dict[treebank] = (uas, las)
			udify_dict[treebank] = float(lines[2].split()[4]), float(lines[2].split()[5])

	return udify_dict, udpipe_dict