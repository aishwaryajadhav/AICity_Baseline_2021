import os
import pickle
import statistics
import argparse
import json 
import numpy as np

parser = argparse.ArgumentParser(description='2-stage model encoding generation')
parser.add_argument('--encoding_dir', default="configs/baseline.yaml", type=str)
parser.add_argument('--dataset', default = 'train', type=str,help='Evaluate of which json')
parser.add_argument('--tracks', type=str,help='Train/val tracks json')
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()


multiclass_sim = []
dissim = []

with open(os.path.join(args.encoding_dir, args.dataset+"_stage1_text_subject_encodings.pkl"), 'rb') as f:
    subj_embed = pickle.load(f)

with open(os.path.join(args.encoding_dir, args.dataset+"_stage1_crops_car_encodings.pkl"), 'rb') as f:
    crop_embed = pickle.load(f)

with open(args.tracks) as f:
    tracks = json.load(f)

os.makedirs(args.save_dir,exist_ok = True)

#Stage 1

crop_ids = list(crop_embed.keys())
sorted_retr = {}
mrr_sum = 0.0

for qid, query in subj_embed.items():
    qscores = []
    for cid in crop_ids:
        cars = crop_embed[cid]
        score = np.matmul(query,cars.T)
        qscores.append(score)
        
        if(cid in tracks[qid]['targets']):
            multiclass_sim.append(score)
        else:
            dissim.append(score)

    ranks = np.argsort(qscores)
    sorted_retr[qid] = np.array(crop_ids)[ranks]

    #MRR calc
    # target_rank = ret_tracks.index(qid) + 1
    # mrr_sum += (1/target_rank)        

with open(os.path.join(args.save_dir, 'stage_1_retrievals.pkl'),'wb') as f:
    pickle.dump(sorted_retr, f)

print(multiclass_sim)

print('Similar target mean: ', np.mean(multiclass_sim))
print('Dissimilar target mean: ', np.mean(dissim))
print('Similar target std: ', statistics.pstdev(np.array(multiclass_sim)))



print('Dissimilar target std: ', statistics.pstdev(np.array(dissim)))

