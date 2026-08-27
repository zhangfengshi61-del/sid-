"""DEEP SID-MLP analysis: WHY is it 'lossless' on aggregate Recall@10 yet diverges from the
teacher for 96-98% of users? Records per-user, for BOTH teacher(TIGER) and SID-MLP student:
  - gt rank in the top-N item list (dedup) -> R@{1,5,10,20,50}, NDCG@10, MRR
  - prefix-coverage: deepest SID prefix (d1/d1d2/d1d2d3/full) at which gt survives the beam
  - target-item POPULARITY (train-label frequency) -> head/mid/tail/cold slices
  - WIN/LOSS contingency teacher-hit@10 x student-hit@10 (onlyS = student wins where TEACHER MISSES
    = CE fitting the benchmark target, NOT accelerating the teacher).
Outputs JSON. Hypotheses: (H3 CE-overfit) student>=teacher HEAD, <teacher MID/TAIL; aggregate
parity = (onlyS wins on head) - (onlyT losses on tail).
"""
import sys, os, json, math, argparse
from collections import defaultdict
import torch, torch.nn.functional as F
sys.path.insert(0, './repos/SID-MLP')
os.chdir('./repos')
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
from torch.utils.data import DataLoader
from sid_mlp.inference import (build_tiger_pipeline, build_sidmlp_from_checkpoint,
                               configure_precision, _torch_load, find_valid_set)
from sid_mlp.beam import build_valid_tensors, gather_d4_mask, lookup_item_idx
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int, quad_key_int

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--lens_path', required=True)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--valid_set', default=None)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--beam_size', type=int, default=50)
ap.add_argument('--final_top_n', type=int, default=50)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', default='')
ap.add_argument('--max_batches', type=int, default=0)
args = ap.parse_args()
args.encoder_ckpt = None; args.fp32 = False
dev = torch.device('cuda')
nb16 = configure_precision(args, dev)

pipe = build_tiger_pipeline(args)
tiger = pipe.model.to(dev).eval()
t5 = tiger.t5
if nb16: t5.bfloat16()
for p in t5.parameters(): p.requires_grad_(False)
tk = _torch_load(args.checkpoint, weights_only=False)
lens, _ = build_sidmlp_from_checkpoint(args.lens_path, tk['t5.shared.weight'], dev)
if nb16: lens.bfloat16()
del tk
vs = find_valid_set(args.dataset_tag, explicit_path=args.valid_set, lens_path=args.lens_path, graph_dir=args.graph_dir)
(mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, keys4_sorted, vals4_sorted, _im, valid_items_4) = build_valid_tensors(vs, dev)
beam = args.beam_size; final_n = args.final_top_n

def sid_to_item(tup):
    return valid_items_4.get(tuple(int(x) for x in tup), -1)

print('counting popularity from train labels...', flush=True)
pop = defaultdict(int)
train_ds = pipe.tokenized_datasets['train']
train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, collate_fn=pipe.tokenizer.collate_fn['train'])
with torch.inference_mode():
    for b in train_loader:
        for row in b['labels'].tolist():
            it = sid_to_item(row[:4])
            if it != -1: pop[it] += 1
print(f'popularity: {len(pop)} items, max={max(pop.values()) if pop else 1}', flush=True)
counts_sorted = sorted(pop.values())
def pop_bucket(it):
    c = pop.get(it, 0)
    if c == 0: return 'cold'
    n = len(counts_sorted); lo = counts_sorted[n//3]; hi = counts_sorted[2*n//3]
    if c <= lo: return 'tail'
    if c <= hi: return 'mid'
    return 'head'

def student_beam(input_ids, attn):
    B = input_ids.shape[0]
    enc = t5.encoder(input_ids=input_ids, attention_mask=attn, return_dict=True).last_hidden_state
    pad = ~attn.bool()
    ctx = lens.encode(enc, pad)
    d1_lp = F.log_softmax(lens.head1(ctx).float(), dim=-1)
    d1_scores, d1_idx = torch.topk(d1_lp, beam, dim=1)
    d1_tokens = d1_idx + DIGIT_OFFSETS[0]
    ctx_view = ctx.unsqueeze(1).expand(-1, beam, -1).contiguous()
    d1_flat = d1_tokens.reshape(B*beam); ctx_flat = ctx_view.reshape(B*beam,-1)
    logits2 = lens.head2(torch.cat([ctx_flat, lens.embedding(d1_flat)],1)).float()
    d1_cb = (d1_flat-DIGIT_OFFSETS[0]).clamp(0,CODEBOOK_SIZE-1)
    logits2 = logits2.masked_fill(~mask_d1_d2[d1_cb], float('-inf'))
    d2_lp = F.log_softmax(logits2,1).view(B,beam,CODEBOOK_SIZE)
    score2 = d1_scores.unsqueeze(-1)+d2_lp
    best2_score,best2_idx = torch.topk(score2.masked_fill(torch.isnan(score2),float('-inf')).view(B,-1),beam,1)
    sel_beam2=best2_idx//CODEBOOK_SIZE; sel_d2=best2_idx%CODEBOOK_SIZE
    d1_tokens=torch.gather(d1_tokens,1,sel_beam2); d2_tokens=sel_d2+DIGIT_OFFSETS[1]; prefix_score=best2_score
    d1_flat=d1_tokens.reshape(-1); d2_flat=d2_tokens.reshape(-1)
    logits3=lens.head3(torch.cat([ctx_flat,lens.embedding(d1_flat),lens.embedding(d2_flat)],1)).float()
    d1_cb=(d1_flat-DIGIT_OFFSETS[0]).clamp(0,CODEBOOK_SIZE-1); d2_cb=(d2_flat-DIGIT_OFFSETS[1]).clamp(0,CODEBOOK_SIZE-1)
    logits3=logits3.masked_fill(~mask_d1d2_d3[d1_cb,d2_cb],float('-inf'))
    d3_lp=F.log_softmax(logits3,1).view(B,beam,CODEBOOK_SIZE)
    score3=prefix_score.unsqueeze(-1)+d3_lp
    best3_score,best3_idx=torch.topk(score3.masked_fill(torch.isnan(score3),float('-inf')).view(B,-1),beam,1)
    sel_beam3=best3_idx//CODEBOOK_SIZE; sel_d3=best3_idx%CODEBOOK_SIZE
    d1_tokens=torch.gather(d1_tokens,1,sel_beam3); d2_tokens=torch.gather(d2_tokens,1,sel_beam3); d3_tokens=sel_d3+DIGIT_OFFSETS[2]
    d1_flat=d1_tokens.reshape(-1); d2_flat=d2_tokens.reshape(-1); d3_flat=d3_tokens.reshape(-1)
    logits4=lens.head4(torch.cat([ctx_flat,lens.embedding(d1_flat),lens.embedding(d2_flat),lens.embedding(d3_flat)],1)).float()
    key3=triplet_key_int(d1_flat-DIGIT_OFFSETS[0],d2_flat-DIGIT_OFFSETS[1],d3_flat-DIGIT_OFFSETS[2])
    logits4=logits4.masked_fill(~gather_d4_mask(d4_keys_sorted,d4_masks_sorted,key3),float('-inf'))
    d4_lp=F.log_softmax(logits4,1)
    top_d4_val,top_d4_idx=torch.topk(d4_lp,CODEBOOK_SIZE,1); d4_tokens=top_d4_idx+DIGIT_OFFSETS[3]
    quad=best3_score.unsqueeze(-1)+top_d4_val.view(B,beam,CODEBOOK_SIZE)
    topq_idx=torch.topk(quad.masked_fill(torch.isnan(quad),float('-inf')).view(B,-1),final_n,1).indices
    beam_idx=topq_idx//CODEBOOK_SIZE
    d1f=torch.gather(d1_tokens,1,beam_idx); d2f=torch.gather(d2_tokens,1,beam_idx)
    d3f=torch.gather(d3_tokens,1,beam_idx); d4f=torch.gather(d4_tokens.view(B,-1),1,topq_idx)
    return torch.stack([d1f,d2f,d3f,d4f],-1), d1_tokens, d2_tokens, d3_tokens

def dedup_items(sid_list):
    seen=set(); out=[]
    for tup in sid_list:
        it=sid_to_item(tup)
        if it==-1 or it in seen: continue
        seen.add(it); out.append(it)
    return out
def rank_of(items, gt):
    try: return items.index(gt)
    except ValueError: return -1

CUTS=[1,5,10,20,50]; BKTS=['all','head','mid','tail','cold']
def blank(): return {'n':0,'recall':{c:0 for c in CUTS},'ndcg10':0.0,'mrr':0.0,'cov':{'d1':0,'d1d2':0,'d1d2d3':0,'item':0}}
def upd(acc, rank, cov):
    acc['n']+=1
    for c in CUTS:
        if rank!=-1 and rank<c: acc['recall'][c]+=1
    if rank!=-1:
        acc['ndcg10']+= (1.0/math.log2(rank+2)) if rank<10 else 0.0
        acc['mrr']+= 1.0/(rank+1)
    for k in cov: acc['cov'][k]+=cov[k]
teach={b:blank() for b in BKTS}; stud={b:blank() for b in BKTS}
exact={b:{'n':0,'agree1':0,'setid':0,'jac':0.0,'jac20':0.0} for b in BKTS}
contin={b:{'both':0,'onlyT':0,'onlyS':0,'neither':0} for b in BKTS}

ds = pipe.tokenized_datasets['test']
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pipe.tokenizer.collate_fn['test'])
print('running test eval...', flush=True)
with torch.inference_mode():
    for bi,batch in enumerate(loader):
        if args.max_batches and bi>=args.max_batches: break
        d={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
        input_ids=d['input_ids']; attn=d['attention_mask']; B=input_ids.shape[0]
        labs=batch['labels'].tolist()
        preds_t=tiger.generate(d, n_return_sequences=final_n)
        s_sids,s_d1,s_d2,s_d3 = student_beam(input_ids, attn)
        pt=preds_t.tolist(); ps=s_sids.tolist(); sd1=s_d1.tolist(); sd2=s_d2.tolist(); sd3=s_d3.tolist()
        for r in range(B):
            gt=tuple(int(x) for x in labs[r][:4]); gt_it=sid_to_item(gt)
            if gt_it==-1: continue
            bkt=pop_bucket(gt_it)
            t_items=dedup_items(pt[r]); t_rank=rank_of(t_items, gt_it); t_pref=[tuple(x) for x in pt[r]]
            tcov={'d1':int(any(p[0]==gt[0] for p in t_pref)),'d1d2':int(any(p[0]==gt[0] and p[1]==gt[1] for p in t_pref)),
                  'd1d2d3':int(any(p[0]==gt[0] and p[1]==gt[1] and p[2]==gt[2] for p in t_pref)),'item':int(t_rank!=-1)}
            s_items=dedup_items(ps[r]); s_rank=rank_of(s_items, gt_it)
            scov={'d1':int(gt[0] in sd1[r]),'d1d2':int(any(sd1[r][i]==gt[0] and sd2[r][i]==gt[1] for i in range(beam))),
                  'd1d2d3':int(any(sd1[r][i]==gt[0] and sd2[r][i]==gt[1] and sd3[r][i]==gt[2] for i in range(beam))),'item':int(s_rank!=-1)}
            th=int(t_rank!=-1 and t_rank<10); sh=int(s_rank!=-1 and s_rank<10)
            ckey='both' if (th and sh) else ('onlyT' if th else ('onlyS' if sh else 'neither'))
            tset=set(t_items[:10]); sset=set(s_items[:10]); t20=set(t_items[:20]); s20=set(s_items[:20])
            for tgt in ('all',bkt):
                upd(teach[tgt], t_rank, tcov); upd(stud[tgt], s_rank, scov)
                exact[tgt]['n']+=1
                exact[tgt]['agree1']+= int(len(t_items)>0 and len(s_items)>0 and t_items[0]==s_items[0])
                exact[tgt]['setid']+= int(tset==sset and len(tset)>0)
                exact[tgt]['jac']+= len(tset&sset)/max(len(tset|sset),1)
                exact[tgt]['jac20']+= len(t20&s20)/max(len(t20|s20),1)
                contin[tgt][ckey]+=1
        if bi%20==0: print(f'batch {bi} done', flush=True)

def fin(acc):
    n=max(acc['n'],1)
    return {'n':acc['n'],'recall':{c:acc['recall'][c]/n for c in CUTS},'ndcg10':acc['ndcg10']/n,'mrr':acc['mrr']/n,
            'cov':{k:acc['cov'][k]/n for k in acc['cov']}}
out={'tag':args.tag,'category':args.category,'teacher':{},'student':{},'exact':{},'contin':{}}
for b in BKTS:
    out['teacher'][b]=fin(teach[b]); out['student'][b]=fin(stud[b])
    n=max(exact[b]['n'],1)
    out['exact'][b]={'n':exact[b]['n'],'agree1':exact[b]['agree1']/n,'setid':exact[b]['setid']/n,'jac':exact[b]['jac']/n,'jac20':exact[b]['jac20']/n}
    out['contin'][b]=dict(contin[b])
with open(args.out,'w') as f: json.dump(out,f,indent=2)
print('=== SUMMARY', args.tag, args.category, '===')
for b in BKTS:
    t=out['teacher'][b]; s=out['student'][b]; e=out['exact'][b]; c=out['contin'][b]
    print(f"[{b:4s} n={t['n']:6d}] R@1 T={t['recall'][1]:.4f} S={s['recall'][1]:.4f} | R@10 T={t['recall'][10]:.4f} S={s['recall'][10]:.4f} | NDCG10 T={t['ndcg10']:.4f} S={s['ndcg10']:.4f} | MRR T={t['mrr']:.4f} S={s['mrr']:.4f} | setid={e['setid']:.3f} jac={e['jac']:.3f} | winS={c['onlyS']} lossS={c['onlyT']} both={c['both']}")
print('OUT', args.out, flush=True); print('DEEPDONE', flush=True)
