"""RESCORE-DEPTH ablation on the genrec TIGER teacher (same model SID-MLP distills).
For each user, take the teacher's full-beam top-N candidate items, then RE-RANK them using only
the first k SID tokens of the teacher's own log-likelihood (k=1..L). Report per-popularity-bucket
Recall@10 / NDCG@10 as a function of k.

Tests the ADDRESSING-vs-CAPACITY asymmetry directly:
  - k=L reproduces the teacher's exact ranking (sanity).
  - if HEAD R@10 saturates at small k but TAIL keeps rising to k=L, then the later (capacity)
    tokens do live disambiguation FOR THE TAIL ONLY -> (a) explains why SID-MLP (which weakens all
    tokens) loses the tail, (b) motivates adaptive suffix training (cheapen suffix where prefix is
    already unique / head; keep it for crowded-prefix / tail), (c) gives a tunable exact<->approx knob.
"""
import sys, os, json, math, argparse
from collections import defaultdict
import torch, torch.nn.functional as F
sys.path.insert(0, './repos/SID-MLP')
os.chdir('./repos')
for k in ['HF_DATASETS_OFFLINE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']: os.environ[k]='1'
from torch.utils.data import DataLoader
from genrec.pipeline import Pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=24)
ap.add_argument('--topn', type=int, default=50)
ap.add_argument('--max_batches', type=int, default=0)
a = ap.parse_args()
dev = torch.device('cuda')

pipe = Pipeline(model_name='TIGER', dataset_name='AmazonReviews2023', checkpoint_path=a.checkpoint,
                config_dict={'category': a.category, 'custom_sem_ids_path': a.sem_ids,
                             'cache_dir':'./repos/cache',
                             'run_id':'rescore_depth'})
tok = pipe.tokenizer
tiger = pipe.model.to(dev).eval()
t5 = tiger.t5.bfloat16()
for p in t5.parameters(): p.requires_grad_(False)
n_digit = tok.n_digit
START = 0  # decoder_start_token_id

# item2tokens: map item_id -> tuple of n_digit token ids (with offsets). Build reverse for ranking.
item2tok = tok.item2tokens   # dict item-> tuple(len>=n_digit) ; may include a final dedup/eos token
# normalize to first n_digit tokens
def code_of(it): 
    c = item2tok[it]; return tuple(int(x) for x in c[:n_digit])
# popularity from train labels
print('popularity...', flush=True)
pop = defaultdict(int)
tr = DataLoader(pipe.tokenized_datasets['train'], batch_size=256, shuffle=False, collate_fn=tok.collate_fn['train'])
# labels are token-sequences; map to item via tok? we count by the gt item id directly from dataset if available.
# genrec test/val provide 'labels' = target item's SID tokens. We need item id. Build sid->item map.
sid2item = {}
for it in item2tok:
    sid2item[code_of(it)] = it
def labrow_to_item(row):
    return sid2item.get(tuple(int(x) for x in row[:n_digit]), -1)
with torch.inference_mode():
    for b in tr:
        for row in b['labels'].tolist():
            it=labrow_to_item(row)
            if it!=-1: pop[it]+=1
cs=sorted(pop.values()); 
print(f'pop {len(pop)} items max={max(pop.values()) if pop else 1}', flush=True)
def bucket(it):
    c=pop.get(it,0)
    if c==0: return 'cold'
    n=len(cs); lo=cs[n//3]; hi=cs[2*n//3]
    return 'tail' if c<=lo else ('mid' if c<=hi else 'head')

KS=list(range(1,n_digit+1)); BKTS=['all','head','mid','tail','cold']
rec={b:{k:0 for k in KS} for b in BKTS}; ndcg={b:{k:0.0 for k in KS} for b in BKTS}; cnt={b:0 for b in BKTS}

ds=pipe.tokenized_datasets['test']
loader=DataLoader(ds,batch_size=a.bs,shuffle=False,collate_fn=tok.collate_fn['test'])
print('eval...', flush=True)
with torch.inference_mode():
    for bi,batch in enumerate(loader):
        if a.max_batches and bi>=a.max_batches: break
        d={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
        input_ids=d['input_ids']; attn=d['attention_mask']; B=input_ids.shape[0]
        # teacher top-N candidates (SID tuples)
        cand=tiger.generate(d, n_return_sequences=a.topn)  # (B,topn,n_digit)
        # encoder once
        enc=t5.encoder(input_ids=input_ids, attention_mask=attn, return_dict=True)
        ench=enc.last_hidden_state  # (B,S,d)
        S=ench.shape[1]
        # build decoder inputs for all candidates: (B*topn, n_digit) -> prepend START
        cflat=cand.reshape(B*a.topn, n_digit)
        din=torch.cat([torch.full((B*a.topn,1),START,device=dev,dtype=cflat.dtype), cflat[:,:-1]],dim=1)  # (B*topn,n_digit)
        # expand encoder outputs per candidate
        ench_e=ench.unsqueeze(1).expand(-1,a.topn,-1,-1).reshape(B*a.topn,S,-1)
        attn_e=attn.unsqueeze(1).expand(-1,a.topn,-1).reshape(B*a.topn,S)
        from transformers.modeling_outputs import BaseModelOutput
        out=t5(encoder_outputs=BaseModelOutput(last_hidden_state=ench_e), attention_mask=attn_e, decoder_input_ids=din)
        logp=F.log_softmax(out.logits.float(),dim=-1)  # (B*topn,n_digit,vocab)
        tok_logp=torch.gather(logp,2,cflat.unsqueeze(-1)).squeeze(-1)  # (B*topn,n_digit) logP of each true token
        tok_logp=tok_logp.view(B,a.topn,n_digit)
        cumlp=torch.cumsum(tok_logp,dim=2)  # S_k at index k-1
        candt=cand.tolist()
        labs=batch['labels'].tolist()
        for r in range(B):
            gt=labrow_to_item(labs[r])
            if gt==-1: continue
            bkt=bucket(gt); 
            for tgt in ('all',bkt): cnt[tgt]+=1
            items=[sid2item.get(tuple(candt[r][j]),-1) for j in range(a.topn)]
            for k in KS:
                sc=cumlp[r,:,k-1]  # (topn,)
                order=torch.argsort(sc,descending=True).tolist()
                seen=set(); ranked=[]
                for j in order:
                    it=items[j]
                    if it==-1 or it in seen: continue
                    seen.add(it); ranked.append(it)
                try: rk=ranked.index(gt)
                except ValueError: rk=-1
                if rk!=-1 and rk<10:
                    for tgt in ('all',bkt):
                        rec[tgt][k]+=1; ndcg[tgt][k]+=1.0/math.log2(rk+2)
        if bi%20==0: print(f'batch {bi}', flush=True)

out={'category':a.category,'topn':a.topn,'n_digit':n_digit,'rec':{},'ndcg':{},'n':{}}
for b in BKTS:
    n=max(cnt[b],1); out['n'][b]=cnt[b]
    out['rec'][b]={k:rec[b][k]/n for k in KS}; out['ndcg'][b]={k:ndcg[b][k]/n for k in KS}
with open(a.out,'w') as f: json.dump(out,f,indent=2)
print('=== RESCORE-DEPTH', a.category, '(R@10 by #tokens used k=1..%d) ==='%n_digit)
for b in BKTS:
    n=cnt[b]; row=' '.join(f'k{k}={rec[b][k]/max(n,1):.4f}' for k in KS)
    print(f'[{b:4s} n={n:6d}] {row}')
print('OUT',a.out,flush=True); print('RDDONE',flush=True)
