"""EXACTNESS GAP: TIGER teacher beam vs SID-MLP student beam, per-user top-K, ALIGNED (same genrec
test loader, same batch). Teacher = tiger.generate; student = the d1-d4 MLP-head beam (from
inference.evaluate_loader). Reports: top-1 SID agreement, top-K set-identity %, mean Jaccard.
This is the fine-grained 'is SID-MLP lossy' — aggregate Recall matches teacher, but does the
actual ranked list? (our 'exact' vs their 'approximate' differentiator)."""
import sys, os, argparse
import torch, torch.nn.functional as F
sys.path.insert(0, './repos/SID-MLP')
os.chdir('./repos')
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
from torch.utils.data import DataLoader
from sid_mlp.inference import (build_tiger_pipeline, build_sidmlp_from_checkpoint,
                               configure_precision, _torch_load, find_valid_set)
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)        # TIGER teacher .pth
ap.add_argument('--lens_path', required=True)          # SID-MLP student
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--valid_set', default=None)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--beam_size', type=int, default=50)
ap.add_argument('--final_top_n', type=int, default=50)
ap.add_argument('--K', type=int, default=10)
ap.add_argument('--max_batches', type=int, default=0)
ap.add_argument('--debug', action='store_true')
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
ds = pipe.tokenized_datasets['test']
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pipe.tokenizer.collate_fn['test'])
beam = args.beam_size; final_n = args.final_top_n; K = args.K

def student_beam(input_ids, attn):
    B = input_ids.shape[0]
    enc = t5.encoder(input_ids=input_ids, attention_mask=attn, return_dict=True).last_hidden_state
    pad = ~attn.bool()
    ctx = lens.encode(enc, pad)
    d1_lp = F.log_softmax(lens.head1(ctx).float(), dim=-1)
    d1_scores, d1_idx = torch.topk(d1_lp, beam, dim=1)
    d1_tokens = d1_idx + DIGIT_OFFSETS[0]
    ctx_view = ctx.unsqueeze(1).expand(-1, beam, -1).contiguous()
    d1_flat = d1_tokens.reshape(B * beam); ctx_flat = ctx_view.reshape(B * beam, -1)
    logits2 = lens.head2(torch.cat([ctx_flat, lens.embedding(d1_flat)], dim=1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
    logits2 = logits2.masked_fill(~mask_d1_d2[d1_cb], float('-inf'))
    d2_lp = F.log_softmax(logits2, dim=-1).view(B, beam, CODEBOOK_SIZE)
    score2 = d1_scores.unsqueeze(-1) + d2_lp
    best2_score, best2_idx = torch.topk(score2.masked_fill(torch.isnan(score2), float('-inf')).view(B, -1), beam, dim=1)
    sel_beam2 = best2_idx // CODEBOOK_SIZE; sel_d2 = best2_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam2); d2_tokens = sel_d2 + DIGIT_OFFSETS[1]
    prefix_score = best2_score
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1)
    logits3 = lens.head3(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat)], dim=1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1); d2_cb = (d2_flat - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE - 1)
    logits3 = logits3.masked_fill(~mask_d1d2_d3[d1_cb, d2_cb], float('-inf'))
    d3_lp = F.log_softmax(logits3, dim=-1).view(B, beam, CODEBOOK_SIZE)
    score3 = prefix_score.unsqueeze(-1) + d3_lp
    best3_score, best3_idx = torch.topk(score3.masked_fill(torch.isnan(score3), float('-inf')).view(B, -1), beam, dim=1)
    sel_beam3 = best3_idx // CODEBOOK_SIZE; sel_d3 = best3_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam3); d2_tokens = torch.gather(d2_tokens, 1, sel_beam3); d3_tokens = sel_d3 + DIGIT_OFFSETS[2]
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1); d3_flat = d3_tokens.reshape(-1)
    logits4 = lens.head4(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat), lens.embedding(d3_flat)], dim=1)).float()
    key3 = triplet_key_int(d1_flat - DIGIT_OFFSETS[0], d2_flat - DIGIT_OFFSETS[1], d3_flat - DIGIT_OFFSETS[2])
    logits4 = logits4.masked_fill(~gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    d4_lp = F.log_softmax(logits4, dim=-1)
    top_d4_val, top_d4_idx = torch.topk(d4_lp, CODEBOOK_SIZE, dim=1)
    d4_tokens = top_d4_idx + DIGIT_OFFSETS[3]
    quad = best3_score.unsqueeze(-1) + top_d4_val.view(B, beam, CODEBOOK_SIZE)
    topq_idx = torch.topk(quad.masked_fill(torch.isnan(quad), float('-inf')).view(B, -1), final_n, dim=1).indices
    beam_idx = topq_idx // CODEBOOK_SIZE
    d1f = torch.gather(d1_tokens, 1, beam_idx); d2f = torch.gather(d2_tokens, 1, beam_idx)
    d3f = torch.gather(d3_tokens, 1, beam_idx); d4f = torch.gather(d4_tokens.view(B, -1), 1, topq_idx)
    return torch.stack([d1f, d2f, d3f, d4f], dim=-1)  # (B, final_n, 4) sorted by score desc

agree1 = setid = total = 0; jac = 0.0
with torch.inference_mode():
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches: break
        dbatch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        input_ids = dbatch['input_ids']; attn = dbatch['attention_mask']; B = input_ids.shape[0]
        preds_t = tiger.generate(dbatch, n_return_sequences=K)              # (B,K,4) teacher
        stu = student_beam(input_ids, attn)[:, :K, :]                        # (B,K,4) student
        if args.debug and bi == 0:
            print('TEACHER preds shape', tuple(preds_t.shape), 'sample', preds_t[0, 0].tolist())
            print('STUDENT preds shape', tuple(stu.shape), 'sample', stu[0, 0].tolist(), flush=True)
        pt = preds_t.tolist(); ps = stu.tolist()
        for r in range(B):
            tset = set(tuple(x) for x in pt[r]); sset = set(tuple(x) for x in ps[r])
            total += 1
            if pt[r][0] == ps[r][0]: agree1 += 1
            if tset == sset: setid += 1
            jac += len(tset & sset) / max(len(tset | sset), 1)
print(f"\n=== EXACTNESS GAP {args.category}: teacher(TIGER) vs SID-MLP student, top-{K}, n={total} ===")
print(f"top-1 SID agreement = {agree1/total*100:.2f}%  |  top-{K} set-identical = {setid/total*100:.2f}%  |  mean Jaccard = {jac/total:.4f}")
print(f"=> {100-setid/total*100:.2f}% of users get a DIFFERENT top-{K} list than the teacher (aggregate Recall still matches).", flush=True)
