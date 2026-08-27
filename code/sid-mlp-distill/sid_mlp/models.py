"""Model definitions for the public SID-MLP code path.

Only the paper decoder is kept: one cross-attention context readout and four
prefix-conditioned 256-way MLP heads.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CODEBOOK_SIZE, DIGIT_OFFSETS


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dim: int = 0,
    ):
        super().__init__()
        inner = attn_dim if attn_dim > 0 else d_model
        if inner % num_heads != 0:
            raise ValueError("attn_dim/d_model must be divisible by num_heads")

        self.inner = inner
        self.num_heads = num_heads
        self.head_dim = inner // num_heads
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.dropout = dropout

    def precompute_kv(self, encoder_seq: torch.Tensor):
        batch_size, seq_len, _ = encoder_seq.shape
        k = self.k_proj(encoder_seq).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()
        v = self.v_proj(encoder_seq).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()
        return k, v

    def forward(
        self,
        query_vec: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = query_vec.shape[0]
        q = self.q_proj(query_vec).view(batch_size, self.num_heads, 1, self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        any_valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
        safe_scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        probs = F.softmax(safe_scores, dim=-1)
        probs = torch.where(any_valid, probs, torch.zeros_like(probs))
        if self.dropout > 0 and self.training:
            probs = F.dropout(probs, p=self.dropout)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.inner)
        return self.out_proj(out).squeeze(1)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.fc(x))))


def make_mlp(in_dim, out_dim, hidden_dim, num_layers, dropout):
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    ]
    for _ in range(num_layers - 1):
        layers.append(MLPBlock(hidden_dim, dropout=dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SIDMLP(nn.Module):
    """SID-MLP decoder distilled from a frozen TIGER teacher."""

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        embed_dim=128,
        d_model=128,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        head_hidden=512,
        head_layers: int = 1,
        attn_dim: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attn_dim=attn_dim,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)

        self.head1 = make_mlp(d_model, CODEBOOK_SIZE, head_hidden, head_layers, dropout)
        self.head2 = make_mlp(d_model + embed_dim, CODEBOOK_SIZE, head_hidden, head_layers, dropout)
        self.head3 = make_mlp(d_model + 2 * embed_dim, CODEBOOK_SIZE, head_hidden, head_layers, dropout)
        self.head4 = make_mlp(d_model + 3 * embed_dim, CODEBOOK_SIZE, head_hidden, head_layers, dropout)

    def post_attn(self, query: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        x = self.norm1(query + attn_out)
        return self.norm2(x + self.ffn(x))

    def encode(self, enc_seq, enc_mask):
        valid = (~enc_mask).to(enc_seq.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (enc_seq * valid.unsqueeze(-1)).sum(dim=1) / denom
        query = self.query_proj(pooled)
        k, v = self.cross_attn.precompute_kv(enc_seq)
        return self.post_attn(query, self.cross_attn(query, k, v, key_padding_mask=enc_mask))

    def forward(self, enc_seq, enc_mask, d1_ids=None, d2_ids=None, d3_ids=None):
        ctx = self.encode(enc_seq, enc_mask)
        logits1 = self.head1(ctx)

        if d1_ids is None:
            d1_ids = logits1.argmax(dim=-1) + DIGIT_OFFSETS[0]
        with torch.no_grad():
            e1 = self.embedding(d1_ids)
        logits2 = self.head2(torch.cat([ctx, e1], dim=1))

        if d2_ids is None:
            d2_ids = logits2.argmax(dim=-1) + DIGIT_OFFSETS[1]
        with torch.no_grad():
            e2 = self.embedding(d2_ids)
        logits3 = self.head3(torch.cat([ctx, e1, e2], dim=1))

        if d3_ids is None:
            d3_ids = logits3.argmax(dim=-1) + DIGIT_OFFSETS[2]
        with torch.no_grad():
            e3 = self.embedding(d3_ids)
        logits4 = self.head4(torch.cat([ctx, e1, e2, e3], dim=1))

        return logits1, logits2, logits3, logits4


__all__ = ["CrossAttention", "MLPBlock", "SIDMLP", "make_mlp"]


class SIDMLPPD(SIDMLP):
    """s23: prefix-DEPENDENT context variant. The s22 capacity ladder localized the fidelity
    bottleneck to the prefix-INDEPENDENT one-shot context (heads flat, context the only + lever,
    d1@1 immune); this variant recomputes the cross-attention readout per digit with a query
    conditioned on the decoded prefix. Heads/embeddings unchanged. pproj* are ZERO-init so the
    model starts exactly at the base parameterization (ctx_i == ctx at init)."""

    def __init__(self, *args, **kwargs):
        self.pd_depth = int(kwargs.pop("pd_depth", 4))
        self.pd_mode = str(kwargs.pop("pd_mode", "full") or "full").lower()
        if self.pd_mode not in {"full", "cheap", "train_only"}:
            raise ValueError(f"unknown pd_mode={self.pd_mode}")
        super().__init__(*args, **kwargs)
        d_model = self.query_proj.out_features
        embed_dim = self.embedding.embedding_dim
        self.pproj2 = nn.Linear(embed_dim, d_model, bias=False)
        self.pproj3 = nn.Linear(2 * embed_dim, d_model, bias=False)
        self.pproj4 = nn.Linear(3 * embed_dim, d_model, bias=False)
        for m in (self.pproj2, self.pproj3, self.pproj4):
            nn.init.zeros_(m.weight)

    def q0_kv(self, enc_seq, enc_mask):
        valid = (~enc_mask).to(enc_seq.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (enc_seq * valid.unsqueeze(-1)).sum(dim=1) / denom
        q0 = self.query_proj(pooled)
        k, v = self.cross_attn.precompute_kv(enc_seq)
        return q0, k, v

    def ctx_from_q(self, q, k, v, enc_mask):
        return self.post_attn(q, self.cross_attn(q, k, v, key_padding_mask=enc_mask))

    def prefix_ctx(self, q0, ctx_base, k, v, enc_mask, delta):
        # cheap-PD keeps a single history read and lets the prefix apply a small residual.
        # train_only uses the full PD path during training; frontier eval ignores it.
        if getattr(self, "pd_mode", "full") == "cheap":
            return ctx_base + delta
        return self.ctx_from_q(q0 + delta, k, v, enc_mask)

    def forward(self, enc_seq, enc_mask, d1_ids=None, d2_ids=None, d3_ids=None):
        q0, k, v = self.q0_kv(enc_seq, enc_mask)
        ctx1 = self.ctx_from_q(q0, k, v, enc_mask)
        logits1 = self.head1(ctx1)

        if d1_ids is None:
            d1_ids = logits1.argmax(dim=-1) + DIGIT_OFFSETS[0]
        pd_depth = int(getattr(self, "pd_depth", 4))
        e1 = self.embedding(d1_ids)
        ctx2 = self.prefix_ctx(q0, ctx1, k, v, enc_mask, self.pproj2(e1)) if pd_depth >= 2 else ctx1
        logits2 = self.head2(torch.cat([ctx2, e1], dim=1))

        if d2_ids is None:
            d2_ids = logits2.argmax(dim=-1) + DIGIT_OFFSETS[1]
        e2 = self.embedding(d2_ids)
        ctx3 = self.prefix_ctx(q0, ctx1, k, v, enc_mask, self.pproj3(torch.cat([e1, e2], dim=1))) if pd_depth >= 3 else ctx1
        logits3 = self.head3(torch.cat([ctx3, e1, e2], dim=1))

        if d3_ids is None:
            d3_ids = logits3.argmax(dim=-1) + DIGIT_OFFSETS[2]
        e3 = self.embedding(d3_ids)
        ctx4 = self.prefix_ctx(q0, ctx1, k, v, enc_mask, self.pproj4(torch.cat([e1, e2, e3], dim=1))) if pd_depth >= 4 else ctx1
        logits4 = self.head4(torch.cat([ctx4, e1, e2, e3], dim=1))

        return logits1, logits2, logits3, logits4


class SIDMLPJP(SIDMLP):
    """s24 arm C: one JOINT head over the OCCUPIED (d1,d2) prefixes replaces head1+head2.
    Motivation (three-way diagnostic): 86-91% of fidelity misses are cross-d1 displacements;
    the factorized p(d1)*p(d2|d1) routes every cross-branch comparison through the single d1
    softmax, while the joint head calibrates any two occupied prefixes directly. Suffix heads
    unchanged (disambiguation, s23). jp_pairs is a registered buffer (codebook-space (P,2)),
    so checkpoints are self-describing."""

    def __init__(self, embedding_weight, jp_pairs=None, **kwargs):
        super().__init__(embedding_weight, **kwargs)
        if jp_pairs is None:
            raise ValueError("SIDMLPJP requires jp_pairs (P,2) codebook-space occupied prefixes")
        jp_pairs = jp_pairs.long()
        P = jp_pairs.shape[0]
        d_model = self.query_proj.out_features
        self.headJP = make_mlp(d_model, P,
                               kwargs.get("head_hidden", 512),
                               kwargs.get("head_layers", 1),
                               kwargs.get("dropout", 0.1))
        del self.head1
        del self.head2
        inv = torch.full((CODEBOOK_SIZE * CODEBOOK_SIZE,), -1, dtype=torch.long)
        inv[jp_pairs[:, 0] * CODEBOOK_SIZE + jp_pairs[:, 1]] = torch.arange(P)
        self.register_buffer("jp_pairs", jp_pairs)
        self.register_buffer("jp_inv", inv)

    def forward(self, enc_seq, enc_mask, d1_ids=None, d2_ids=None, d3_ids=None):
        if d1_ids is None or d2_ids is None:
            raise ValueError("SIDMLPJP.forward is teacher-forced on (d1,d2); use the JP beam for inference")
        ctx = self.encode(enc_seq, enc_mask)
        logitsJP = self.headJP(ctx)
        with torch.no_grad():
            e1 = self.embedding(d1_ids)
            e2 = self.embedding(d2_ids)
        logits3 = self.head3(torch.cat([ctx, e1, e2], dim=1))
        if d3_ids is None:
            d3_ids = logits3.argmax(dim=-1) + DIGIT_OFFSETS[2]
        with torch.no_grad():
            e3 = self.embedding(d3_ids)
        logits4 = self.head4(torch.cat([ctx, e1, e2, e3], dim=1))
        return logitsJP, logits3, logits4


class SetReranker(nn.Module):
    """s26: candidate-set-conditioned reranker — the strongest natural verify-free
    alternative to exact rescoring. Every scorer in the refutation matrix so far
    (factorized one-shot / PD / JP) scores candidates INDEPENDENTLY; this head sees the
    candidate set JOINTLY (as the exact rescore does, but learned) and emits a residual on
    the base student's deployed factorized scores. Zero-init output => training starts
    exactly at the base ranking (the SIDMLPPD trick): any move is attributable to the
    added conditioning. arch='set' = transformer over the candidate set; arch='point' =
    same features, per-candidate MLP (no cross-candidate attention) — the control that
    isolates set-conditioning itself.
    Features per candidate (GPT-Pro-hardened): fresh position-specific digit embeddings
    (NOT tied to the student's factorized geometry) + the frozen semantic-ID embeddings +
    per-digit conditional logprobs + per-row z-scored base score + base rank; plus the
    base context vector added to every token."""

    def __init__(self, embedding_weight, d_model=128, rr_dim=256, rr_layers=2,
                 rr_heads=4, dropout=0.1, arch="set", pos_embed_dim=64):
        super().__init__()
        self.arch = arch
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        E = embedding_weight.shape[1]
        self.pos_digit_embed = nn.Embedding(4 * CODEBOOK_SIZE, pos_embed_dim)
        nn.init.normal_(self.pos_digit_embed.weight, std=0.02)
        self.cand_proj = nn.Linear(4 * E + 4 * pos_embed_dim, rr_dim)
        self.ctx_proj = nn.Linear(d_model, rr_dim)
        self.feat_proj = nn.Linear(6, rr_dim)   # [z_score, rank/M, z_lp1..z_lp4]
        if arch == "set":
            enc_layer = nn.TransformerEncoderLayer(
                rr_dim, rr_heads, dim_feedforward=4 * rr_dim, dropout=dropout,
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, rr_layers)
        else:
            self.encoder = nn.Sequential(
                nn.LayerNorm(rr_dim), nn.Linear(rr_dim, 4 * rr_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(4 * rr_dim, rr_dim), nn.GELU(),
            )
        self.out = nn.Linear(rr_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _rowz(x, valid_mask):
        vm = valid_mask.to(x.dtype)
        cnt = vm.sum(1, keepdim=True).clamp_min(1.0)
        xm = x.masked_fill(~valid_mask, 0.0)
        mu = (xm * vm).sum(1, keepdim=True) / cnt
        var = (((xm - mu) * vm) ** 2).sum(1, keepdim=True) / cnt
        return ((xm - mu) / (var.sqrt() + 1e-4)) * vm

    def forward(self, ctx, cand_cb, base_scores, valid_mask, comp_lp=None):
        """ctx: (B, D); cand_cb: (B, M, 4) codebook digits; base_scores: (B, M) deployed
        factorized scores; valid_mask: (B, M) bool; comp_lp: (B, M, 4) per-digit
        conditional logprobs (zeros if unavailable). Returns (B, M) = base + residual."""
        B, M, _ = cand_cb.shape
        cc = cand_cb.long().clamp(0, CODEBOOK_SIZE - 1)
        offs = torch.as_tensor(DIGIT_OFFSETS, device=cand_cb.device)
        e_sem = self.embedding(cc + offs).reshape(B, M, -1)
        pos_ids = cc + (torch.arange(4, device=cand_cb.device) * CODEBOOK_SIZE)
        e_pos = self.pos_digit_embed(pos_ids).reshape(B, M, -1)
        z = self._rowz(base_scores, valid_mask)
        rk = torch.argsort(torch.argsort(
            base_scores.masked_fill(~valid_mask, float('-inf')), dim=1, descending=True),
            dim=1).to(base_scores.dtype) / max(M - 1, 1)
        if comp_lp is None:
            comp_lp = base_scores.new_zeros(B, M, 4)
        zl = torch.stack([self._rowz(comp_lp[:, :, i], valid_mask) for i in range(4)], -1)
        feats = torch.cat([z.unsqueeze(-1), rk.unsqueeze(-1), zl], dim=-1)
        tok = (self.cand_proj(torch.cat([e_sem, e_pos], -1))
               + self.ctx_proj(ctx).unsqueeze(1) + self.feat_proj(feats))
        if self.arch == "set":
            h = self.encoder(tok, src_key_padding_mask=~valid_mask)
        else:
            h = self.encoder(tok)
        resid = self.out(h).squeeze(-1)
        return base_scores + resid.masked_fill(~valid_mask, 0.0)
