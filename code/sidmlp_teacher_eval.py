"""Teacher (genrec TIGER) eval-only: accuracy + throughput, to get the speedup ratio vs SID-MLP
on the SAME hardware. Skips pipeline.run()'s training; calls trainer.evaluate directly."""
import sys, os, time, argparse, torch
sys.path.insert(0, './repos/SID-MLP')
os.chdir('./repos')   # cache/ symlink lives here
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
from genrec.pipeline import Pipeline
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--bs', type=int, default=32)
a = ap.parse_args()

pipe = Pipeline(model_name='TIGER', dataset_name='AmazonReviews2023', checkpoint_path=a.ckpt,
                config_dict={'category': a.category, 'custom_sem_ids_path': a.sem_ids,
                             'cache_dir': './repos/cache',
                             'run_id': 'teacher_eval', 'eval_batch_size': a.bs})
tok = pipe.tokenizer
test_ds = pipe.tokenized_datasets['test']
test_dl = DataLoader(test_ds, batch_size=a.bs, shuffle=False, collate_fn=tok.collate_fn['test'])
model, test_dl = pipe.accelerator.prepare(pipe.model, test_dl)
n = len(test_ds)
torch.cuda.synchronize(); t0 = time.time()
res = pipe.trainer.evaluate(test_dl)
torch.cuda.synchronize(); dt = time.time() - t0
print(f'TEACHER {a.category}: {res} | time={dt:.1f}s throughput={n/dt:.1f}/s n={n} bs={a.bs}', flush=True)
