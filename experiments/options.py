import argparse

parser = argparse.ArgumentParser(description='SG-SPL: Structure & Geometry-regularized Prompt Learning for ZS-SBIR')

# Experiment
parser.add_argument('--exp_name', type=str, default='SG-SPL')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints/')

# Dataset
parser.add_argument('--dataset', type=str, default='sketchy_2',
                    choices=['sketchy_1', 'sketchy_2', 'tuberlin', 'quickdraw'],
                    help='Dataset split to use')
parser.add_argument('--split', type=str, default='zs',
                    choices=['zs', 'gzs'],
                    help='zs=Zero-Shot, gzs=Generalized Zero-Shot')

# Paths — adjust to your data location
parser.add_argument('--sketchy_dir', type=str, default='datasets/Sketchy/',
                    help='Root of Sketchy-Extended: must contain sketch/ and photo/')
parser.add_argument('--tuberlin_dir', type=str, default='datasets/TUBerlin/',
                    help='Root of TU-Berlin-Extended: must contain sketches/ and images/')
parser.add_argument('--quickdraw_dir', type=str, default='datasets/QuickDraw/',
                    help='Root of QuickDraw-Extended: must contain sketches/ and images/')

# DataLoader
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--num_workers',type=int, default=4)
parser.add_argument('--image_size', type=int, default=224)

# CLIP backbone
parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                    help='CLIP model name (e.g. ViT-B/32) OR path to a local .pt file for offline use.')
parser.add_argument('--embed_dim',  type=int, default=512,
                    help='CLIP embedding dimension (512 for ViT-B/32, 768 for ViT-L/14)')

# Prompt Tuning  (CLIP-AT style)
parser.add_argument('--n_prompts', type=int, default=3,
                    help='Number of learnable prompt tokens per modality')
parser.add_argument('--prompt_dim', type=int, default=768,
                    help='Prompt token dimension = CLIP visual transformer WIDTH '
                         '(768 for ViT-B/32 and ViT-B/16, 1024 for ViT-L/14). '
                         'NOTE: this is the internal hidden dim, NOT the output embed_dim=512.')

# Optimizer
parser.add_argument('--lr_prompt', type=float, default=1e-4, help='Learning rate for prompt parameters')
parser.add_argument('--lr_ln', type=float, default=1e-6, help='Learning rate for CLIP LayerNorm parameters')
# parser.add_argument('--weight_decay',type=float,default=1e-4)
parser.add_argument('--max_epochs', type=int, default=20)


# Loss weights
# Triplet loss (always on)
parser.add_argument('--triplet_margin', type=float, default=0.2)

# L_cls — classification loss
parser.add_argument('--l_cls', type=float, default=1.0, help='Weight for classification loss L_cls')

# L_SSC — Semantic Structure Consistency
parser.add_argument('--l_ssc', type=float, default=1.0, help='Weight for L_SSC (set 0 to disable)')
parser.add_argument('--ssc_dist', type=str, default='mse', choices=['mse', 'kl'],
                    help='Distance function for L_SSC: mse (EBSeg original) or kl')
parser.add_argument('--ssc_temp', type=float, default=0.1, help='Temperature T for KL variant of L_SSC')

# L_xmod — Cross-modal Structure Consistency
parser.add_argument('--l_x', type=float, default=0.5, help='Weight for L_xmod inside L_SSC term (set 0 to ablate xmod)')

# L_asym_sph — Asymmetric Hyperspherical Anchoring
parser.add_argument('--l_sph_ph', type=float, default=1.0,
                    help='lambda_ph: anchor weight for photo modality (stronger)')
parser.add_argument('--l_sph_sk', type=float, default=0.2,
                    help='lambda_sk: anchor weight for sketch modality (weaker → let sketch adapt)')


# EMA prototype bank
parser.add_argument('--ema_m', type=float, default=0.9, 
                    help='EMA momentum for prototype bank update')
parser.add_argument('--bank_warmup',type=int, default=10, 
                    help='Minimum number of active prototypes before computing L_SSC/L_xmod')
parser.add_argument('--no_proto_grad', action='store_true', 
                    help='Stop gradient flow through prototype bank (ablation)')


# Text anchor templates
parser.add_argument('--text_templates', type=str, nargs='+',
                    default=['a photo of a {}.', 'a sketch of a {}.','a drawing of a {}.', 'an image of a {}.'],
                    help='Templates for building text anchor matrix A')

# Trainer
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=str, default='16-mixed', choices=['32', '16-mixed', 'bf16-mixed'])
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--val_every', type=int, default=1, help='Run validation every N epochs')
parser.add_argument('--sanity_steps', type=int, default=-1,
                    help='Number of validation steps to run before training (-1 for full epoch, 2 for default check, 0 to disable)')
parser.add_argument('--ckpt_path',  type=str, default=None,
                    help='Path to checkpoint to resume from (Lightning 2.x style)')

opts = parser.parse_args(args=[])  # default options; override via CLI in train.py
