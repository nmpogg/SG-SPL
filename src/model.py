import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from collections import defaultdict
from torchmetrics.functional import retrieval_average_precision

from src.coprompt import VisualPromptLearner
from src.utils import load_clip_to_cpu, get_all_categories, retrieval_precision, visualize_tsne
from src.splits import VISUALIZE_CLASSES, UNSEEN_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        clip_model.apply(freeze_all_but_bn)
        self.dtype = clip_model.dtype
        
        self.prompt_learner_photo = VisualPromptLearner(cfg, clip_model)
        self.prompt_learner_sketch = VisualPromptLearner(cfg, clip_model)
        
        self.ph_encoder = copy.deepcopy(clip_model.visual)
        self.sk_encoder = copy.deepcopy(clip_model.visual)
        self.logit_scale = clip_model.logit_scale

    def forward(self, img_tensor, type='photo'):
        if type == 'photo':
            prompt_learner = self.prompt_learner_photo
            image_encoder = self.ph_encoder
        else:
            prompt_learner = self.prompt_learner_sketch
            image_encoder = self.sk_encoder
            
        shared_ctx = prompt_learner(img_tensor.shape[0])
        
        # passed compound_deeper_prompts as []
        image_features = image_encoder(
            img_tensor.type(self.dtype), shared_ctx, []
        )
        image_features_normalize = F.normalize(image_features, dim=-1)
        return image_features_normalize

class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        self.args = args
        self.classname = classname
        clip_model = load_clip_to_cpu(args)
        
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.best_metric = 1e-3
        
        # Frozen branch for text anchors and geometry regularization
        self.clip_frozen = copy.deepcopy(clip_model)
        self.clip_frozen.requires_grad_(False)
        self.clip_frozen.eval()
        
        self.model = CustomCLIP(cfg=args, clip_model=clip_model)
        
        # Precompute text anchor matrix
        self.build_text_anchor(classname)
        
        n_seen = len(classname)
        self.register_buffer('proto_sk', torch.zeros(n_seen, 512))
        self.register_buffer('proto_ph', torch.zeros(n_seen, 512))
        self.register_buffer('proto_mask', torch.zeros(n_seen, dtype=torch.bool))
        self.ema_m = 0.9
        
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        self.saved_features = defaultdict(lambda: {"sketch": [], "photo": []})

    def on_train_epoch_start(self):
        self.clip_frozen.eval()

    @torch.no_grad()
    def build_text_anchor(self, class_names):
        import clip
        templates = ["a photo of a {}.", "a sketch of a {}.", "a drawing of a {}.", "an image of a {}."]
        embs = []
        for c in class_names:
            toks = clip.tokenize([t.format(c.replace('_',' ')) for t in templates])
            e = self.clip_frozen.encode_text(toks).float() # Frozen
            e = F.normalize(F.normalize(e, dim=-1).mean(0), dim=-1)
            embs.append(e)
        E = torch.stack(embs) # [C_s, 512]
        self.register_buffer('text_emb_seen', E)
        self.register_buffer('anchor_A', E @ E.t())

    def forward(self, img_tensor, dtype='photo'):
        return self.model(img_tensor, type=dtype)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)
        return [optimizer] , [scheduler]
        
    @torch.no_grad()
    def update_protos(self, feats, cat_idx, which):
        bank = self.proto_sk if which == 'sk' else self.proto_ph
        for c in cat_idx.unique():
            f = F.normalize(feats[cat_idx == c].mean(0), dim=-1)
            if self.proto_mask[c]:
                bank[c] = F.normalize(self.ema_m * bank[c] + (1 - self.ema_m) * f, dim=-1)
            else:
                bank[c] = f
            self.proto_mask[c] = True

    def structural_losses(self, sk_feat, img_feat, cat_idx, dist='mse', T=0.1):
        idx = self.proto_mask.nonzero(as_tuple=True)[0]
        if idx.numel() < 10:
            z = torch.tensor(0., device=self.device)
            return z, z
            
        Psk, Pph = self.proto_sk.clone(), self.proto_ph.clone()
        for c in cat_idx.unique():
            Psk[c] = F.normalize(sk_feat[cat_idx == c].mean(0), dim=-1)
            Pph[c] = F.normalize(img_feat[cat_idx == c].mean(0), dim=-1)
            
        A = self.anchor_A[idx][:, idx]
        Ssk = Psk[idx] @ Psk[idx].t()
        Sph = Pph[idx] @ Pph[idx].t()
        
        if dist == 'mse':
            loss_ssc = F.mse_loss(Ssk, A) + F.mse_loss(Sph, A)
            loss_xmod = F.mse_loss(Ssk, Sph.detach()) + F.mse_loss(Sph, Ssk.detach())
        else: # kl
            kl = lambda S, R: F.kl_div(F.log_softmax(S/T, -1), F.softmax(R/T, -1), reduction='batchmean')
            loss_ssc = kl(Ssk, A) + kl(Sph, A)
            loss_xmod = 0.5 * (kl(Ssk, Sph.detach()) + kl(Sph, Ssk.detach()))
            
        return loss_ssc, loss_xmod
    
    def loss_fn(self, sk_feat, img_feat, neg_feat):
        triplet = nn.TripletMarginWithDistanceLoss(distance_function=self.distance_fn, margin=0.3)
        return triplet(sk_feat, img_feat, neg_feat)

    def classification_loss(self, sk_feat, img_feat, cat_idx):
        logits_sk = self.model.logit_scale.exp() * sk_feat @ self.text_emb_seen.t()
        logits_img = self.model.logit_scale.exp() * img_feat @ self.text_emb_seen.t()
        return 0.5 * (F.cross_entropy(logits_sk, cat_idx) + F.cross_entropy(logits_img, cat_idx))

    def training_step(self, batch, batch_idx):
        sk, img, neg, cat_name, cat_idx = batch
        img_feat = self.forward(img, dtype='photo')
        sk_feat = self.forward(sk, dtype='sketch')
        neg_feat = self.forward(neg, dtype='photo')
        
        loss_tri = self.loss_fn(sk_feat, img_feat, neg_feat)
        loss_cls = self.classification_loss(sk_feat, img_feat, cat_idx)
        
        self.update_protos(sk_feat.detach(), cat_idx, 'sk')
        self.update_protos(img_feat.detach(), cat_idx, 'ph')
        
        loss_ssc, loss_xmod = self.structural_losses(sk_feat, img_feat, cat_idx, dist=self.args.ssc_dist)
        
        loss = loss_tri + self.args.lambd * loss_cls + self.args.l_ssc * (loss_ssc + self.args.l_x * loss_xmod)
        self.log_dict({'tri': loss_tri, 'cls': loss_cls, 'ssc': loss_ssc, 'xmod': loss_xmod, 'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image_tensor, label = batch
        if dataloader_idx == 0:
            feat = self.forward(image_tensor, dtype='sketch')
            self.val_step_outputs_sk.append((feat, label))
            modality = "sketch"
        else:
            feat = self.forward(image_tensor, dtype='photo')
            self.val_step_outputs_ph.append((feat, label))
            modality = "photo"
        
        if self.args.visualize:
            feat = feat.detach().cpu()
            label = label.detach().cpu()
            for f, l in zip(feat, label):
                self.saved_features[str(int(l))][modality].append(f)
    
    def on_validation_epoch_end(self):
        if self.args.visualize:
            visualize_classes = VISUALIZE_CLASSES[self.args.dataset]
            visualize_tsne(visualize_classes, self.saved_features, mode="photo")
            visualize_tsne(visualize_classes, self.saved_features, mode="sketch")
        else:
            query_len = len(self.val_step_outputs_sk)
            gallery_len = len(self.val_step_outputs_ph)
            
            if query_len == 0 or gallery_len == 0:
                print("Skip validation")
                return

            query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
            gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])
            
            all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
            all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))
            
            gallery = gallery_feat_all
            ap = torch.zeros(len(query_feat_all))
            precision = torch.zeros(len(query_feat_all))
            if self.args.dataset == "sketchy_2":
                map_k = 200
                p_k = 200
            else:
                map_k = 0
                if self.args.dataset == "quickdraw":
                    p_k = 200
                else:
                    p_k = 100
                    
            for idx, sk_feat in enumerate(query_feat_all):
                category = all_sketch_category[idx]
                distance = self.distance_fn(sk_feat.unsqueeze(0), gallery)
                target = torch.zeros(len(gallery), dtype=torch.bool, device=device)
                target[np.where(all_photo_category == category)] = True
                
                if map_k != 0:
                    top_k_actual = min(map_k, len(gallery)) 
                    ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=top_k_actual)
                else: 
                    ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
                    
                precision[idx] = retrieval_precision(distance.cpu(), target.cpu(), top_k=p_k)
                
            mAP = torch.mean(ap)
            precision = torch.mean(precision)
            self.log("mAP", mAP, on_step=False, on_epoch=True)
            if self.global_step > 0:
                self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
            
            if map_k != 0:
                print('mAP@{}: {}, P@{}: {}, Best mAP: {}'.format(map_k, mAP.item(), p_k, precision, self.best_metric))
            else:
                print('mAP@all: {}, P@{}: {}, Best mAP: {}'.format(mAP.item(), p_k, precision, self.best_metric))
            train_loss = self.trainer.callback_metrics.get("train_loss", None)

            if train_loss is not None:
                print(f"Train loss (epoch avg): {train_loss.item():.6f}")
                
        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
        self.saved_features.clear()