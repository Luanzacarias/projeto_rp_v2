# centernet_light_helmet.py (treino + validação, early stopping, AMP novo, checkpoint, precisão)
import os
import math
import random
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------
# Métricas: IoU e extração de caixas GT a partir dos targets
# ------------------------
def box_iou_matrix(a, b):
    # a: [Na, 4], b: [Nb, 4], formato [x1,y1,x2,y2]
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    x1 = torch.max(a[:, None, 0], b[None, :, 0])
    y1 = torch.max(a[:, None, 1], b[None, :, 1])
    x2 = torch.min(a[:, None, 2], b[None, :, 2])
    y2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-9)

def targets_to_boxes(wh, reg, mask, down=16):
    """
    Converte (wh, reg, mask) do target em uma lista de caixas GT em pixels.
    wh/reg/mask: tensores [2,H,W], [2,H,W], [H,W]
    """
    H, W = mask.shape
    ys, xs = torch.nonzero(mask > 0, as_tuple=True)
    if xs.numel() == 0:
        return torch.empty((0,4), device=wh.device)

    bw = wh[0, ys, xs]  # em células (grid)
    bh = wh[1, ys, xs]
    rx = reg[0, ys, xs]
    ry = reg[1, ys, xs]

    # centro (em células) + regressão
    xs_f = xs.float() + rx
    ys_f = ys.float() + ry

    # para pixels
    x1 = (xs_f - bw/2) * down
    y1 = (ys_f - bh/2) * down
    x2 = (xs_f + bw/2) * down
    y2 = (ys_f + bh/2) * down

    return torch.stack([x1, y1, x2, y2], dim=-1)

# ------------------------
# Utils: gaussian, focal loss
# ------------------------
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2*radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter/6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
    masked_gaussian = gaussian[radius-top:radius+bottom, radius-left:radius+right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def focal_loss(pred, gt):
    # Modified focal from CenterNet
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = neg_loss
    else:
        loss = (pos_loss + neg_loss) / num_pos
    return loss

# ------------------------
# Dataset: images + YOLO txt -> alvos CenterNet
# ------------------------
class YoloToCenterNetDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=512, down=16, transform=None):
        self.img_paths = sorted([p for p in glob(os.path.join(images_dir, "*")) if os.path.isfile(p)])
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.down = down  # output stride
        self.out_size = img_size // down
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, fname + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        # targets
        num_classes = 1
        heatmap = np.zeros((num_classes, self.out_size, self.out_size), dtype=np.float32)
        wh = np.zeros((2, self.out_size, self.out_size), dtype=np.float32)
        reg = np.zeros((2, self.out_size, self.out_size), dtype=np.float32)
        reg_mask = np.zeros((self.out_size, self.out_size), dtype=np.uint8)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls = 0  # uma classe
                    xc = float(parts[1]) * self.img_size
                    yc = float(parts[2]) * self.img_size
                    bw = float(parts[3]) * self.img_size
                    bh = float(parts[4]) * self.img_size

                    # para o mapa de saída
                    xc_o = xc / self.down
                    yc_o = yc / self.down
                    bw_o = bw / self.down
                    bh_o = bh / self.down

                    ct = np.array([xc_o, yc_o])
                    ct_int = ct.astype(np.int32)

                    # raio gaussiano
                    radius = gaussian_radius((math.ceil(bh_o), math.ceil(bw_o)))
                    radius = max(0, int(radius))
                    draw_gaussian(heatmap[cls], ct_int, radius)

                    x_i, y_i = ct_int[0], ct_int[1]
                    if 0 <= x_i < self.out_size and 0 <= y_i < self.out_size:
                        wh[0, y_i, x_i] = bw_o
                        wh[1, y_i, x_i] = bh_o
                        reg[0, y_i, x_i] = ct[0] - x_i
                        reg[1, y_i, x_i] = ct[1] - y_i
                        reg_mask[y_i, x_i] = 1

        img_t = torch.from_numpy(img).permute(2,0,1).float()
        return img_t, torch.from_numpy(heatmap).float(), torch.from_numpy(wh).float(), torch.from_numpy(reg).float(), torch.from_numpy(reg_mask).float()

# gaussian radius helper usado no CenterNet
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(0.0, b1 ** 2 - 4 * a1 * c1))
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(0.0, b2 ** 2 - 4 * a2 * c2))
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(0.0, b3 ** 2 - 4 * a3 * c3))
    r3  = (b3 + sq3) / 2
    return max(0.0, min(r1, r2, r3))

# ------------------------
# Modelo
# ------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(3,64,3,2,1)   # /2
        self.l1 = ConvBNReLU(64,128,3,2,1)   # /4
        self.l2 = ConvBNReLU(128,256,3,2,1)  # /8
        self.l3 = ConvBNReLU(256,512,3,2,1)  # /16
        self.reduce = ConvBNReLU(512,256,1,1,0)

    def forward(self,x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.reduce(x)
        return x

class CenterNetLight(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.hm_head = nn.Sequential(
            ConvBNReLU(256,128,3,1,1),
            nn.Conv2d(128, num_classes, 1)
        )
        self.wh_head = nn.Sequential(
            ConvBNReLU(256,128,3,1,1),
            nn.Conv2d(128, 2, 1)
        )
        self.reg_head = nn.Sequential(
            ConvBNReLU(256,128,3,1,1),
            nn.Conv2d(128, 2, 1)
        )

    def forward(self, x):
        f = self.backbone(x)
        hm = torch.sigmoid(self.hm_head(f))
        wh = self.wh_head(f)
        reg = self.reg_head(f)
        return hm, wh, reg

# ------------------------
# Decode helper
# ------------------------
def decode_heatmap(hm, wh, reg, K=100, down=16):
    # hm: (B, C, H, W)
    batch, c, H, W = hm.shape
    hm_ = hm.reshape(batch, c, -1)
    scores, inds = torch.topk(hm_, K)
    scores = scores.view(batch, K)
    inds = inds.view(batch, K)
    xs = (inds % W).float()
    ys = (inds // W).float()
    # gather wh/reg
    wh = wh.reshape(batch, 2, -1)
    reg = reg.reshape(batch, 2, -1)
    wh_k = torch.stack([wh[:,0,:].gather(1, inds), wh[:,1,:].gather(1, inds)], dim=-1)  # B,K,2
    reg_k = torch.stack([reg[:,0,:].gather(1, inds), reg[:,1,:].gather(1, inds)], dim=-1)
    xs = xs + reg_k[...,0]
    ys = ys + reg_k[...,1]
    # convert to boxes
    x1 = (xs - wh_k[...,0]/2) * down
    y1 = (ys - wh_k[...,1]/2) * down
    x2 = (xs + wh_k[...,0]/2) * down
    y2 = (ys + wh_k[...,1]/2) * down
    boxes = torch.stack([x1,y1,x2,y2], dim=-1)
    return boxes, scores

# ------------------------
# Train / Val helpers
# ------------------------
def compute_losses(hm_pred, wh_pred, reg_pred, hms, whs, regs, masks, device):
    loss_hm = focal_loss(hm_pred, hms)
    mask_bool = (masks > 0)
    mask_expand = mask_bool.unsqueeze(1)  # B,1,H,W
    if mask_expand.sum() > 0:
        loss_wh = F.l1_loss(wh_pred * mask_expand, whs * mask_expand, reduction='sum') / (mask_expand.sum()+1e-6)
        loss_reg = F.l1_loss(reg_pred * mask_expand, regs * mask_expand, reduction='sum') / (mask_expand.sum()+1e-6)
    else:
        loss_wh = torch.tensor(0.0, device=device)
        loss_reg = torch.tensor(0.0, device=device)
    loss = loss_hm + 0.1*loss_wh + 1.0*loss_reg
    return loss, {'hm': loss_hm.item(), 'wh': loss_wh.item(), 'reg': loss_reg.item()}

@torch.no_grad()
def validate(model, loader, device, amp=False, iou_thr=0.5, score_thr=0.3, topk=100, down=16):
    model.eval()
    val_loss = 0.0
    hm_loss = 0.0
    wh_loss = 0.0
    reg_loss = 0.0
    total_tp = 0
    total_fp = 0

    for imgs, hms, whs, regs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)
        whs = whs.to(device, non_blocking=True)
        regs = regs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type=="cuda")):
            hm_pred, wh_pred, reg_pred = model(imgs)
            loss, parts = compute_losses(hm_pred, wh_pred, reg_pred, hms, whs, regs, masks, device)

        val_loss += loss.item()
        hm_loss += parts['hm']
        wh_loss += parts['wh']
        reg_loss += parts['reg']

        # precisão (greedy IoU)
        boxes_pred, scores_pred = decode_heatmap(hm_pred, wh_pred, reg_pred, K=topk, down=down)  # [B,K,4], [B,K]
        B = imgs.shape[0]
        for b in range(B):
            scores_b = scores_pred[b]
            keep = scores_b > score_thr
            if keep.sum() == 0:
                continue
            boxes_b = boxes_pred[b][keep]
            scores_b = scores_b[keep]

            gt_b = targets_to_boxes(whs[b], regs[b], masks[b], down=down)  # [N,4]
            order = torch.argsort(scores_b, descending=True)
            boxes_b = boxes_b[order]

            if gt_b.numel() == 0:
                total_fp += boxes_b.shape[0]
                continue

            iou = box_iou_matrix(boxes_b, gt_b)  # [M, N]
            matched_gt = torch.zeros(gt_b.shape[0], dtype=torch.bool, device=device)
            tp = 0
            for i in range(boxes_b.shape[0]):
                best_gt = torch.argmax(iou[i])
                if iou[i, best_gt] >= iou_thr and not matched_gt[best_gt]:
                    matched_gt[best_gt] = True
                    tp += 1
            fp = boxes_b.shape[0] - tp
            total_tp += int(tp)
            total_fp += int(fp)

    n = len(loader)
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    return {
        'loss': val_loss / max(1, n),
        'hm': hm_loss / max(1, n),
        'wh': wh_loss / max(1, n),
        'reg': reg_loss / max(1, n),
        'precision': precision
    }

def train(model, train_loader, val_loader, optimizer, device, epochs=30, amp=True, grad_clip=1.0, scheduler=None, patience=10, outdir="weights", down=16):
    model.to(device)
    # novo GradScaler API
    scaler = torch.amp.GradScaler(enabled=(amp and device.type=="cuda"))
    best_val = float('inf')
    patience_ctr = 0

    os.makedirs(outdir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, hms, whs, regs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            hms = hms.to(device, non_blocking=True)
            whs = whs.to(device, non_blocking=True)
            regs = regs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp and device.type == "cuda":
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    hm_pred, wh_pred, reg_pred = model(imgs)
                    loss, _ = compute_losses(hm_pred, wh_pred, reg_pred, hms, whs, regs, masks, device)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                hm_pred, wh_pred, reg_pred = model(imgs)
                loss, _ = compute_losses(hm_pred, wh_pred, reg_pred, hms, whs, regs, masks, device)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{running/steps:.4f}")

        # validação
        val_stats = validate(
            model, val_loader, device,
            amp=amp, iou_thr=0.5, score_thr=0.3, topk=100, down=down
        )
        if scheduler is not None:
            scheduler.step(val_stats['loss'] if hasattr(scheduler, 'optimizer') else None)

        print(
            f"Epoch {epoch}: train_loss={running/len(train_loader):.4f} | "
            f"val_loss={val_stats['loss']:.4f} "
            f"(hm={val_stats['hm']:.4f}, wh={val_stats['wh']:.4f}, reg={val_stats['reg']:.4f}) | "
            f"val_precision@IoU0.50,score{0.3:.2f}={val_stats['precision']:.4f}"
        )

        # checkpoint do melhor
        if val_stats['loss'] < best_val - 1e-6:
            best_val = val_stats['loss']
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(outdir, "best.pth"))
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                        'best_val': best_val}, os.path.join(outdir, "last.pt"))
            print(f">>> Melhor val_loss até agora: {best_val:.4f} | checkpoint salvo em {outdir}/best.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping (paciência={patience}). Melhor val_loss: {best_val:.4f}")
                break

# ------------------------
# Builders de DataLoader
# ------------------------
def build_split_loaders(images, labels, imgsize, down, batch, workers, val_split=0.2, seed=42):
    full_ds = YoloToCenterNetDataset(images, labels, img_size=imgsize, down=down)
    n = len(full_ds)
    val_n = int(round(n * val_split))
    train_n = max(1, n - val_n)
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader

def build_separate_loaders(tr_images, tr_labels, va_images, va_labels, imgsize, down, batch, workers):
    train_ds = YoloToCenterNetDataset(tr_images, tr_labels, img_size=imgsize, down=down)
    val_ds   = YoloToCenterNetDataset(va_images, va_labels, img_size=imgsize, down=down)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # treino
    parser.add_argument("--images", required=True, help="pasta de imagens (treino)")
    parser.add_argument("--labels", required=True, help="pasta de labels YOLO (treino)")
    # validação opcional dedicada
    parser.add_argument("--val-images", type=str, default=None, help="pasta de imagens (val)")
    parser.add_argument("--val-labels", type=str, default=None, help="pasta de labels YOLO (val)")
    # split interno se não houver pastas de val
    parser.add_argument("--val-split", type=float, default=0.2, help="fração de validação quando não há pastas de val")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsize", type=int, default=640)
    parser.add_argument("--down", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="desativa mixed precision (AMP)")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience (épocas sem melhorar)")
    parser.add_argument("--outdir", type=str, default="weights")
    args = parser.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Loaders
    if args.val_images and args.val_labels:
        train_loader, val_loader = build_separate_loaders(
            args.images, args.labels, args.val_images, args.val_labels,
            imgsize=args.imgsize, down=args.down, batch=args.batch, workers=args.workers
        )
    else:
        train_loader, val_loader = build_split_loaders(
            args.images, args.labels,
            imgsize=args.imgsize, down=args.down, batch=args.batch, workers=args.workers,
            val_split=args.val_split, seed=args.seed
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNetLight(num_classes=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    print("Model parameter count (trainable):", sum(p.numel() for p in model.parameters() if p.requires_grad))
    train(model,
          train_loader,
          val_loader,
          optimizer,
          device,
          epochs=args.epochs,
          amp=(not args.no_amp),
          grad_clip=1.0,
          scheduler=scheduler,
          patience=args.patience,
          outdir=args.outdir,
          down=args.down)

    # Salva último estado (o melhor já foi salvo como best.pth)
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.outdir, "last.pth"))
    print(f"Treino finalizado. Pesos: {args.outdir}/best.pth (melhor) e {args.outdir}/last.pth (último).")
