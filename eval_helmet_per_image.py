# eval_helmet_per_image.py (compatível com treino: ConvNormReLU com .norm)
import os
import math
from glob import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Modelo (MESMA definição do treino) ----------
class ConvNormReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, norm='bn'):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        if norm == 'gn':
            groups = min(32, out_c) if out_c >= 8 else 1
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_c)
        else:
            self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.act(self.norm(self.conv(x)))

class SimpleBackbone(nn.Module):
    def __init__(self, norm='bn'):
        super().__init__()
        self.stem   = ConvNormReLU(3,64,3,2,1,norm)   # /2
        self.l1     = ConvNormReLU(64,128,3,2,1,norm) # /4
        self.l2     = ConvNormReLU(128,256,3,2,1,norm)# /8
        self.l3     = ConvNormReLU(256,512,3,2,1,norm)# /16
        self.reduce = ConvNormReLU(512,256,1,1,0,norm)
    def forward(self,x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.reduce(x)
        return x

class CenterNetLight(nn.Module):
    def __init__(self, num_classes=1, norm='bn'):
        super().__init__()
        self.backbone = SimpleBackbone(norm=norm)
        self.hm_head  = nn.Sequential(ConvNormReLU(256,128,3,1,1,norm), nn.Conv2d(128, num_classes, 1))
        self.wh_head  = nn.Sequential(ConvNormReLU(256,128,3,1,1,norm), nn.Conv2d(128, 2, 1))
        self.reg_head = nn.Sequential(ConvNormReLU(256,128,3,1,1,norm), nn.Conv2d(128, 2, 1))
    def forward(self, x):
        f = self.backbone(x)
        hm  = torch.sigmoid(self.hm_head(f))
        wh  = self.wh_head(f)
        reg = self.reg_head(f)
        return hm, wh, reg

# ---------- Decode / Métricas ----------
def decode_heatmap(hm, wh, reg, K=100, down=16):
    # hm: (B, C, H, W)  -> caixas [B,K,4] em pixels e scores [B,K]
    batch, c, H, W = hm.shape
    hm_ = hm.reshape(batch, c, -1)
    scores, inds = torch.topk(hm_, K)
    scores = scores.view(batch, K)
    inds = inds.view(batch, K)
    xs = (inds % W).float()
    ys = (inds // W).float()

    wh  = wh.reshape(batch, 2, -1)
    reg = reg.reshape(batch, 2, -1)
    wh_k  = torch.stack([wh[:,0,:].gather(1, inds),  wh[:,1,:].gather(1, inds)],  dim=-1)  # B,K,2
    reg_k = torch.stack([reg[:,0,:].gather(1, inds), reg[:,1,:].gather(1, inds)], dim=-1)

    xs = xs + reg_k[...,0]
    ys = ys + reg_k[...,1]

    x1 = (xs - wh_k[...,0]/2) * down
    y1 = (ys - wh_k[...,1]/2) * down
    x2 = (xs + wh_k[...,0]/2) * down
    y2 = (ys + wh_k[...,1]/2) * down
    boxes = torch.stack([x1,y1,x2,y2], dim=-1)
    return boxes, scores

def box_iou_matrix(a, b):
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

def nms(boxes, scores, iou_thr=0.5):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes, scores, iou_thr)
    except Exception:
        keep = []
        idxs = scores.argsort(descending=True)
        while idxs.numel() > 0:
            i = idxs[0]
            keep.append(i.item())
            if idxs.numel() == 1:
                break
            ious = box_iou_matrix(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
            idxs = idxs[1:][ious <= iou_thr]
        return torch.tensor(keep, device=boxes.device, dtype=torch.long)

def load_gt_boxes_yolo(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return torch.empty((0,4))
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: 
                continue
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            x1 = cx - bw/2; y1 = cy - bh/2; x2 = cx + bw/2; y2 = cy + bh/2
            boxes.append([x1,y1,x2,y2])
    return torch.tensor(boxes, dtype=torch.float32) if len(boxes) else torch.empty((0,4))

# ---------- Visualização opcional ----------
def draw_boxes(img_bgr, boxes, color, thickness=2, text=None):
    out = img_bgr.copy()
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = [int(v) for v in b]
        cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
        if text is not None:
            cv2.putText(out, f"{text[i] if isinstance(text, (list, tuple)) else text}",
                        (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

# ---------- Avaliação por imagem ----------
def evaluate_folder(model, device, images_dir, labels_dir, img_size=640, down=16,
                    score_thr=0.3, iou_thr=0.5, topk=100, nms_iou=0.5,
                    save_vis=None, print_each=True):
    img_paths = sorted([p for p in glob(os.path.join(images_dir, "*")) if os.path.isfile(p)])
    assert len(img_paths)>0, "Nenhuma imagem encontrada."

    model.eval()
    total_TP = total_FP = total_FN = 0
    presence_correct = 0

    for img_path in tqdm(img_paths, desc="Avaliando"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            if print_each: print(f"[WARN] Não abriu: {img_path}")
            continue
        H0, W0 = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32)/255.0
        tensor = torch.from_numpy(img_res).permute(2,0,1).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
            hm, wh, reg = model(tensor)
            boxes_pred, scores_pred = decode_heatmap(hm, wh, reg, K=topk, down=down)  # [1,K,4], [1,K]

        boxes = boxes_pred[0]; scores = scores_pred[0]

        # reescala para resolução original
        sx, sy = W0 / img_size, H0 / img_size
        boxes = boxes * torch.tensor([sx, sy, sx, sy], device=boxes.device)

        keep = scores > score_thr
        boxes = boxes[keep]; scores = scores[keep]

        if boxes.numel() > 0 and nms_iou > 0:
            keep_idx = nms(boxes, scores, iou_thr=nms_iou)
            boxes = boxes[keep_idx]; scores = scores[keep_idx]

        # GT
        fname = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, fname + ".txt")
        gt = load_gt_boxes_yolo(label_path, W0, H0).to(device)

        # presença (tem/não tem)
        pred_has = boxes.shape[0] > 0
        gt_has = gt.shape[0] > 0
        if pred_has == gt_has:
            presence_correct += 1

        # TP/FP/FN por greedy IoU
        TP = FP = FN = 0
        if boxes.numel() == 0 and gt.numel() == 0:
            pass
        elif boxes.numel() == 0 and gt.numel() > 0:
            FN = gt.shape[0]
        elif boxes.numel() > 0 and gt.numel() == 0:
            FP = boxes.shape[0]
        else:
            iou = box_iou_matrix(boxes, gt)  # [M,N]
            matched_gt = torch.zeros(gt.shape[0], dtype=torch.bool, device=device)
            order = torch.argsort(scores, descending=True)
            boxes = boxes[order]; iou = iou[order]
            for i in range(boxes.shape[0]):
                best_gt = torch.argmax(iou[i])
                if iou[i, best_gt] >= iou_thr and not matched_gt[best_gt]:
                    matched_gt[best_gt] = True
                    TP += 1
                else:
                    FP += 1
            FN = int((~matched_gt).sum().item())

        total_TP += TP; total_FP += FP; total_FN += FN

        if print_each:
            print(f"{os.path.basename(img_path)} | TP={TP} FP={FP} FN={FN} | preds={int(scores.numel())} gt={gt.shape[0]}")

        if save_vis:
            vis = img_bgr.copy()
            if gt.numel() > 0:
                vis = draw_boxes(vis, gt.cpu().numpy(), (0,255,0), 2)            # GT verde
            if boxes.numel() > 0:
                txt = [f"{float(s):.2f}" for s in scores.cpu().numpy()]
                vis = draw_boxes(vis, boxes.cpu().numpy(), (0,0,255), 2, text=txt)# Pred vermelho
            os.makedirs(save_vis, exist_ok=True)
            cv2.imwrite(os.path.join(save_vis, fname + ".jpg"), vis)

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    presence_acc = presence_correct / len(img_paths) if len(img_paths) > 0 else 0.0

    print("\n=== RESULTADOS FINAIS ===")
    print(f"Imagens avaliadas: {len(img_paths)}")
    print(f"TP={total_TP} FP={total_FP} FN={total_FN}")
    print(f"Precisão (P) : {precision:.4f}")
    print(f"Recall   (R) : {recall:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"Acurácia por PRESENÇA (prediz 'tem capacete' vs 'não tem'): {presence_acc:.4f}")

    return {
        "tp": total_TP, "fp": total_FP, "fn": total_FN,
        "precision": precision, "recall": recall, "f1": f1,
        "presence_acc": presence_acc, "images": len(img_paths)
    }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="pasta com imagens de teste")
    ap.add_argument("--labels", required=True, help="pasta com labels YOLO das imagens de teste")
    ap.add_argument("--weights", required=True, help="caminho para best.pth/last.pth")
    ap.add_argument("--imgsize", type=int, default=640)
    ap.add_argument("--down", type=int, default=16)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--nms-iou", type=float, default=0.5, help="IoU do NMS (0 desativa)")
    ap.add_argument("--cpu", action="store_true", help="forçar CPU")
    ap.add_argument("--save-vis", type=str, default=None, help="pasta para salvar imagens com predições (opcional)")
    ap.add_argument("--quiet", action="store_true", help="não imprimir linha por imagem")
    ap.add_argument("--norm", type=str, default="bn", choices=["bn","gn"], help="tipo de normalização usado no treino")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = CenterNetLight(num_classes=1, norm=args.norm)
    state = torch.load(args.weights, map_location=device)
    # suporta best.pth (state_dict puro) e last.pt (dict com "model")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)  # nomes agora batem (.norm)

    model.to(device)

    evaluate_folder(model, device,
                    images_dir=args.images,
                    labels_dir=args.labels,
                    img_size=args.imgsize,
                    down=args.down,
                    score_thr=args.score_thr,
                    iou_thr=args.iou_thr,
                    topk=args.topk,
                    nms_iou=args.nms_iou,
                    save_vis=args.save_vis,
                    print_each=(not args.quiet))

if __name__ == "__main__":
    main()
