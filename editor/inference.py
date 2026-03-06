from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

try:
    from LayoutLM.graph_linking.entity_encoder import ENTITY_LABEL2ID
    from LayoutLM.graph_linking.gnn_model import GATLinkPredictor
    from LayoutLM.graph_linking.graph_builder import build_graph
    HAS_GRAPH_LINKING = True
except ImportError:
    HAS_GRAPH_LINKING = False


class InferenceError(RuntimeError):
    pass


@dataclass
class _WordPred:
    text: str
    box_1000: List[int]
    label: str
    emb: torch.Tensor


class LayoutLMGraphLinkingService:
    """
    Image -> canonical JSON:
      1) OCR + token classification via fine-tuned LayoutLMv3
      2) Entity block reconstruction from BIO labels
      3) Tentative Q->A linking via trained GNN
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        gnn_checkpoint: Optional[Path] = None,
        threshold: float = 0.5,
    ):
        root = Path(__file__).resolve().parent.parent
        self.checkpoint_dir = checkpoint_dir or (root / "LayoutLM" / "layoutlmv3-funsd" / "checkpoint-608")
        self.gnn_checkpoint = gnn_checkpoint or (
            root / "LayoutLM" / "graph_linking" / "checkpoints" / "best_model.pt"
        )
        self.threshold = threshold
        self.device = self._pick_device()

        if not self.checkpoint_dir.exists():
            raise InferenceError(f"LayoutLMv3 checkpoint not found: {self.checkpoint_dir}")
        if not self.gnn_checkpoint.exists():
            raise InferenceError(f"GNN checkpoint not found: {self.gnn_checkpoint}")
        self._check_ocr_dependencies()

        try:
            from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast
            image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
            self.ocr_processor = LayoutLMv3ImageProcessor(apply_ocr=True)
            tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
                str(self.checkpoint_dir), local_files_only=True
            )
            self.processor = LayoutLMv3Processor(
                image_processor=image_processor, tokenizer=tokenizer, apply_ocr=False
            )
        except Exception as e:
            raise InferenceError(
                "Failed to load LayoutLMv3 processor from local checkpoint. "
                "Ensure OCR dependencies are available."
            ) from e

        self.token_model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(self.checkpoint_dir),
            local_files_only=True,
        ).to(self.device)
        self.token_model.eval()

        if HAS_GRAPH_LINKING and self.gnn_checkpoint.exists():
            self.gnn_model = GATLinkPredictor().to(self.device)
            ckpt = torch.load(self.gnn_checkpoint, map_location=self.device)
            self.gnn_model.load_state_dict(ckpt["model_state"])
            self.gnn_model.eval()
        else:
            self.gnn_model = None

        self.id2label = {int(k): v for k, v in self.token_model.config.id2label.items()}

    def extract(self, image_path: str | Path) -> Dict:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        words, boxes = self._run_ocr(image)
        if not words:
            return {"blocks": [], "links": [], "tables": []}

        word_preds = self._predict_word_labels(image, words, boxes)
        entities, node_feats = self._words_to_entities(word_preds)

        links = []
        if HAS_GRAPH_LINKING and self.gnn_model and entities and node_feats.numel() > 0:
            links = self._predict_links(entities, node_feats)

        blocks = [
            {
                "id": ent["id"],
                "text": ent["text"],
                "bbox": self._bbox_1000_to_px(ent["box"], width, height),
                "label": ent["label"],
            }
            for ent in entities
        ]

        return {"blocks": blocks, "links": links, "tables": []}

    def _run_ocr(self, image: Image.Image) -> Tuple[List[str], List[List[int]]]:
        """
        Uses processor OCR path (requires OCR runtime in environment).
        Boxes are in FUNSD-style 0..1000 coordinates.
        """
        try:
            ocr_out = self.ocr_processor(image, return_tensors="pt")
            words = list(ocr_out.words[0])
            boxes = [list(map(int, b)) for b in ocr_out.boxes[0]]
            return words, boxes
        except Exception as e:
            raise InferenceError(
                "OCR failed. Install OCR dependencies (pytesseract + Tesseract binary) "
                "or provide pre-tokenized OCR upstream."
            ) from e

    @staticmethod
    def _check_ocr_dependencies() -> None:
        try:
            import pytesseract  # type: ignore
            
            # Explicitly set the binary path for Homebrew on Apple Silicon
            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
        except Exception as e:
            raise InferenceError(
                "Missing OCR dependency: pytesseract. Install with `pip install pytesseract` "
                "and ensure the Tesseract binary is installed."
            ) from e
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise InferenceError(
                "Tesseract binary not found. Install Tesseract OCR and make sure it is in PATH."
            ) from e

    def _predict_word_labels(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[List[int]],
    ) -> List[_WordPred]:
        enc = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        word_ids = enc.word_ids(batch_index=0)
        inputs = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.token_model(**inputs, output_hidden_states=True)

        logits = out.logits.squeeze(0)  # (seq, num_labels)
        pred_ids = logits.argmax(-1).cpu().tolist()
        hidden = out.hidden_states[-1].squeeze(0).detach().cpu()  # (seq, 768)

        n_words = len(words)
        emb_sum = torch.zeros(n_words, hidden.shape[-1], dtype=torch.float32)
        emb_count = torch.zeros(n_words, dtype=torch.float32)
        word_label_ids = [0] * n_words  # default O
        seen = [False] * n_words

        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid >= n_words:
                continue
            emb_sum[wid] += hidden[tok_idx]
            emb_count[wid] += 1.0
            if not seen[wid]:
                word_label_ids[wid] = pred_ids[tok_idx]
                seen[wid] = True

        emb_count = emb_count.clamp(min=1.0)
        word_emb = emb_sum / emb_count.unsqueeze(-1)

        preds: List[_WordPred] = []
        for i in range(n_words):
            label_name = self.id2label.get(word_label_ids[i], "O")
            preds.append(
                _WordPred(
                    text=words[i],
                    box_1000=boxes[i],
                    label=label_name,
                    emb=word_emb[i],
                )
            )
        return preds

    def _words_to_entities(self, words: List[_WordPred]) -> Tuple[List[Dict], torch.Tensor]:
        entities: List[Dict] = []
        node_feats: List[torch.Tensor] = []

        current: List[_WordPred] = []
        current_label: Optional[str] = None

        def flush():
            nonlocal current, current_label
            if not current:
                return
            ent_label = current_label if current_label is not None else "other"
            box = self._union_box([w.box_1000 for w in current])
            text = " ".join(w.text for w in current).strip()
            emb = torch.stack([w.emb for w in current], dim=0).mean(dim=0)
            label_id = ENTITY_LABEL2ID.get(ent_label, 0)
            feat = torch.cat(
                [
                    emb,
                    torch.tensor([v / 1000.0 for v in box], dtype=torch.float32),
                    self._one_hot(label_id),
                ],
                dim=0,
            )
            ent_id = len(entities)
            entities.append(
                {
                    "idx": ent_id,
                    "id": ent_id,
                    "label": ent_label,
                    "label_id": label_id,
                    "box": box,
                    "text": text,
                    "words": [w.text for w in current],
                }
            )
            node_feats.append(feat)
            current = []
            current_label = None

        for w in words:
            tag = w.label.upper()
            if tag == "O":
                flush()
                current = [w]
                current_label = "other"
                flush()
                continue

            if tag.startswith("B-"):
                flush()
                current = [w]
                current_label = tag[2:].lower()
                continue

            if tag.startswith("I-"):
                raw_label = tag[2:].lower()
                if current and current_label == raw_label:
                    current.append(w)
                else:
                    flush()
                    current = [w]
                    current_label = raw_label
                continue

            flush()

        flush()

        if not node_feats:
            return [], torch.zeros((0, 776), dtype=torch.float32)
        return entities, torch.stack(node_feats, dim=0)

    def _predict_links(self, entities: List[Dict], node_feats: torch.Tensor) -> List[Dict]:
        graph = build_graph(entities, node_feats, gt_linking=set())
        if graph.edge_index.shape[1] == 0:
            return []

        graph = graph.to(self.device)
        with torch.no_grad():
            preds = self.gnn_model.predict_links(graph, threshold=self.threshold).cpu()

        edge_index = graph.edge_index.cpu()
        links = set()
        for i in range(edge_index.shape[1]):
            if not bool(preds[i].item()):
                continue
            src = int(edge_index[0, i].item())
            dst = int(edge_index[1, i].item())
            src_ent = entities[src]
            dst_ent = entities[dst]
            if src_ent["label"] == "question" and dst_ent["label"] == "answer":
                links.add((src_ent["id"], dst_ent["id"]))

        return [{"question_id": qid, "answer_id": aid} for qid, aid in sorted(links)]

    @staticmethod
    def _one_hot(label_id: int, num_classes: int = 4) -> torch.Tensor:
        v = torch.zeros(num_classes, dtype=torch.float32)
        if 0 <= label_id < num_classes:
            v[label_id] = 1.0
        return v

    @staticmethod
    def _union_box(boxes: List[List[int]]) -> List[int]:
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return [int(x1), int(y1), int(x2), int(y2)]

    @staticmethod
    def _bbox_1000_to_px(box: List[int], width: int, height: int) -> List[float]:
        return [
            box[0] * width / 1000.0,
            box[1] * height / 1000.0,
            box[2] * width / 1000.0,
            box[3] * height / 1000.0,
        ]

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
