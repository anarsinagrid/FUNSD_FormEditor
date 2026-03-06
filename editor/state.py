from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QObject, Signal


class DocumentState(QObject):
    """
    Canonical state holder. Pure data + signals; no UI logic.
    """

    state_changed = Signal()
    selection_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.data: Dict = {"blocks": [], "links": [], "tables": []}
        self.selected_block_id: Optional[int] = None
        self._id_counter = 0
        self._table_id_counter = 0
        self.ocr_engine = "doctr"
        self.model_choice = {"arch": "layoutlmv3", "source": "gt"}  # ocr|gt paired with arch
        self.image_size = None  # (width, height) of current image, px

    # -------- Blocks --------
    def add_block(self, text: str, bbox: List[float], label: str = "other") -> Dict:
        block = {
            "id": self._id_counter,
            "text": text,
            "bbox": bbox,
            "label": label,
        }
        self._id_counter += 1
        self.data["blocks"].append(block)
        self.state_changed.emit()
        return block

    def update_block(self, block_id: int, **kwargs) -> None:
        block = self.get_block(block_id)
        if not block:
            return
        block.update({k: v for k, v in kwargs.items() if v is not None})
        self.state_changed.emit()

    def delete_block(self, block_id: int) -> None:
        self.data["blocks"] = [b for b in self.data.get("blocks", []) if b["id"] != block_id]
        self.data["links"] = [
            l
            for l in self.data.get("links", [])
            if l["question_id"] != block_id and l["answer_id"] != block_id
        ]
        for table in self.data.get("tables", []):
            table["cells"] = [c for c in table.get("cells", []) if c["block_id"] != block_id]
        if self.selected_block_id == block_id:
            self.selected_block_id = None
        self.state_changed.emit()

    def split_block(self, block_id: int, split_at: int | None = None) -> List[int]:
        """
        Horizontal split (left / right): creates two side-by-side boxes at the
        midpoint of the bounding box's x-axis.
        Words are distributed: first half to left box, second half to right box.
        """
        block = self.get_block(block_id)
        if not block:
            return []
        bbox = block.get("bbox", [0, 0, 100, 20])
        x1, y1, x2, y2 = bbox
        mid_x = (x1 + x2) / 2.0

        text  = block.get("text", "")
        words = text.split()
        if split_at is None:
            split_at = max(1, len(words) // 2)
        split_at = max(1, min(split_at, len(words) - 1))
        if len(words) >= 2:
            text_left  = " ".join(words[:split_at])
            text_right = " ".join(words[split_at:])
        else:
            text_left  = text
            text_right = ""

        label    = block.get("label", "other")
        # x split proportional to word index within the text
        n_words  = len(words)
        frac     = split_at / n_words if n_words > 0 else 0.5
        split_x  = x1 + (x2 - x1) * frac
        left_id  = self.add_block(text_left,  [x1,      y1, split_x, y2], label)["id"]
        right_id = self.add_block(text_right, [split_x, y1, x2,      y2], label)["id"]
        self.delete_block(block_id)
        return [left_id, right_id]

    def merge_blocks(self, block_ids: List[int]) -> Optional[int]:
        blocks = [self.get_block(bid) for bid in block_ids]
        blocks = [b for b in blocks if b]
        if len(blocks) < 2:
            return None
        x1 = min(b.get("bbox", [0, 0, 0, 0])[0] for b in blocks)
        y1 = min(b.get("bbox", [0, 0, 0, 0])[1] for b in blocks)
        x2 = max(b.get("bbox", [0, 0, 0, 0])[2] for b in blocks)
        y2 = max(b.get("bbox", [0, 0, 0, 0])[3] for b in blocks)
        text = self._merge_block_text_layout_aware(blocks)
        label = self._majority_label(blocks)
        new_block = self.add_block(text, [x1, y1, x2, y2], label)
        for b in blocks:
            self.delete_block(b.get("id", -1))
        return new_block["id"]

    def get_block(self, block_id: int) -> Optional[Dict]:
        for b in self.data.get("blocks", []):
            if b["id"] == block_id:
                return b
        return None

    # -------- Links --------
    def add_link(self, question_id: int, answer_id: int) -> None:
        q = self.get_block(question_id)
        a = self.get_block(answer_id)
        if not q or not a:
            return
        q_label = str(q.get("label", "other")).lower()
        a_label = str(a.get("label", "other")).lower()
        if q_label != "question" or a_label != "answer":
            return
        if any(
            l
            for l in self.data.get("links", [])
            if l["question_id"] == question_id and l["answer_id"] == answer_id
        ):
            return
        if "links" not in self.data:
            self.data["links"] = []
        self.data["links"].append({"question_id": question_id, "answer_id": answer_id})
        self.state_changed.emit()

    def remove_link(self, question_id: int, answer_id: int) -> None:
        self.data["links"] = [
            l
            for l in self.data.get("links", [])
            if not (l["question_id"] == question_id and l["answer_id"] == answer_id)
        ]
        self.state_changed.emit()

    def links_for_question(self, question_id: int) -> List[Dict]:
        return [l for l in self.data.get("links", []) if l["question_id"] == question_id]

    # -------- Tables --------
    def add_table_from_blocks(self, block_ids: List[int]) -> Optional[int]:
        blocks = [self.get_block(bid) for bid in block_ids]
        blocks = [b for b in blocks if b]
        if not blocks:
            return None

        # Row grouping by y-center proximity
        rows: List[List[Dict]] = []
        sorted_blocks = sorted(blocks, key=lambda b: (self._center(b.get("bbox", [0,0,0,0]))[1], b.get("bbox", [0,0,0,0])[0]))
        row_thresh = 25  # pixels
        for b in sorted_blocks:
            cx, cy = self._center(b.get("bbox", [0,0,0,0]))
            placed = False
            for row in rows:
                rcy = self._center(row[0].get("bbox", [0,0,0,0]))[1]
                if abs(cy - rcy) <= row_thresh:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])

        # Sort each row by x
        for row in rows:
            row.sort(key=lambda b: self._center(b.get("bbox", [0,0,0,0]))[0])

        table_id = self._table_id_counter
        self._table_id_counter += 1

        cells = []
        for r_idx, row in enumerate(rows):
            for c_idx, b in enumerate(row):
                cells.append({"row": r_idx, "col": c_idx, "block_id": b.get("id", -1)})

        if "tables" not in self.data:
            self.data["tables"] = []
        self.data["tables"].append({"table_id": table_id, "cells": cells})
        self.state_changed.emit()
        return table_id

    def move_block_to_cell(self, table_id: int, block_id: int, row: int, col: int):
        table = self.get_table(table_id)
        if not table:
            return
        table_cells = table.get("cells", [])
        table["cells"] = [c for c in table_cells if c.get("block_id") != block_id]
        table["cells"].append({"row": row, "col": col, "block_id": block_id})
        self.state_changed.emit()

    def add_row(self, table_id: int):
        table = self.get_table(table_id)
        if not table:
            return
        max_row = max((c.get("row", 0) for c in table.get("cells", [])), default=-1)
        # nothing else to do except signal
        for c in table.get("cells", []):
            if c.get("row", 0) > max_row:
                max_row = c.get("row", 0)
        new_row = max_row + 1
        table["cells"] = table.get("cells", [])
        table["cells"].extend([])  # placeholder to keep structure clear
        self.state_changed.emit()
        return new_row

    def add_col(self, table_id: int):
        table = self.get_table(table_id)
        if not table:
            return
        max_col = max((c.get("col", 0) for c in table.get("cells", [])), default=-1)
        new_col = max_col + 1
        table["cells"] = table.get("cells", [])
        table["cells"].extend([])  # no-op; structure is sparse
        self.state_changed.emit()
        return new_col

    def remove_table(self, table_id: int):
        self.data["tables"] = [t for t in self.data.get("tables", []) if t.get("table_id") != table_id]
        self.state_changed.emit()

    def get_table(self, table_id: int) -> Optional[Dict]:
        for t in self.data.get("tables", []):
            if t.get("table_id") == table_id:
                return t
        return None

    # -------- Selection --------
    def select_block(self, block_id: Optional[int]):
        self.selected_block_id = block_id
        self.selection_changed.emit(self.get_block(block_id) if block_id is not None else None)

    # -------- Persistence --------
    def load_json(self, data: Dict):
        # Convert FUNSD if present
        if "form" in data and "blocks" not in data:
            blocks = []
            links = []
            for item in data.get("form", []):
                blocks.append({
                    "id": item.get("id", len(blocks)),
                    "text": item.get("text", ""),
                    "bbox": item.get("box", [0, 0, 0, 0]),
                    "label": item.get("label", "other")
                })
                for link in item.get("linking", []):
                    links.append({
                        "question_id": link[0],
                        "answer_id": link[1]
                    })
            
            # Deduplicate links
            unique_links = []
            seen = set()
            for l in links:
                tup = (l["question_id"], l["answer_id"])
                if tup not in seen:
                    unique_links.append(l)
                    seen.add(tup)
            
            self.data = {
                "blocks": blocks,
                "links": unique_links,
                "tables": data.get("tables", [])
            }
        else:
            self.data = deepcopy(data)
            if "blocks" not in self.data:
                self.data["blocks"] = []
            if "links" not in self.data:
                self.data["links"] = []
            if "tables" not in self.data:
                self.data["tables"] = []

        max_block_id = max((b.get("id", -1) for b in self.data.get("blocks", [])), default=-1)
        max_table_id = max((t.get("table_id", -1) for t in self.data.get("tables", [])), default=-1)
        self._id_counter = max_block_id + 1
        self._table_id_counter = max_table_id + 1
        self.selected_block_id = None
        self.state_changed.emit()

    def to_json(self) -> Dict:
        return deepcopy(self.data)

    # -------- Helpers --------
    @staticmethod
    def _center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _vertical_overlap_ratio(a: List[float], b: List[float]) -> float:
        ay1, ay2 = a[1], a[3]
        by1, by2 = b[1], b[3]
        overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
        a_h = max(1.0, ay2 - ay1)
        b_h = max(1.0, by2 - by1)
        return overlap / min(a_h, b_h)

    @staticmethod
    def _join_text_fragments(parts: List[str]) -> str:
        cleaned = [p.strip() for p in parts if p and p.strip()]
        if not cleaned:
            return ""
        out = cleaned[0]
        for part in cleaned[1:]:
            if out.endswith("-"):
                out += part
            elif part[:1] in ",.;:!?)]":
                out += part
            else:
                out += " " + part
        return out

    def _merge_block_text_layout_aware(self, blocks: List[Dict]) -> str:
        """
        Merge selected blocks into coherent multi-line text:
        - Group blocks into lines by vertical proximity / overlap.
        - Join each line left->right with spaces.
        - Join lines top->bottom with newlines.
        """
        if not blocks:
            return ""

        heights = [
            max(1.0, b.get("bbox", [0, 0, 0, 0])[3] - b.get("bbox", [0, 0, 0, 0])[1])
            for b in blocks
        ]
        heights_sorted = sorted(heights)
        mid = len(heights_sorted) // 2
        median_h = (
            heights_sorted[mid]
            if len(heights_sorted) % 2 == 1
            else (heights_sorted[mid - 1] + heights_sorted[mid]) / 2.0
        )
        line_thresh = max(8.0, median_h * 0.6)

        sorted_blocks = sorted(
            blocks,
            key=lambda b: (
                self._center(b.get("bbox", [0, 0, 0, 0]))[1],
                b.get("bbox", [0, 0, 0, 0])[0],
            ),
        )

        lines: List[Dict] = []
        for block in sorted_blocks:
            bbox = block.get("bbox", [0, 0, 0, 0])
            cx, cy = self._center(bbox)
            chosen = None
            best_dist = None

            for line in lines:
                line_bbox = line["bbox"]
                line_cy = line["cy"]
                dist = abs(cy - line_cy)
                overlap = self._vertical_overlap_ratio(bbox, line_bbox)
                if dist <= line_thresh or overlap >= 0.4:
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        chosen = line

            if chosen is None:
                lines.append({
                    "blocks": [block],
                    "cy": cy,
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                })
            else:
                chosen["blocks"].append(block)
                chosen["cy"] = sum(
                    self._center(b.get("bbox", [0, 0, 0, 0]))[1]
                    for b in chosen["blocks"]
                ) / len(chosen["blocks"])
                lb = chosen["bbox"]
                chosen["bbox"] = [
                    min(lb[0], bbox[0]),
                    min(lb[1], bbox[1]),
                    max(lb[2], bbox[2]),
                    max(lb[3], bbox[3]),
                ]

        lines.sort(key=lambda l: (l["cy"], l["bbox"][0]))
        merged_lines: List[str] = []
        for line in lines:
            row_blocks = sorted(line["blocks"], key=lambda b: b.get("bbox", [0, 0, 0, 0])[0])
            row_text = self._join_text_fragments([b.get("text", "") for b in row_blocks])
            if row_text:
                merged_lines.append(row_text)

        return "\n".join(merged_lines)

    @staticmethod
    def _majority_label(blocks: List[Dict]) -> str:
        counts: Dict[str, int] = {}
        for b in blocks:
            label = str(b.get("label", "other")).lower()
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return "other"
        # Prefer non-"other" if tied
        best = sorted(counts.items(), key=lambda kv: (kv[1], kv[0] != "other"), reverse=True)
        return best[0][0]
