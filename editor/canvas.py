from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, Qt, QRectF, QTimer, QEvent
from PySide6.QtGui import QColor, QPen, QBrush, QPixmap, QFont, QTextCursor
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsTextItem,
    QWidget,
    QHBoxLayout,
    QPushButton,
)
from editor.commands import MergeBlocksCommand, DeleteBlockCommand

LABEL_COLORS = {
    "question": QColor("#3b82f6"),  # blue
    "answer": QColor("#22c55e"),    # green
    "header": QColor("#a855f7"),    # purple
    "other": QColor("#9ca3af"),     # gray
}


class ResizeHandle(QGraphicsRectItem):
    SIZE = 5

    def __init__(self, parent: "BlockItem", corner: str):
        super().__init__(-self.SIZE / 2, -self.SIZE / 2, self.SIZE, self.SIZE, parent)
        self.corner = corner
        self.setBrush(QBrush(QColor("white")))
        self.setPen(QPen(QColor("black")))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            parent: BlockItem = self.parentItem()
            if parent:
                parent.resize_from_handle(self)
        return super().itemChange(change, value)


class EditableTextItem(QGraphicsTextItem):
    """Text item that commits edits when focus leaves the item."""
    def __init__(self, text: str, parent: "BlockItem"):
        super().__init__(text, parent)
        self._owner = parent
        self.setTextInteractionFlags(Qt.NoTextInteraction)

    def focusOutEvent(self, event):
        self._owner.commit_text_edit()
        super().focusOutEvent(event)


class BlockItem(QGraphicsRectItem):
    def __init__(self, block: dict, state, command_manager, editable: bool = False, show_bboxes: bool = True):
        bbox = block.get("bbox", [0, 0, 100, 50])
        x1, y1, x2, y2 = bbox
        super().__init__(x1, y1, x2 - x1, y2 - y1)
        self.block_id = block.get("id", -1)
        self.state = state
        self.cmd = command_manager
        self.editable = editable
        self._is_updating = False
        self._last_bbox = block.get("bbox", [0, 0, 100, 50])
        self._show_bboxes = show_bboxes
        self._last_text = block.get("text", "")
        flags = QGraphicsItem.ItemSendsGeometryChanges
        if self.editable:
            flags |= QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable
        self.setFlags(flags)
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.handles = {
            "tl": ResizeHandle(self, "tl"),
            "tr": ResizeHandle(self, "tr"),
            "bl": ResizeHandle(self, "bl"),
            "br": ResizeHandle(self, "br"),
        }
        self.text_item = EditableTextItem(block.get("text", ""), self)
        self._requested_font_size = block.get("font_size")
        font = QFont()
        # Use a high-level requested size, then auto-fit down to bbox.
        font.setPointSizeF(float(self._requested_font_size) if self._requested_font_size else 11.0)
        self.text_item.setFont(font)
        self.text_item.document().setDocumentMargin(0.0)
        # Wrap text to fit the block width from the start (also set per resize).
        self.text_item.setTextWidth(max(x2 - x1 - 2, 1))
        # Show text in edit mode (black on white canvas); hide in view mode.
        self.text_item.setDefaultTextColor(QColor("#111111"))
        self.text_item.setVisible(self.editable)

        self.update_handle_positions()
        self._fit_text_to_bbox()
        self.update_style(block.get("label", "other"))

    # ---- visuals ----
    def update_style(self, label: str):
        # Use the flag set at construction time (updated by CanvasView.refresh)
        show_bboxes = self._show_bboxes

        color = LABEL_COLORS.get(label, QColor("black"))
        
        if show_bboxes:
            self.setPen(QPen(color, 2))
            for h in self.handles.values():
                h.setVisible(self.editable)
        else:
            self.setPen(QPen(Qt.NoPen))
            for h in self.handles.values():
                h.hide()

        if hasattr(self, "text_item"):
            # Always black text – readable on the white canvas in edit mode.
            self.text_item.setDefaultTextColor(QColor("#111111"))

    def update_handle_positions(self):
        if hasattr(self, "_is_updating") and self._is_updating:
            return
        self._is_updating = True
        rect = self.rect()
        self.handles["tl"].setPos(rect.topLeft())
        self.handles["tr"].setPos(rect.topRight())
        self.handles["bl"].setPos(rect.bottomLeft())
        self.handles["br"].setPos(rect.bottomRight())
        if hasattr(self, "text_item"):
            self.text_item.setPos(rect.topLeft() + QPointF(1.0, 0.0))
        self._is_updating = False

    # ---- interactions ----
    def resize_from_handle(self, handle: ResizeHandle):
        if not self.editable:
            return
        if hasattr(self, "_is_updating") and self._is_updating:
            return
        self._is_updating = True
        rect = QRectF(self.rect())
        pos = handle.pos()
        if handle.corner == "tl":
            rect.setTopLeft(pos)
        elif handle.corner == "tr":
            rect.setTopRight(pos)
        elif handle.corner == "bl":
            rect.setBottomLeft(pos)
        elif handle.corner == "br":
            rect.setBottomRight(pos)
        # Prevent negative sizes
        if rect.width() < 5 or rect.height() < 5:
            self._is_updating = False
            return
        self.setRect(rect)
        # Manually update other handles
        self.handles["tl"].setPos(rect.topLeft())
        self.handles["tr"].setPos(rect.topRight())
        self.handles["bl"].setPos(rect.bottomLeft())
        self.handles["br"].setPos(rect.bottomRight())
        if hasattr(self, "text_item"):
            self.text_item.setPos(rect.topLeft() + QPointF(1.0, 0.0))
            self.text_item.setTextWidth(max(rect.width() - 2, 1))
            self._fit_text_to_bbox()
        self._is_updating = False

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            # move handles along with item; bbox updated on mouse release
            self.update_handle_positions()
        if change == QGraphicsItem.ItemSelectedHasChanged and bool(value):
            # Only sync single selection to state if we are the only one selected
            # This prevents lasso selection from triggering state changes per-item
            scene = self.scene()
            if scene and len(scene.selectedItems()) <= 1:
                self.state.select_block(self.block_id)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        if not self.editable:
            super().mouseReleaseEvent(event)
            return
        rect = self.mapRectToScene(self.rect())
        bbox = [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]
        
        # Only issue an update if the block actually moved or resized
        # This prevents normal clicks or lasso drags from spamming state changes
        if [round(v, 2) for v in bbox] != [round(v, 2) for v in self._last_bbox]:
            self._last_bbox = bbox
            from editor.commands import UpdateBlockCommand
            
            self.cmd.push(
                UpdateBlockCommand(
                    self.state,
                    self.block_id,
                    {"bbox": bbox},
                )
            )
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.editable:
            self.text_item.setVisible(True)
            self.text_item.setTextInteractionFlags(Qt.TextEditorInteraction)
            self.text_item.setFocus()
            cursor = self.text_item.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_item.setTextCursor(cursor)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def commit_text_edit(self):
        text_now = self.text_item.toPlainText()
        self.text_item.setTextInteractionFlags(Qt.NoTextInteraction)
        self.text_item.setVisible(self.editable)  # keep visible in edit mode
        self._fit_text_to_bbox()
        if not self.editable:
            self._last_text = text_now
            return
        if text_now != self._last_text:
            self._last_text = text_now
            from editor.commands import UpdateBlockCommand
            self.cmd.push(
                UpdateBlockCommand(
                    self.state,
                    self.block_id,
                    {"text": text_now},
                )
            )

    def _fit_text_to_bbox(self):
        """
        Auto-fit editable text to stay inside bbox and avoid severe overlap.
        """
        if not self.editable or not hasattr(self, "text_item"):
            return
        rect = self.rect()
        avail_w = max(6.0, rect.width() - 2.0)
        avail_h = max(6.0, rect.height() - 1.0)
        self.text_item.setTextWidth(avail_w)

        base = float(self._requested_font_size) if self._requested_font_size else min(11.0, avail_h * 0.9)
        max_size = max(5.0, min(base, 18.0))
        min_size = 4.0
        size = max_size

        font = self.text_item.font()
        while size >= min_size:
            font.setPointSizeF(size)
            self.text_item.setFont(font)
            doc_h = self.text_item.document().size().height()
            if doc_h <= avail_h + 0.5:
                break
            size -= 0.5

        if size < min_size:
            font.setPointSizeF(min_size)
            self.text_item.setFont(font)


class SelectionOverlay(QWidget):
    """Floating controls for multi-selection actions."""

    def __init__(self, parent, on_merge, on_delete):
        super().__init__(parent)
        self.on_merge = on_merge
        self.on_delete = on_delete
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
            }
            QPushButton {
                border: 1px solid #334155;
                background: #1e293b;
                color: #cbd5e1;
                border-radius: 8px;
                padding: 6px 10px;
                font-weight: 600;
                min-width: 70px;
            }
            QPushButton:hover { background: #334155; border-color: #3b82f6; color: #f1f5f9; }
            QPushButton#danger { background: #450a0a; border-color: #7f1d1d; color: #fca5a5; }
            QPushButton#danger:hover { background: #7f1d1d; border-color: #dc2626; }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        merge_btn = QPushButton("Merge")
        merge_btn.clicked.connect(self.on_merge)
        layout.addWidget(merge_btn)

        del_btn = QPushButton("Delete")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(self.on_delete)
        layout.addWidget(del_btn)


class TableOverlay(QGraphicsRectItem):
    def __init__(self, rect: QRectF, rows: int, cols: int):
        super().__init__(rect)
        pen = QPen(QColor("#94a3b8"))
        pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        self.setBrush(Qt.NoBrush)
        self.setZValue(-1)
        # add grid lines as children
        if rows > 1:
            h_step = rect.height() / rows
            for r in range(1, rows):
                y = rect.top() + r * h_step
                line = QGraphicsRectItem(rect.left(), y, rect.width(), 0.5, self)
                line.setPen(pen)
        if cols > 1:
            w_step = rect.width() / cols
            for c in range(1, cols):
                x = rect.left() + c * w_step
                line = QGraphicsRectItem(x, rect.top(), 0.5, rect.height(), self)
                line.setPen(pen)


class CanvasView(QGraphicsView):
    _ZOOM_FACTOR = 1.15   # per scroll tick

    def __init__(self, state, command_manager):
        super().__init__()
        self.state = state
        self.cmd = command_manager
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setBackgroundBrush(QBrush(QColor("white")))
        # Force the viewport widget background to white regardless of any app stylesheet.
        from PySide6.QtGui import QPalette
        pal = self.viewport().palette()
        pal.setColor(QPalette.Window, QColor("white"))
        pal.setColor(QPalette.WindowText, QColor("#111111"))
        self.viewport().setPalette(pal)
        self.viewport().setAutoFillBackground(True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._background_pixmap: Optional[QPixmap] = None
        self._refresh_pending = False
        self._zoom_level = 0   # cumulative scroll ticks
        self._user_zoomed = False
        self.edit_mode = False
        self._overlay = SelectionOverlay(self.viewport(), self._merge_selection, self._delete_selection)
        self._overlay.hide()

        state.state_changed.connect(self._schedule_refresh)
        state.selection_changed.connect(self.highlight)
        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.show_bboxes = True
        self.show_links = True
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def set_background_image(self, image_path: str):
        pix = QPixmap(image_path)
        if pix.isNull():
            self._background_pixmap = None
            self.refresh()
            return
        self._background_pixmap = pix
        self._user_zoomed = False
        self.refresh()

    def clear_background_image(self):
        self._background_pixmap = None
        self._user_zoomed = False
        self.refresh()

    def set_edit_mode(self, enabled: bool):
        self.edit_mode = enabled
        self.setDragMode(QGraphicsView.RubberBandDrag if enabled else QGraphicsView.ScrollHandDrag)
        if not self._user_zoomed:
            self.fit_width(force=True)
        self.refresh()
        if not enabled:
            self._overlay.hide()

    def toggle_bboxes(self):
        self.show_bboxes = not self.show_bboxes
        # Tie graph-link visibility to the same toggle so "off" removes graph lines too.
        self.show_links = self.show_bboxes
        self.refresh()

    # ---- scene refresh ----
    def refresh(self):
        self.scene.clear()
        if self._background_pixmap is not None:
            bg = QGraphicsPixmapItem(self._background_pixmap)
            bg.setZValue(-1000)
            self.scene.addItem(bg)
            self.scene.setSceneRect(self._background_pixmap.rect())
        for block in self.state.data.get("blocks", []):
            item = BlockItem(block, self.state, self.cmd, editable=self.edit_mode, show_bboxes=self.show_bboxes)
            self.scene.addItem(item)
        self._draw_links()
        self._draw_tables()
        self._refresh_pending = False
        if not self._user_zoomed:
            self.fit_width(force=False)

    def _schedule_refresh(self):
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self.refresh)

    def _draw_tables(self):
        for table in self.state.data.get("tables", []):
            cells = table.get("cells", [])
            if not cells:
                continue
            blocks = [self.state.get_block(c["block_id"]) for c in cells]
            blocks = [b for b in blocks if b]
            if not blocks:
                continue
            x1 = min(b["bbox"][0] for b in blocks)
            y1 = min(b["bbox"][1] for b in blocks)
            x2 = max(b["bbox"][2] for b in blocks)
            y2 = max(b["bbox"][3] for b in blocks)
            rows = max((c.get("row", 0) for c in cells), default=-1) + 1
            cols = max((c.get("col", 0) for c in cells), default=-1) + 1
            overlay = TableOverlay(QRectF(x1, y1, x2 - x1, y2 - y1), rows, cols)
            self.scene.addItem(overlay)

    def _draw_links(self):
        """Render question->answer links as light orange lines."""
        if not self.show_links:
            return
        links = self.state.data.get("links", [])
        if not links:
            return

        block_by_id = {b.get("id"): b for b in self.state.data.get("blocks", [])}
        pen = QPen(QColor("#f59e0b"), 1.5)
        pen.setStyle(Qt.DashLine)

        for link in links:
            qid = link.get("question_id")
            aid = link.get("answer_id")
            q = block_by_id.get(qid)
            a = block_by_id.get(aid)
            if not q or not a:
                continue
            qb = q.get("bbox", [0, 0, 0, 0])
            ab = a.get("bbox", [0, 0, 0, 0])
            qcx, qcy = (qb[0] + qb[2]) / 2.0, (qb[1] + qb[3]) / 2.0
            acx, acy = (ab[0] + ab[2]) / 2.0, (ab[1] + ab[3]) / 2.0

            line = QGraphicsLineItem(qcx, qcy, acx, acy)
            line.setPen(pen)
            line.setZValue(-10)
            self.scene.addItem(line)

    def highlight(self, block):
        # Only select implicitly if a specific block was targeted by the inspector/state
        # Do not override manual lasso selections with clear commands
        if block is None:
            return
        # If the state targeted a selection, make sure the UI reflects it natively
        for item in self.scene.items():
            if isinstance(item, BlockItem) and item.block_id == block.get("id", -1):
                item.setSelected(True)

    # ---- zoom ----
    def _apply_zoom_step(self, step: float):
        if step == 0:
            return
        new_level = max(-10.0, min(20.0, self._zoom_level + step))
        applied = new_level - self._zoom_level
        if applied == 0:
            return
        factor = self._ZOOM_FACTOR ** applied
        self.scale(factor, factor)
        self._zoom_level = new_level
        self._user_zoomed = True
        self._update_overlay_position()

    def wheelEvent(self, event):
        """Ctrl+scroll → zoom in/out. Plain scroll → pan (default)."""
        zoom_mods = Qt.ControlModifier | Qt.MetaModifier
        if event.modifiers() & zoom_mods:
            delta = event.angleDelta().y()
            if delta != 0:
                self._apply_zoom_step(1.0 if delta > 0 else -1.0)
                event.accept()
                return
        else:
            super().wheelEvent(event)

    def event(self, event):
        # Trackpad pinch on macOS/Qt arrives as NativeGesture, not wheel.
        if event.type() == QEvent.Type.NativeGesture:
            zoom_type = getattr(Qt.NativeGestureType, "ZoomNativeGesture", None)
            gesture_type = getattr(event, "gestureType", lambda: None)()
            if zoom_type is not None and gesture_type == zoom_type:
                value = float(getattr(event, "value", lambda: 0.0)())
                self._apply_zoom_step(value * 6.0)
                event.accept()
                return True
        return super().event(event)

    def reset_zoom(self):
        """Fit the entire scene into view."""
        self.fit_width(force=True)

    def fit_width(self, force: bool = False):
        """Scale so the scene width fits the viewport while keeping aspect."""
        if not force and self._user_zoomed:
            return
        rect = self.scene.sceneRect()
        if rect.isNull() or rect.width() <= 0:
            return
        view_w = max(1.0, float(self.viewport().width()))
        scale_factor = view_w / rect.width()
        self.resetTransform()
        self.scale(scale_factor, scale_factor)
        self._zoom_level = 0
        if force:
            self._user_zoomed = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit_width(force=False)
        self._update_overlay_position()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._update_overlay_position()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._update_overlay_position()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._update_overlay_position()

    # ---- overlay helpers ----
    def _on_selection_changed(self):
        if not self.edit_mode:
            self._overlay.hide()
            return
        items = [i for i in self.scene.selectedItems() if isinstance(i, BlockItem)]
        if len(items) < 2:
            self._overlay.hide()
            return
        self._overlay.adjustSize()
        self._overlay.show()
        self._overlay.raise_()
        self._update_overlay_position()

    def _combined_rect(self, items):
        xs1, ys1, xs2, ys2 = [], [], [], []
        for item in items:
            r = item.sceneBoundingRect()
            xs1.append(r.left())
            ys1.append(r.top())
            xs2.append(r.right())
            ys2.append(r.bottom())
        return QRectF(min(xs1), min(ys1), max(xs2) - min(xs1), max(ys2) - min(ys1))

    def _update_overlay_position(self):
        if not self._overlay.isVisible():
            return
        items = [i for i in self.scene.selectedItems() if isinstance(i, BlockItem)]
        if len(items) < 2:
            self._overlay.hide()
            return
        rect = self._combined_rect(items)
        poly = self.mapFromScene(rect)
        vr = poly.boundingRect()
        x = vr.center().x() - self._overlay.width() / 2.0
        y = vr.top() - self._overlay.height() - 8
        if y < 4:
            y = vr.bottom() + 8
        x = max(4.0, min(x, self.viewport().width() - self._overlay.width() - 4.0))
        y = max(4.0, min(y, self.viewport().height() - self._overlay.height() - 4.0))
        self._overlay.move(int(x), int(y))

    def _merge_selection(self):
        ids = [i.block_id for i in self.scene.selectedItems() if isinstance(i, BlockItem)]
        if len(ids) >= 2:
            self.cmd.push(MergeBlocksCommand(self.state, ids))
        self._overlay.hide()

    def _delete_selection(self):
        ids = [i.block_id for i in self.scene.selectedItems() if isinstance(i, BlockItem)]
        if not ids:
            return
        for bid in ids:
            self.cmd.push(DeleteBlockCommand(self.state, bid))
        self._overlay.hide()
