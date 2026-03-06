from __future__ import annotations

import json
import sys

from PySide6.QtCore import Qt, QEvent, QSettings
from PySide6.QtGui import QAction, QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QToolBar,
    QLabel,
    QComboBox,
    QSplitter,
    QPushButton,
    QDialog,
    QVBoxLayout,
    QDialogButtonBox,
    QFrame,
    QTextBrowser,
    QStatusBar,
    QCheckBox,
    QProgressDialog,
    QSpinBox,
    QSlider,
)
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtGui import QWheelEvent

from editor.state import DocumentState
from editor.canvas import CanvasView
from editor.inspector import InspectorPanel
from editor.commands import (
    CommandManager,
    SplitBlockCommand,
    MergeBlocksCommand,
    DeleteBlockCommand,
)

# ---------------------------------------------------------------------------
# Global stylesheet – dark, premium look
# ---------------------------------------------------------------------------
APP_STYLESHEET = """
/* Base */
QMainWindow, QWidget {
    background-color: #0f172a;
    color: #e2e8f0;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}
QGraphicsView {
    background-color: #ffffff;
    color: #0f172a;
    border: none;
}

/* Toolbar */
QToolBar {
    background-color: #1e293b;
    border-bottom: 1px solid #334155;
    padding: 5px 6px;
    spacing: 3px;
}
QToolBar QLabel {
    color: #64748b;
    font-size: 11px;
    padding: 0 4px;
}
QToolBar::separator {
    background-color: #334155;
    width: 1px;
    margin: 5px 6px;
}
QToolBar QPushButton {
    background-color: #1e3a5f;
    color: #bfdbfe;
    border: 1px solid #2d5494;
    border-radius: 6px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 500;
    min-height: 28px;
    min-width: 64px;
}
QToolBar QPushButton:hover {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
}
QToolBar QPushButton:pressed {
    background-color: #1e3a5f;
    border-color: #3b82f6;
}
QToolBar QPushButton:checked {
    background-color: #1d4ed8;
    border-color: #3b82f6;
    color: #ffffff;
}
QToolBar QPushButton:disabled {
    background-color: #1e293b;
    color: #334155;
    border-color: #233046;
}
QToolBar QPushButton#editBtn {
    background-color: #1e40af;
    color: #93c5fd;
    border-color: #3b82f6;
}
QToolBar QPushButton#editBtn:checked {
    background-color: #1d4ed8;
    border-color: #3b82f6;
    color: #ffffff;
}
QToolBar QPushButton#deleteBtn {
    background-color: #3b0808;
    color: #fca5a5;
    border-color: #7f1d1d;
}
QToolBar QPushButton#deleteBtn:hover {
    background-color: #450a0a;
    border-color: #dc2626;
    color: #fca5a5;
}

/* ComboBox */
QComboBox {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 5px 10px;
    min-height: 26px;
    min-width: 110px;
}
QComboBox:hover { border-color: #3b82f6; }
QComboBox::drop-down { border: 0; padding-right: 5px; }
QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #e2e8f0;
    selection-background-color: #1d4ed8;
    border: 1px solid #334155;
    outline: none;
}

/* Help button */
QPushButton#helpBtn {
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #60a5fa;
    min-width: 28px;
    min-height: 28px;
    padding: 0;
    font-size: 14px;
    font-weight: bold;
    border-radius: 14px;
}
QPushButton#helpBtn:hover {
    background-color: #1d4ed8;
    border-color: #3b82f6;
    color: #ffffff;
}

/* Status bar */
QStatusBar {
    background-color: #0f172a;
    border-top: 1px solid #1e293b;
    color: #94a3b8;
    font-size: 11px;
}
QStatusBar QPushButton {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 11px;
    min-height: 20px;
    margin: 2px;
}
QStatusBar QPushButton:hover {
    background-color: #334155;
    color: #f1f5f9;
}

/* Splitter */
QSplitter::handle { background-color: #1e293b; width: 2px; }
QSplitter::handle:hover { background-color: #3b82f6; }
"""


# ---------------------------------------------------------------------------
# Welcome / help dialog
# ---------------------------------------------------------------------------
class WelcomeDialog(QDialog):
    """Full onboarding dialog shown on first launch (or via ? button)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to the Document Editor")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                color: #0f172a;
            }
            QTextBrowser {
                background-color: #f8fafc;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
            }
            QDialogButtonBox QPushButton {
                background-color: #dbeafe;
                color: #0f172a;
                border: 1px solid #bfdbfe;
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: 600;
                min-width: 100px;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #cbdcf8;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("📄  Document Editor — Quick Start Guide")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #60a5fa; padding-bottom: 4px;")
        layout.addWidget(title)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml("""
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; color: #e2e8f0; }
  h3 { color: #60a5fa; margin-top: 14px; margin-bottom: 4px; }
  p, li { margin: 4px 0; line-height: 1.55; }
  .key { background: #0f3460; border: 1px solid #1e4a8a; border-radius: 4px;
         padding: 1px 6px; font-family: monospace; color: #93c5fd; }
  .tag { border-radius: 4px; padding: 1px 8px; font-size: 12px; font-weight: bold; }
  .q { background: #1d4ed8; color: #bfdbfe; }
  .a { background: #15803d; color: #bbf7d0; }
  .h { background: #7e22ce; color: #e9d5ff; }
  .o { background: #374151; color: #d1d5db; }
</style>

<h3>🚀 Step 1 — Load a Document Image</h3>
<p>Click <b> Load Image</b> in the toolbar to open a scanned form or document.<br>
The system will automatically run OCR and the LayoutLM model on it.</p>

<h3> Step 2 — Navigate the Canvas</h3>
<ul>
  <li><span class="key">Ctrl + Scroll</span> or <b>Pinch</b> to zoom in / out.</li>
  <li><b>Scroll</b> or <b>drag</b> (in view mode) to pan.</li>
  <li>Coloured bounding boxes show detected text regions:</li>
</ul>
<p>
  &nbsp;&nbsp;<span class="tag q">QUESTION</span>&nbsp;
  <span class="tag a">ANSWER</span>&nbsp;
  <span class="tag h">HEADER</span>&nbsp;
  <span class="tag o">OTHER</span>
</p>

<h3> Step 3 — Edit Mode</h3>
<p>Toggle <b> Edit Mode</b> to switch to editing. In this mode:</p>
<ul>
  <li><b>Click</b> a box to select it. The right-hand Inspector will show its details.</li>
  <li><b>Drag</b> a box to move it, or drag its corner handles to resize it.</li>
  <li><b>Double-click</b> a box to edit its text inline.</li>
</ul>

<h3> Split — Divide One Block Into Two</h3>
<p>Select a single block and click <b> Split</b>. The block will be divided into
two halves. Useful when OCR merged two separate fields into one.</p>

<h3> Merge — Combine Multiple Blocks Into One</h3>
<p>Select <b>2 or more</b> blocks by clicking them (hold <span class="key">Shift</span>
or lasso-drag). Then click <b> Merge</b>. The selected blocks are merged
into a single bounding box, with their text concatenated.</p>
<p>⚠️ Merge is irreversible in the current session unless you use <b>Undo</b>
(<span class="key">Ctrl+Z</span>).</p>

<h3> Saving Your Work</h3>
<p>Click <b>Save JSON</b> to export your annotations to a JSON file that can be
loaded later with <b> Load JSON</b>.</p>

<h3>🔄 Switching OCR Engine or Model</h3>
<p>Use the <b>OCR Engine</b> and <b>Model</b> dropdowns to re-run extraction with
a different backend. The canvas will refresh automatically.</p>
""")
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("Got it — Let's start!")
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class EditModeHint(QDialog):
    """Small helper shown when entering Edit Mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Mode Tips")
        self.setModal(True)
        self.setMinimumWidth(380)
        self.setStyleSheet("""
            QDialog { background-color: #0f172a; }
            QLabel { color: #e2e8f0; font-size: 13px; }
            QLabel#hdr { color: #f1f5f9; font-size: 15px; font-weight: 700; }
            QCheckBox { color: #94a3b8; font-size: 12px; }
            QCheckBox::indicator {
                border: 1px solid #334155; background: #1e293b;
                border-radius: 3px; width: 14px; height: 14px;
            }
            QCheckBox::indicator:checked { background: #1d4ed8; border-color: #3b82f6; }
            QDialogButtonBox QPushButton {
                background-color: #1d4ed8; color: #ffffff;
                border: 1px solid #3b82f6; border-radius: 6px;
                padding: 6px 24px; font-size: 13px; font-weight: 600; min-width: 72px;
            }
            QDialogButtonBox QPushButton:hover { background-color: #2563eb; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 16)
        layout.setSpacing(12)

        hdr = QLabel("Edit Mode")
        hdr.setObjectName("hdr")
        layout.addWidget(hdr)

        text = QLabel(
            "OCR is prone to mistakes on noisy or handwritten data — use Edit Mode to correct them.\n\n"
            "\u2022 Double-tap a box to edit its text directly.\n"
            "\u2022 Select individual boxes and use Merge to join tokens that belong together.\n"
            "\u2022 Drag box edges to reposition or resize over the correct region.\n"
            "\u2022 Use the toolbar to add new boxes if any text is missing from the annotation."
        )
        text.setWordWrap(True)
        layout.addWidget(text)

        layout.addSpacing(4)
        self._dont_show = QCheckBox("Don't show this again")
        layout.addWidget(self._dont_show)

        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)

    def dont_show_again(self) -> bool:
        return self._dont_show.isChecked()


class SplitDialog(QDialog):
    """Choose the word index where a block's text is split into left/right boxes."""

    def __init__(self, words: list[str], parent=None):
        super().__init__(parent)
        self._words = words
        self.setWindowTitle("Split Block")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setStyleSheet("""
            QDialog { background-color: #0f172a; color: #e2e8f0; }
            QLabel { color: #e2e8f0; font-size: 12px; }
            QLabel#hdr { color: #f1f5f9; font-size: 14px; font-weight: 700; }
            QSlider::groove:horizontal {
                border: 1px solid #334155;
                height: 8px;
                background: #1e293b;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                border: 1px solid #60a5fa;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #1d4ed8;
                border: 1px solid #1d4ed8;
                height: 8px;
                border-radius: 4px;
            }
            QSpinBox {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 28px;
            }
            QSpinBox:focus { border-color: #3b82f6; }
            QLabel#preview {
                background: #0b1220;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
                color: #cbd5e1;
            }
            QDialogButtonBox QPushButton {
                background-color: #1d4ed8; color: #ffffff;
                border: 1px solid #3b82f6; border-radius: 6px;
                padding: 6px 22px; font-size: 13px; font-weight: 600; min-width: 90px;
            }
            QDialogButtonBox QPushButton:hover { background-color: #2563eb; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(10)

        hdr = QLabel("Split selected block")
        hdr.setObjectName("hdr")
        layout.addWidget(hdr)

        info = QLabel(
            "Split is left/right. Choose how many words go into the left box.\n"
            "The remaining words go into the right box."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        max_split = max(1, len(words) - 1)
        default_split = max(1, len(words) // 2)

        layout.addWidget(QLabel("Split after N words:"))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(1, max_split)
        self._slider.setValue(default_split)
        self._slider.setToolTip("Drag to choose split point.")
        layout.addWidget(self._slider)

        # Keep a spinbox for precise adjustment, synced with the slider.
        self._spin = QSpinBox()
        self._spin.setRange(1, max_split)
        self._spin.setValue(default_split)
        self._spin.setToolTip("Exact split point (word index).")
        layout.addWidget(self._spin)

        layout.addWidget(QLabel("Left preview:"))
        self._left = QLabel("")
        self._left.setObjectName("preview")
        self._left.setWordWrap(True)
        layout.addWidget(self._left)

        layout.addWidget(QLabel("Right preview:"))
        self._right = QLabel("")
        self._right.setObjectName("preview")
        self._right.setWordWrap(True)
        layout.addWidget(self._right)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._slider.setValue)
        self._spin.valueChanged.connect(self._update_preview)
        self._update_preview()

    def split_index(self) -> int | None:
        if len(self._words) < 2:
            return None
        v = int(self._spin.value())
        return max(1, min(v, len(self._words) - 1))

    def _update_preview(self):
        idx = self.split_index()
        if idx is None:
            self._left.setText("")
            self._right.setText("")
            return
        self._left.setText(" ".join(self._words[:idx]))
        self._right.setText(" ".join(self._words[idx:]))


class ZoomableImageView(QGraphicsView):
    """Lightweight zoomable image viewer. Ctrl+scroll to zoom, plain scroll to pan."""
    _ZOOM_FACTOR = 1.15

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom_level = 0
        self._user_zoomed = False
        self._pixmap_item = None

    def set_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self.fit_width(force=True)

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

    def wheelEvent(self, event: QWheelEvent):
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

    def fit_width(self, force: bool = False):
        if not force and self._user_zoomed:
            return
        rect = self._scene.sceneRect()
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


# ---------------------------------------------------------------------------
# Helper: styled toolbar button
# ---------------------------------------------------------------------------
def _toolbar_btn(label: str, tooltip: str, object_name: str = "") -> QPushButton:
    btn = QPushButton(label)
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.PointingHandCursor)
    if object_name:
        btn.setObjectName(object_name)
    return btn


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📄 Document Editor")
        self.resize(1300, 850)

        self.state = DocumentState()
        self.commands = CommandManager()
        self.settings = QSettings("formGeneration", "DocumentEditor")
        self.show_edit_hint = self.settings.value("show_edit_hint", True, type=bool)

        central = QWidget()

        self.canvas = CanvasView(self.state, self.commands)
        self.inspector = InspectorPanel(self.state, self.commands)
        self.current_image_path = None
        self.edit_mode = False
        self._edit_layout_applied = False

        # Zoomable side-by-side image viewer (Ctrl+scroll to zoom, scroll to pan)
        self.image_view = ZoomableImageView()
        self.image_view.hide()

        # Draggable splitters for left/right pane widths.
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.image_view)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.inspector)
        self.splitter.setChildrenCollapsible(False)
        # Keep the two main panes at parity; inspector remains narrower.
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([500, 500, 280])

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self.setCentralWidget(central)

        self._build_toolbar()
        self._build_statusbar()
        self.state.state_changed.connect(self._on_state_changed)

        # Show a small "Get Started" popup on every launch
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._show_get_started)

    # ------------------------------------------------------------------ get started
    def _show_get_started(self):
        """Small always-on-top dialog pointing users to Load Image / Load JSON."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Get Started")
        dlg.setFixedWidth(320)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowStaysOnTopHint)
        dlg.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                border: 1px solid #1e293b;
            }
            QLabel {
                color: #e2e8f0;
            }
            QLabel#sub {
                color: #64748b;
                font-size: 11px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 9px 14px;
                font-size: 13px;
                font-weight: 500;
                min-height: 34px;
            }
            QPushButton#primary {
                background-color: #1d4ed8;
                border-color: #3b82f6;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#primary:hover { background-color: #2563eb; }
            QPushButton:hover { background-color: #334155; color: #f1f5f9; }
            QPushButton#skip {
                background-color: transparent;
                border: none;
                color: #475569;
                font-size: 11px;
                min-height: 20px;
                padding: 2px 8px;
            }
            QPushButton#skip:hover { color: #64748b; }
        """)

        vlay = QVBoxLayout(dlg)
        vlay.setContentsMargins(22, 20, 22, 14)
        vlay.setSpacing(10)

        title = QLabel("Document Editor")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f1f5f9;")
        vlay.addWidget(title)

        sub = QLabel("Load a scanned form to start annotating.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        vlay.addWidget(sub)

        vlay.addSpacing(4)

        img_btn = QPushButton("Load Image")
        img_btn.setObjectName("primary")
        img_btn.setToolTip("Open a scanned document image — OCR + LayoutLM run automatically.")
        img_btn.clicked.connect(dlg.accept)
        img_btn.clicked.connect(self._load_image_and_extract)
        vlay.addWidget(img_btn)

        json_btn = QPushButton("Load JSON")
        json_btn.setToolTip("Resume from a previously saved annotation JSON file.")
        json_btn.clicked.connect(dlg.accept)
        json_btn.clicked.connect(self._load_json)
        vlay.addWidget(json_btn)

        skip_btn = QPushButton("Skip")
        skip_btn.setObjectName("skip")
        skip_btn.clicked.connect(dlg.reject)
        vlay.addWidget(skip_btn, alignment=Qt.AlignRight)

        dlg.exec()

    # ------------------------------------------------------------------ welcome
    def _show_welcome(self):
        dlg = WelcomeDialog(self)
        dlg.exec()

    # ------------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # ── File & extraction ────────────────────────────────────────────────
        load_img_btn = _toolbar_btn(
            "Load Image",
            "Open a scanned document image.\nOCR + LayoutLM will run automatically.",
            "loadImageBtn",
        )
        load_img_btn.clicked.connect(self._load_image_and_extract)

        load_btn = _toolbar_btn("Load JSON", "Load previously saved annotation JSON.", "loadJsonBtn")
        load_btn.clicked.connect(self._load_json)

        save_btn = _toolbar_btn("Save JSON", "Save current annotations to a JSON file.", "saveJsonBtn")
        save_btn.clicked.connect(self._save_json)

        # ── Edit mode toggle ──────────────────────────────────────────────────
        self.edit_btn = _toolbar_btn(
            "Edit Mode",
            "Toggle Edit Mode.\n"
            "In Edit Mode: click blocks to select, drag to move/resize,\n"
            "double-click to edit text inline.\n"
            "In View Mode: pan and zoom freely.",
            "editBtn",
        )
        self.edit_btn.setCheckable(True)
        self.edit_btn.toggled.connect(self._set_edit_mode)

        # ── Block editing operations ──────────────────────────────────────────
        split_btn = _toolbar_btn(
            "Split",
            "Split the selected block into two halves.\n"
            "Select exactly one block first (Enable Edit Mode).",
            "splitBtn",
        )
        split_btn.clicked.connect(self._split_selected)

        merge_btn = _toolbar_btn(
            "Merge",
            "Merge 2+ selected blocks into one.\n"
            "Shift+Click or lasso-drag to select multiple,\n"
            "then click Merge. Undo with Ctrl+Z.\n\n"
            "Click Info next to the ? button for a detailed guide.",
            "mergeBtn",
        )
        merge_btn.clicked.connect(self._merge_selected)

        delete_btn = _toolbar_btn("Delete", "Delete selected block(s). Undo with Ctrl+Z.", "deleteBtn")
        delete_btn.clicked.connect(self._delete_selected)

        # ── View toggle ───────────────────────────────────────────────────────
        toggle_bbox_btn = _toolbar_btn("Boxes", "Show / hide bounding box overlays.", "boxesBtn")
        toggle_bbox_btn.clicked.connect(self.canvas.toggle_bboxes)

        # ── OCR / model selectors ─────────────────────────────────────────────
        self.ocr_combo = QComboBox(self)
        # UI intentionally limited to docTR only.
        # (Removes paddle/tesseract from selection to match available models.)
        self.ocr_combo.addItems(["doctr"])
        self.ocr_combo.setToolTip("OCR engine is fixed to docTR in this editor build.")
        # Set default silently (model_combo doesn't exist yet, so block the signal)
        self.ocr_combo.blockSignals(True)
        self.ocr_combo.setCurrentText("doctr")
        self.ocr_combo.blockSignals(False)
        self.ocr_combo.setEnabled(False)

        # ── Help buttons ──────────────────────────────────────────────────────
        merge_help_btn = _toolbar_btn("Info", "Open the Merge feature guide.")
        merge_help_btn.setFixedSize(36, 36)
        merge_help_btn.clicked.connect(self._show_merge_help)

        help_btn = QPushButton("?")
        help_btn.setObjectName("helpBtn")
        help_btn.setToolTip("Open the Quick Start guide.")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.setFixedSize(36, 36)
        help_btn.clicked.connect(self._show_welcome)

        # ── Assemble ──────────────────────────────────────────────────────────
        toolbar.addWidget(load_img_btn)
        toolbar.addWidget(load_btn)
        toolbar.addWidget(save_btn)
        toolbar.addSeparator()
        toolbar.addWidget(self.edit_btn)
        toolbar.addWidget(split_btn)
        toolbar.addWidget(merge_btn)
        toolbar.addWidget(delete_btn)
        toolbar.addSeparator()
        toolbar.addWidget(toggle_bbox_btn)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  OCR:"))
        toolbar.addWidget(self.ocr_combo)
        toolbar.addSeparator()
        toolbar.addWidget(merge_help_btn)
        toolbar.addWidget(help_btn)
        # Force docTR LayoutLM baseline (no model selection UI).
        self.state.model_choice = {"arch": "layoutlmv3", "source": "ocr"}
        self._set_edit_mode(False)

    # ------------------------------------------------------------------ status bar
    def _build_statusbar(self):
        bar = QStatusBar()
        bar.setStyleSheet("""
            QStatusBar {
                background-color: #0f172a;
                border-top: 1px solid #1e293b;
                color: #64748b;
                font-size: 11px;
                padding: 3px 8px;
            }
        """)
        self.setStatusBar(bar)
        self._status_label = QLabel("Tip: hover over any button for guidance.")
        self._status_label.setStyleSheet("color: #64748b; font-size: 11px;")
        bar.addWidget(self._status_label)

    # ---------------------------------------------------------- merge help popup
    def _show_merge_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("How to Merge Blocks")
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet("""
            QDialog { background-color: #ffffff; color: #0f172a; }
            QTextBrowser {
                background-color: #f8fafc; color: #0f172a;
                border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;
                font-size: 13px;
            }
            QPushButton {
                background-color: #dbeafe; color: #0f172a;
                border: 1px solid #bfdbfe; border-radius: 6px;
                padding: 8px 24px; font-size: 13px; font-weight: 600;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #cbdcf8; }
        """)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        browser = QTextBrowser()
        browser.setHtml("""
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; color: #e2e8f0; }
  h3 { color: #60a5fa; }
  li { margin: 6px 0; line-height: 1.6; }
  .key { background: #0f3460; border: 1px solid #1e4a8a; border-radius: 4px;
         padding: 1px 6px; font-family: monospace; color: #93c5fd; }
  .warn { background: #451a03; color: #fde68a; border-radius: 6px;
          padding: 8px 12px; margin-top: 10px; }
</style>
<h3>Merging Blocks</h3>
<p>The <b>Merge</b> feature combines two or more selected bounding boxes into a single
block, joining their text and computing a bounding box that covers all of them.</p>

<b>When to use it:</b>
<ul>
  <li>OCR over-segmented a single field into many small boxes.</li>
  <li>A label and its value were split across multiple blocks.</li>
</ul>

<b>How to do it (step-by-step):</b>
<ol>
  <li>Make sure <b> Edit Mode</b> is enabled.</li>
  <li>Select at least <b>2 blocks</b>:<br>
    &nbsp;&nbsp;• <span class="key">Shift + Click</span> to add to selection, or<br>
    &nbsp;&nbsp;• Click and drag on empty canvas space to lasso multiple boxes.</li>
  <li>Click the <b> Merge</b> button in the toolbar.</li>
  <li>The blocks collapse into one. Text is concatenated with a space.</li>
</ol>

<div class="warn">
⚠️ <b>Tip:</b> Merge cannot be partially undone. Use <span class="key">Ctrl+Z</span>
(Undo) immediately if the result isn't what you expected.
</div>
""")
        layout.addWidget(browser)

        ok_btn = QPushButton("Got it!")
        ok_btn.clicked.connect(dlg.accept)
        layout.addWidget(ok_btn, alignment=Qt.AlignRight)
        dlg.exec()

    def _maybe_show_edit_hint(self):
        if not self.show_edit_hint:
            return
        dlg = EditModeHint(self)
        dlg.exec()
        if dlg.dont_show_again():
            self.show_edit_hint = False
            self.settings.setValue("show_edit_hint", False)

    # ------------------------------------------------------------------ combos
    def _on_ocr_changed(self, text):
        self.state.ocr_engine = text
        self._maybe_reextract_current_image()

    # ------------------------------------------------------------------ actions
    def _set_edit_mode(self, enabled: bool):
        self.edit_mode = enabled
        self._apply_view_mode()
        if enabled:
            self._maybe_show_edit_hint()

    def _on_state_changed(self):
        if self.edit_mode and self.current_image_path:
            self._refresh_annotated_preview()

    def _apply_view_mode(self):
        if self.edit_mode:
            self.canvas.set_edit_mode(True)
            self.canvas.clear_background_image()
            if self.current_image_path:
                self._refresh_annotated_preview()
                self.image_view.show()
                self._apply_edit_layout_defaults(force=not self._edit_layout_applied)
        else:
            self.canvas.set_edit_mode(False)
            if self.current_image_path:
                self.canvas.set_background_image(self.current_image_path)
                self.canvas.fit_width(force=True)
            self.image_view.hide()

    def _refresh_annotated_preview(self):
        if not self.current_image_path:
            return
        pix = QPixmap(self.current_image_path)
        if pix.isNull():
            return

        label_colors = {
            "question": QColor("#3b82f6"),
            "answer": QColor("#22c55e"),
            "header": QColor("#a855f7"),
            "other": QColor("#9ca3af"),
        }

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Draw links first (under boxes).
        link_pen = QPen(QColor("#f59e0b"), 2)
        painter.setPen(link_pen)
        block_by_id = {b.get("id"): b for b in self.state.data.get("blocks", [])}
        for link in self.state.data.get("links", []):
            q = block_by_id.get(link.get("question_id"))
            a = block_by_id.get(link.get("answer_id"))
            if not q or not a:
                continue
            qb = q.get("bbox", [0, 0, 0, 0])
            ab = a.get("bbox", [0, 0, 0, 0])
            qcx, qcy = (qb[0] + qb[2]) / 2.0, (qb[1] + qb[3]) / 2.0
            acx, acy = (ab[0] + ab[2]) / 2.0, (ab[1] + ab[3]) / 2.0
            painter.drawLine(int(qcx), int(qcy), int(acx), int(acy))

        # Draw annotation boxes.
        for block in self.state.data.get("blocks", []):
            x1, y1, x2, y2 = block.get("bbox", [0, 0, 0, 0])
            label = str(block.get("label", "other")).lower()
            box_pen = QPen(label_colors.get(label, QColor("#9ca3af")), 2)
            painter.setPen(box_pen)
            painter.drawRect(int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1)))

        painter.end()
        self.image_view.set_pixmap(pix)
        self.image_view.fit_width(force=False)
        self.canvas.fit_width(force=False)

    def _apply_edit_layout_defaults(self, force: bool = False):
        """
        Give the image and annotation panes equal width the first time edit
        mode opens (or when forced), keeping the inspector narrower.
        """
        if not force and self._edit_layout_applied:
            return

        total = max(1, self.width())
        inspector_w = max(self.inspector.minimumWidth(), 260)
        remaining = max(1, total - inspector_w)
        left = remaining // 2
        right = remaining - left

        self.splitter.setSizes([left, right, inspector_w])
        self._edit_layout_applied = True

    def _selected_ids(self):
        return [
            item.block_id
            for item in self.canvas.scene.items()
            if hasattr(item, "isSelected") and item.isSelected()
        ]

    def _delete_selected(self):
        ids = self._selected_ids()
        if not ids:
            QMessageBox.information(self, "Delete", "No blocks selected.\nSelect a block first in Edit Mode.")
            return
        for bid in ids:
            self.commands.push(DeleteBlockCommand(self.state, bid))

    def _split_selected(self):
        ids = self._selected_ids()
        if not ids:
            QMessageBox.information(self, "Split", "No block selected.\nEnable Edit Mode and click a block first.")
            return
        if len(ids) > 1:
            QMessageBox.information(self, "Split", "Split works on one block at a time.\nPlease select exactly one block.")
            return
        block = self.state.get_block(ids[0])
        if not block:
            return
        words = block.get("text", "").split()
        if len(words) < 2:
            QMessageBox.information(
                self, "Split",
                "This block has fewer than two words — nothing to split."
            )
            return
        dlg = SplitDialog(words, self)
        if dlg.exec() != QDialog.Accepted:
            return
        split_at = dlg.split_index()
        if split_at is None:
            return
        self.commands.push(SplitBlockCommand(self.state, ids[0], split_at))

    def _merge_selected(self):
        ids = self._selected_ids()
        if len(ids) < 2:
            QMessageBox.information(
                self,
                "Merge — Select Multiple Blocks",
                "You need to select at least 2 blocks to merge them.\n\n"
                "How to select multiple blocks:\n"
                "  • Hold Shift and click each block, or\n"
                "  • Click and drag on empty canvas space to lasso them.\n\n"
                "Then click  Merge again.",
            )
            return
        self.commands.push(MergeBlocksCommand(self.state, ids))

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON", filter="JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state.load_json(data)
            self.canvas.clear_background_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")

    def _save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", filter="JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_json(), f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON: {e}")

    def _load_image_and_extract(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            filter="Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        self._extract_and_render(path)

    def _extract_and_render(self, path: str):
        """Run OCR + LayoutLM inference and load result into the editor."""
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        choice = self.state.model_choice or {"arch": "layoutlmv3", "source": "ocr"}
        arch = choice.get("arch", "layoutlmv3")
        source = choice.get("source", "ocr")

        def _latest_or_best(model_dir: str):
            if not os.path.exists(model_dir):
                return model_dir
            best_ckpt, best_f1 = None, -1.0
            for sub in os.listdir(model_dir):
                if not sub.startswith("checkpoint-"):
                    continue
                sp = os.path.join(model_dir, sub, "trainer_state.json")
                if os.path.exists(sp):
                    try:
                        with open(sp) as f:
                            st = json.load(f)
                        f1 = st.get("best_metric", -1.0) or -1.0
                        if f1 > best_f1:
                            best_f1, best_ckpt = f1, os.path.join(model_dir, sub)
                    except Exception:
                        pass
            if best_ckpt:
                return best_ckpt
            subs = sorted(
                [s for s in os.listdir(model_dir) if s.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1]), reverse=True
            )
            return os.path.join(model_dir, subs[0]) if subs else model_dir

        if arch == "docformer":
            model_dir = os.path.join(root, "LayoutLM", f"docformer-funsd-{self.state.ocr_engine}-base")
            checkpoint_dir = _latest_or_best(model_dir)
        else:
            # Always use the GT-bbox checkpoint (checkpoint-608) as the single
            # inference model for the editor. OCR engine still runs for text
            # extraction; the LayoutLM weights are engine-agnostic at inference.
            checkpoint_dir = os.path.join(
                root, "LayoutLM", "layoutlmv3-funsd", "checkpoint-608"
            )

        def _is_git_lfs_pointer_file(file_path: str) -> bool:
            try:
                with open(file_path, "rb") as f:
                    head = f.read(200)
                return head.startswith(b"version https://git-lfs.github.com/spec/v1")
            except Exception:
                return False

        def _validate_checkpoint_dir(dir_path: str):
            # transformers checkpoints typically include model.safetensors or pytorch_model.bin
            candidates = [
                os.path.join(dir_path, "model.safetensors"),
                os.path.join(dir_path, "pytorch_model.bin"),
            ]
            present = [p for p in candidates if os.path.exists(p)]
            if not present:
                raise FileNotFoundError(
                    "Model weights not found in checkpoint directory.\n"
                    f"Expected one of: {', '.join(os.path.basename(p) for p in candidates)}\n"
                    f"Checkpoint: {dir_path}"
                )
            for p in present:
                if _is_git_lfs_pointer_file(p):
                    raise RuntimeError(
                        "Model weights look like Git LFS pointer files (not downloaded).\n\n"
                        "Fix on the cloned repo:\n"
                        "  1) Install git-lfs\n"
                        "  2) Run: git lfs install\n"
                        "  3) Run: git lfs pull\n\n"
                        f"Pointer file: {p}"
                    )

        _validate_checkpoint_dir(checkpoint_dir)

        # Show progress dialog so users know the app hasn't frozen
        progress = QProgressDialog(
            "Running OCR + LayoutLM inference…\nThis may take 10–30 seconds.",
            None,           # no cancel button
            0, 0,           # 0,0 = indeterminate (pulsing bar)
            self,
        )
        progress.setWindowTitle("Processing")
        progress.setMinimumWidth(340)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowFlags(
            progress.windowFlags()
            & ~Qt.WindowContextHelpButtonHint
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
        )
        progress.setStyleSheet("""
            QProgressDialog {
                background-color: #0f172a;
                color: #e2e8f0;
                border: 1px solid #1e293b;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 13px;
                padding: 8px 4px;
            }
            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 4px;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        progress.show()
        QApplication.processEvents()

        try:
            from LayoutLM.layoutlm_customOCR import run_inference
            data = run_inference(path, checkpoint_dir, self.state.ocr_engine, model_arch=arch)
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Inference Error", f"Failed to extract from image:\n{e}")
            return
        finally:
            progress.close()

        self.state.load_json(data)
        self.current_image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            self.state.image_size = (pix.width(), pix.height())
        self._apply_view_mode()

    def _maybe_reextract_current_image(self):
        if self.current_image_path:
            self._extract_and_render(self.current_image_path)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
