from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QComboBox,
    QPushButton,
    QLabel,
    QFrame,
    QSpinBox,
)
from PySide6.QtCore import Qt

from editor.commands import (
    UpdateBlockCommand,
    DeleteBlockCommand,
)

_INSPECTOR_STYLE = """
QWidget {
    background-color: #0f172a;
    color: #e2e8f0;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 12px;
}
QLabel#sectionHeader {
    color: #3b82f6;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 6px 0 2px 0;
}
QLabel#idLabel {
    background-color: #1e293b;
    color: #93c5fd;
    border: 1px solid #1e40af;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
}
QTextEdit {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 6px;
    font-size: 12px;
    selection-background-color: #1d4ed8;
}
QTextEdit:focus { border-color: #3b82f6; }
QTextEdit:disabled { background-color: #0f172a; color: #334155; }
QComboBox {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 5px 8px;
    min-height: 26px;
}
QComboBox:hover { border-color: #3b82f6; }
QComboBox::drop-down { border: 0; padding-right: 4px; }
QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #e2e8f0;
    selection-background-color: #1d4ed8;
    border: 1px solid #334155;
    outline: none;
}
QSpinBox {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 5px 8px;
    min-height: 26px;
}
QSpinBox:focus { border-color: #3b82f6; }
QPushButton {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 500;
    min-height: 28px;
}
QPushButton:hover { background-color: #334155; border-color: #475569; color: #f1f5f9; }
QPushButton:pressed { background-color: #1e3a5f; border-color: #3b82f6; }
QPushButton:disabled { color: #334155; border-color: #1e293b; }
QPushButton#saveBtn {
    background-color: #1e3a5f;
    border-color: #1d4ed8;
    color: #93c5fd;
}
QPushButton#saveBtn:hover {
    background-color: #1d4ed8;
    border-color: #3b82f6;
    color: #ffffff;
}
QPushButton#deleteBtn {
    color: #f87171;
    border-color: #450a0a;
}
QPushButton#deleteBtn:hover {
    background-color: #450a0a;
    border-color: #dc2626;
    color: #fca5a5;
}
QPushButton#undoRedoBtn {
    background-color: #1e3a5f;
    border-color: #2d5494;
    color: #93c5fd;
    font-size: 12px;
    min-height: 26px;
}
QPushButton#undoRedoBtn:hover {
    background-color: #1d4ed8;
    border-color: #3b82f6;
    color: #ffffff;
}
QFrame[frameShape="4"] {
    color: #1e293b;
    background-color: #1e293b;
    max-height: 1px;
}
"""

LABEL_COLORS_CSS = {
    "question": "background:#1d4ed8; color:#bfdbfe;",
    "answer":   "background:#15803d; color:#bbf7d0;",
    "header":   "background:#7e22ce; color:#e9d5ff;",
    "other":    "background:#374151; color:#d1d5db;",
}


def _section(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionHeader")
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    return line


class InspectorPanel(QWidget):
    def __init__(self, state, command_manager):
        super().__init__()
        self.setStyleSheet(_INSPECTOR_STYLE)
        self.setMinimumWidth(200)
        self.setMaximumWidth(240)
        self.state = state
        self.cmd = command_manager
        self.current_block_id: Optional[int] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Title
        title_lbl = QLabel("Inspector")
        title_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #60a5fa; padding-bottom: 2px;")
        layout.addWidget(title_lbl)
        layout.addWidget(_divider())

        # Block badge
        self.id_label = QLabel("No block selected")
        self.id_label.setObjectName("idLabel")
        self.id_label.setWordWrap(True)
        layout.addWidget(self.id_label)

        # Text
        layout.addWidget(_section("Text Content"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Block text…")
        self.text_edit.setMaximumHeight(80)
        layout.addWidget(self.text_edit)

        # Label
        layout.addWidget(_section("Label"))
        self.label_dropdown = QComboBox()
        self.label_dropdown.addItems(["question", "answer", "header", "other"])
        self.label_dropdown.setToolTip(
            "question – form field label\n"
            "answer   – filled-in value\n"
            "header   – section heading\n"
            "other    – anything else"
        )
        layout.addWidget(self.label_dropdown)

        # Font size
        layout.addWidget(_section("Font Size"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(2, 72)
        self.font_size_spin.setValue(10)
        layout.addWidget(self.font_size_spin)

        layout.addWidget(_divider())

        # Actions
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.setToolTip("Apply text/label changes to this block.")
        layout.addWidget(self.save_btn)

        self.delete_btn = QPushButton("Remove Block")
        self.delete_btn.setObjectName("deleteBtn")
        self.delete_btn.setToolTip("Remove this block. Undo with Ctrl+Z.")
        layout.addWidget(self.delete_btn)

        # Imports needed for undo/redo
        from editor.commands import CommandManager

        layout.addStretch()

        # Tiny hint
        hint = QLabel("Tip: double-click a block in Edit Mode to edit its text inline.")
        hint.setStyleSheet("color: #475569; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addWidget(_divider())

        # Undo / Redo row at the very bottom of the inspector
        from PySide6.QtWidgets import QHBoxLayout as _HBox
        ur_row = _HBox()
        ur_row.setSpacing(4)
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setObjectName("undoRedoBtn")
        self._undo_btn.setToolTip("Undo last action")
        self._undo_btn.clicked.connect(self.cmd.undo)
        self._redo_btn = QPushButton("Redo")
        self._redo_btn.setObjectName("undoRedoBtn")
        self._redo_btn.setToolTip("Redo last undone action")
        self._redo_btn.clicked.connect(self.cmd.redo)
        ur_row.addWidget(self._undo_btn)
        ur_row.addWidget(self._redo_btn)
        layout.addLayout(ur_row)

        # Signals
        self.save_btn.clicked.connect(self.save_changes)
        self.delete_btn.clicked.connect(self.delete_block)
        state.selection_changed.connect(self.load_block)

        self._set_enabled(False)

    def _set_enabled(self, enabled: bool):
        for w in [self.text_edit, self.label_dropdown, self.font_size_spin,
                  self.save_btn, self.delete_btn]:
            w.setEnabled(enabled)

    def load_block(self, block):
        if not block:
            self.current_block_id = None
            self.id_label.setText("No block selected")
            self.text_edit.clear()
            self.label_dropdown.setCurrentIndex(0)
            self.font_size_spin.setValue(10)
            self._set_enabled(False)
            return

        self._set_enabled(True)
        self.current_block_id = block.get("id")
        lbl = str(block.get("label", "other")).lower()
        css = LABEL_COLORS_CSS.get(lbl, LABEL_COLORS_CSS["other"])
        self.id_label.setText(
            f"<span style='{css} padding:2px 6px; border-radius:4px; font-size:10px;'>{lbl.upper()}</span>"
            f"  Block #{block.get('id', '-')}"
        )
        self.id_label.setTextFormat(Qt.RichText)
        self.text_edit.setPlainText(block.get("text", ""))
        self.label_dropdown.setCurrentText(lbl)
        self.font_size_spin.setValue(block.get("font_size", 10))

    def save_changes(self):
        if self.current_block_id is None:
            return
        self.cmd.push(
            UpdateBlockCommand(
                self.state,
                self.current_block_id,
                {
                    "text": self.text_edit.toPlainText(),
                    "label": self.label_dropdown.currentText(),
                    "font_size": self.font_size_spin.value(),
                },
            )
        )

    def delete_block(self):
        if self.current_block_id is None:
            return
        self.cmd.push(DeleteBlockCommand(self.state, self.current_block_id))
        self.current_block_id = None
