from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Optional

from PySide6.QtCore import QObject, Signal


class Command:
    def do(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


class CommandManager(QObject):
    stack_changed = Signal()

    def __init__(self):
        super().__init__()
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    def push(self, cmd: Command):
        cmd.do()
        self._undo_stack.append(cmd)
        self._redo_stack.clear()
        self.stack_changed.emit()

    def undo(self):
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        self.stack_changed.emit()

    def redo(self):
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        cmd.do()
        self._undo_stack.append(cmd)
        self.stack_changed.emit()


class AddBlockCommand(Command):
    def __init__(self, state, text, bbox, label):
        self.state = state
        self.text = text
        self.bbox = bbox
        self.label = label
        self.created_id: Optional[int] = None

    def do(self):
        block = self.state.add_block(self.text, self.bbox, self.label)
        self.created_id = block["id"]
        self.state.select_block(self.created_id)

    def undo(self):
        if self.created_id is not None:
            self.state.delete_block(self.created_id)


class DeleteBlockCommand(Command):
    def __init__(self, state, block_id):
        self.state = state
        self.block_id = block_id
        self.snapshot: Optional[Dict] = None
        self.links_snapshot = []
        self.table_snapshot = []

    def do(self):
        b = self.state.get_block(self.block_id)
        if b is None:
            return
        self.snapshot = deepcopy(b)
        self.links_snapshot = [
            l
            for l in self.state.data.get("links", [])
            if l.get("question_id") == self.block_id or l.get("answer_id") == self.block_id
        ]
        
        self.table_snapshot = []
        for table in self.state.data.get("tables", []):
            for cell in table.get("cells", []):
                if cell.get("block_id") == self.block_id:
                    self.table_snapshot.append((table.get("table_id"), deepcopy(cell)))
                    
        self.state.delete_block(self.block_id)

    def undo(self):
        if not self.snapshot:
            return
        if "blocks" not in self.state.data:
            self.state.data["blocks"] = []
        if "links" not in self.state.data:
            self.state.data["links"] = []
        self.state.data["blocks"].append(self.snapshot)
        self.state.data["links"].extend(self.links_snapshot)
        
        for table_id, cell in getattr(self, "table_snapshot", []):
            table = self.state.get_table(table_id)
            if table is not None:
                if "cells" not in table:
                    table["cells"] = []
                table["cells"].append(cell)
                
        self.state.state_changed.emit()


class UpdateBlockCommand(Command):
    def __init__(self, state, block_id, new_values: Dict):
        self.state = state
        self.block_id = block_id
        self.new_values = new_values
        self.old_values: Optional[Dict] = None

    def do(self):
        block = self.state.get_block(self.block_id)
        if not block:
            return
        self.old_values = deepcopy(block)
        self.state.update_block(self.block_id, **self.new_values)

    def undo(self):
        if self.old_values:
            self.state.update_block(self.block_id, **self.old_values)


class AddLinkCommand(Command):
    def __init__(self, state, qid, aid):
        self.state = state
        self.qid = qid
        self.aid = aid

    def do(self):
        self.state.add_link(self.qid, self.aid)

    def undo(self):
        self.state.remove_link(self.qid, self.aid)


class RemoveLinkCommand(Command):
    def __init__(self, state, qid, aid):
        self.state = state
        self.qid = qid
        self.aid = aid

    def do(self):
        self.state.remove_link(self.qid, self.aid)

    def undo(self):
        self.state.add_link(self.qid, self.aid)


class MergeBlocksCommand(Command):
    def __init__(self, state, block_ids):
        self.state = state
        self.block_ids = block_ids
        self.new_id: Optional[int] = None
        self.snapshot = None

    def do(self):
        self.snapshot = deepcopy(self.state.data)
        self.new_id = self.state.merge_blocks(self.block_ids)

    def undo(self):
        if self.snapshot is not None:
            self.state.load_json(self.snapshot)


class SplitBlockCommand(Command):
    def __init__(self, state, block_id, split_at=None):
        self.state = state
        self.block_id = block_id
        self.split_at = split_at
        self.snapshot = None

    def do(self):
        self.snapshot = deepcopy(self.state.data)
        self.state.split_block(self.block_id, self.split_at)

    def undo(self):
        if self.snapshot is not None:
            self.state.load_json(self.snapshot)


class TableCreateCommand(Command):
    def __init__(self, state, block_ids):
        self.state = state
        self.block_ids = block_ids
        self.snapshot = None

    def do(self):
        self.snapshot = deepcopy(self.state.data)
        self.state.add_table_from_blocks(self.block_ids)

    def undo(self):
        if self.snapshot is not None:
            self.state.load_json(self.snapshot)
