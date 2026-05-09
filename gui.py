"""
gui.py — PyQt6 dashboard for JARVIS v2.
Shows status, conversation log, todo list, and memory stats.
"""

import sys
import threading
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFrame, QListWidget,
    QListWidgetItem, QSplitter, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QTextCursor, QFont

from jarvis import JARVIS


# ── Signals (thread-safe UI updates) ──────────────────────────────────────────
class Signals(QObject):
    status  = pyqtSignal(str, str)   # (state, text)
    log_msg = pyqtSignal(str, str)   # (role, text)
    todos   = pyqtSignal(list)       # updated todo list


signals = Signals()


# ── JARVIS background thread ───────────────────────────────────────────────────
class JARVISThread(QThread):
    def __init__(self):
        super().__init__()
        self.jarvis = None

    def run(self):
        def hook(state, text):
            signals.status.emit(state, text)
            if state == "thinking" and text:
                signals.log_msg.emit("YOU", text)
            elif state == "speaking" and text:
                signals.log_msg.emit("JARVIS", text)
                # Update todos after potential todo changes
                if self.jarvis:
                    pending = [t for t in self.jarvis.todos.todos if not t["done"]]
                    signals.todos.emit(pending)

        self.jarvis = JARVIS(status_callback=hook)
        self.jarvis.start()


# ── Stylesheet ────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget { background:#080c10; color:#c8dce8; }

QLabel#title {
    font-size:28px; font-weight:800; letter-spacing:10px;
    color:#4fc3f7; font-family:'Courier New';
}
QLabel#sub { font-size:9px; letter-spacing:3px; color:#1d3a50; font-family:'Courier New'; }
QLabel#status { font-size:10px; letter-spacing:2px; color:#4fc3f7; font-family:'Courier New'; }
QLabel#ind { font-size:9px; letter-spacing:1px; color:#1a3a50; font-family:'Courier New'; }

QTextEdit#log {
    background:#050810; color:#8fb8c8; border:0.5px solid #0d2030;
    border-radius:4px; font-family:'Courier New'; font-size:11px; padding:8px;
}
QListWidget#todos {
    background:#050810; color:#c8dce8; border:0.5px solid #0d2030;
    border-radius:4px; font-family:'Courier New'; font-size:11px; padding:4px;
}
QListWidget#todos::item { padding:4px 6px; border-bottom:0.5px solid #0d2030; }
QListWidget#todos::item:selected { background:#1a3a50; }

QPushButton {
    background:#0d1f2d; color:#4fc3f7; border:0.5px solid #1a3a50;
    border-radius:3px; padding:5px 12px; font-family:'Courier New';
    font-size:9px; letter-spacing:2px;
}
QPushButton:hover { background:#1a3a50; border-color:#4fc3f7; }

QFrame#sep { color:#0d2030; max-height:1px; }
QLabel#section { font-size:9px; color:#2a5a7a; letter-spacing:3px; font-family:'Courier New'; }
"""

STATE_COLORS = {
    "offline":   ("#555",    "● OFFLINE"),
    "loading":   ("#f59e0b", "● INITIALISING"),
    "listening": ("#22c55e", "● LISTENING"),
    "thinking":  ("#f59e0b", "● PROCESSING"),
    "speaking":  ("#4fc3f7", "● SPEAKING"),
}


class JARVISDashboard(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("JARVIS")
        self.resize(1000, 680)
        self.setMinimumSize(800, 500)
        self.setStyleSheet(STYLE)
        self._build()
        self._connect()
        self._set_state("loading")

        # Animated status
        self._tick = 0
        self._anim = QTimer()
        self._anim.timeout.connect(self._animate)
        self._anim.start(500)

        # Clock
        self._clock = QTimer()
        self._clock.timeout.connect(self._update_clock)
        self._clock.start(1000)
        self._update_clock()

        # Start JARVIS
        self.thread = JARVISThread()
        self.thread.start()

    def _build(self):
        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QHBoxLayout(root_w)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left panel ─────────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(220)
        left.setStyleSheet("background:#060b0f; border-right:0.5px solid #0d2030;")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(16, 20, 16, 16)
        lv.setSpacing(0)

        title = QLabel("J.A.R.V.I.S")
        title.setObjectName("title")
        title.setStyleSheet("font-size:20px; letter-spacing:6px;")
        lv.addWidget(title)

        sub = QLabel("PERSONAL AI")
        sub.setObjectName("sub")
        lv.addWidget(sub)
        lv.addSpacing(20)

        self.status_lbl = QLabel("● OFFLINE")
        self.status_lbl.setObjectName("status")
        lv.addWidget(self.status_lbl)
        lv.addSpacing(6)

        self.clock_lbl = QLabel()
        self.clock_lbl.setObjectName("ind")
        lv.addWidget(self.clock_lbl)
        lv.addSpacing(24)

        sep = QFrame(); sep.setObjectName("sep"); sep.setFrameShape(QFrame.Shape.HLine)
        lv.addWidget(sep)
        lv.addSpacing(16)

        # Todos panel
        todo_lbl = QLabel("TODO LIST")
        todo_lbl.setObjectName("section")
        lv.addWidget(todo_lbl)
        lv.addSpacing(6)

        self.todo_list = QListWidget()
        self.todo_list.setObjectName("todos")
        lv.addWidget(self.todo_list, 1)
        lv.addSpacing(8)

        self.todo_count = QLabel("0 pending")
        self.todo_count.setObjectName("ind")
        lv.addWidget(self.todo_count)
        lv.addSpacing(16)

        sep2 = QFrame(); sep2.setObjectName("sep"); sep2.setFrameShape(QFrame.Shape.HLine)
        lv.addWidget(sep2)
        lv.addSpacing(12)

        self.mem_lbl = QLabel("MEMORY: 0 turns")
        self.mem_lbl.setObjectName("ind")
        lv.addWidget(self.mem_lbl)
        lv.addSpacing(8)

        clear_btn = QPushButton("CLEAR LOG")
        clear_btn.clicked.connect(lambda: self.log.clear())
        lv.addWidget(clear_btn)

        root.addWidget(left)

        # ── Main panel ─────────────────────────────────────────────────────────
        main_w = QWidget()
        mv = QVBoxLayout(main_w)
        mv.setContentsMargins(20, 20, 20, 16)
        mv.setSpacing(10)

        conv_lbl = QLabel("CONVERSATION")
        conv_lbl.setObjectName("section")
        mv.addWidget(conv_lbl)

        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        mv.addWidget(self.log, 1)

        # Bottom indicators
        ind_row = QHBoxLayout()
        self.indicators = {}
        for name in ["MIC", "STT", "LLM", "TTS", "GPU"]:
            lbl = QLabel(f"[ {name} ]")
            lbl.setObjectName("ind")
            ind_row.addWidget(lbl)
            self.indicators[name] = lbl
        ind_row.addStretch()
        self.msg_count_lbl = QLabel("MSGS: 000")
        self.msg_count_lbl.setObjectName("ind")
        ind_row.addWidget(self.msg_count_lbl)
        mv.addLayout(ind_row)

        root.addWidget(main_w, 1)
        self._msg_count = 0

    def _connect(self):
        signals.status.connect(self._on_status)
        signals.log_msg.connect(self._append_msg)
        signals.todos.connect(self._update_todos)

    def _on_status(self, state, text):
        self._set_state(state)
        active = {
            "listening": ["MIC"],
            "thinking":  ["MIC", "STT", "LLM", "GPU"],
            "speaking":  ["TTS", "GPU"],
            "loading":   ["GPU"],
        }.get(state, [])
        for k, lbl in self.indicators.items():
            color = "#4fc3f7" if k in active else "#1a3a50"
            lbl.setStyleSheet(f"color:{color}; font-size:9px; letter-spacing:1px; font-family:'Courier New';")

        # Update memory count if we have access
        if self.thread.jarvis:
            n = len(self.thread.jarvis.memory.turns)
            self.mem_lbl.setText(f"MEMORY: {n} turns")

    def _set_state(self, state):
        color, label = STATE_COLORS.get(state, ("#555", "● OFFLINE"))
        self.status_lbl.setText(label)
        self.status_lbl.setStyleSheet(
            f"color:{color}; font-size:10px; letter-spacing:2px; font-family:'Courier New';"
        )

    def _append_msg(self, role, text):
        self._msg_count += 1
        self.msg_count_lbl.setText(f"MSGS: {self._msg_count:03d}")

        ts = datetime.now().strftime("%H:%M")
        if role == "YOU":
            pc, tc = "#4fc3f7", "#c8dce8"
        else:
            pc, tc = "#22c55e", "#8fb8c8"

        self.log.textCursor().movePosition(QTextCursor.MoveOperation.End)
        html = (
            f'<div style="margin:5px 0;font-family:Courier New;font-size:11px;">'
            f'<span style="color:#1a3a50">[{ts}] </span>'
            f'<span style="color:{pc};font-weight:bold">{role} › </span>'
            f'<span style="color:{tc}">{text}</span></div>'
        )
        self.log.textCursor().insertHtml(html)
        self.log.ensureCursorVisible()

    def _update_todos(self, todos):
        self.todo_list.clear()
        for t in todos:
            pri = " ★" if t.get("priority") == "high" else ""
            item = QListWidgetItem(f"○ {t['task']}{pri}")
            self.todo_list.addItem(item)
        self.todo_count.setText(f"{len(todos)} pending")

    def _animate(self):
        self._tick += 1

    def _update_clock(self):
        from modules.realtime import get_melbourne_time
        try:
            now = get_melbourne_time()
            self.clock_lbl.setText(now.strftime("%H:%M:%S  AEST"))
        except Exception:
            self.clock_lbl.setText(datetime.now().strftime("%H:%M:%S"))
