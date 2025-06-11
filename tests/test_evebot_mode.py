# version: 0.1.0
# path: tests/test_evebot_mode.py
import sys
import types

sys.path.append('src')

# Provide dummy dependencies required by EveBot initialization

def setup_dummy_modules(monkeypatch):
    pg_mod = types.ModuleType("pyautogui")
    pg_mod.moveTo = pg_mod.click = lambda *a, **k: None
    pg_mod.keyDown = pg_mod.keyUp = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "pyautogui", pg_mod)

    cv2_mod = types.ModuleType("cv2")
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)

    cap_mod = types.ModuleType("capture_utils")
    cap_mod.capture_screen = lambda select_region=False: None
    monkeypatch.setitem(sys.modules, "capture_utils", cap_mod)

    ocr_mod = types.ModuleType("ocr")
    ocr_mod.OcrEngine = lambda *a, **k: types.SimpleNamespace(extract_text=lambda img: "")
    monkeypatch.setitem(sys.modules, "ocr", ocr_mod)

    cv_mod = types.ModuleType("cv")
    cv_mod.CvEngine = lambda *a, **k: types.SimpleNamespace(detect_elements=lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "cv", cv_mod)

    ui_mod = types.ModuleType("ui")
    ui_mod.Ui = lambda *a, **k: types.SimpleNamespace(capture=lambda: None, execute=lambda c: None)
    monkeypatch.setitem(sys.modules, "ui", ui_mod)

    sm_mod = types.ModuleType("state_machine")
    sm_mod.FSM = lambda *a, **k: types.SimpleNamespace(on_event=lambda *a, **k: None, state=types.SimpleNamespace(name="IDLE"))
    sm_mod.Event = types.SimpleNamespace(START_MINING=1, DOCK=2)
    monkeypatch.setitem(sys.modules, "state_machine", sm_mod)

    ma_mod = types.ModuleType("mining_actions")
    ma_mod.MiningActions = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mining_actions", ma_mod)

    qt_mod = types.ModuleType("PySide6")
    class DummyWidget:
        def __init__(self, *a, **k):
            pass
        def setReadOnly(self, *a, **k):
            pass
        def setPlaceholderText(self, *a, **k):
            pass
        def append(self, *a, **k):
            pass
        def setText(self, *a, **k):
            pass
        def clicked(self):
            return types.SimpleNamespace(connect=lambda f: None)
        def show(self):
            pass

    qt_widgets = types.SimpleNamespace(
        QApplication=lambda *a, **k: None,
        QWidget=DummyWidget,
        QVBoxLayout=lambda *a, **k: None,
        QPushButton=lambda *a, **k: DummyWidget(),
        QTextEdit=lambda *a, **k: DummyWidget(),
        QLabel=lambda *a, **k: DummyWidget(),
        QLineEdit=lambda *a, **k: DummyWidget(),
        QShortcut=lambda *a, **k: None,
    )
    qt_mod.QtWidgets = qt_widgets
    qt_mod.QtCore = types.ModuleType("QtCore")
    qt_mod.QtGui = types.ModuleType("QtGui")
    monkeypatch.setitem(sys.modules, "PySide6", qt_mod)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", qt_widgets)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qt_mod.QtCore)
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", qt_mod.QtGui)

    agent_mod = types.ModuleType("agent")
    class DummyPilot:
        def __init__(self, *a, **k):
            self.env = types.SimpleNamespace(_action_to_command=lambda i: f"cmd{i}", _compute_reward=lambda: 0.0, get_observation=lambda: {"obs": []})
        def bc_predict(self, obs):
            return 0
    agent_mod.AIPilot = DummyPilot
    monkeypatch.setitem(sys.modules, "agent", agent_mod)

    env_mod = types.ModuleType("env")
    class DummyEnv:
        def _action_to_command(self, idx):
            return f"cmd{idx}"
        def _compute_reward(self):
            return 0.0
        def get_observation(self):
            return {"obs": []}
    env_mod.EveEnv = DummyEnv
    monkeypatch.setitem(sys.modules, "env", env_mod)

    rh_mod = types.ModuleType("roi_capture")
    rh_mod.RegionHandler = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "roi_capture", rh_mod)


def test_set_mode(monkeypatch):
    setup_dummy_modules(monkeypatch)
    from bot_core import EveBot
    bot = EveBot()
    bot.set_mode("manual")
    assert bot.mode == "manual"
    bot.set_mode("assist")
    assert bot.mode == "assist"
    # Remove imported module so other tests can patch PySide6 cleanly
    sys.modules.pop("bot_core", None)
    sys.modules.pop("src.bot_core", None)
