# version: 0.1.0
# path: tests/test_gui_cli_integration.py

import sys
import types
import importlib
import numpy as np


def test_capture_cli(monkeypatch, tmp_path):
    import builtins
    import yaml
    import roi_capture

    inputs = iter(["capture sample", "click", "list", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    monkeypatch.setattr(roi_capture, "capture_screen", lambda select_region=False: np.zeros((10, 10, 3), dtype=np.uint8) if not select_region else (1, 1, 5, 5))

    class TempRegionHandler(roi_capture.RegionHandler):
        def __init__(self, yaml_path=None):
            super().__init__(yaml_path=str(tmp_path / "regions.yaml"))
        def get_screen_resolution(self):
            return (10, 10)

    monkeypatch.setattr(roi_capture, "RegionHandler", TempRegionHandler)

    roi_capture.capture_region_tool()

    data = yaml.safe_load(open(tmp_path / "regions.yaml"))
    assert "sample" in data.get("regions", {})


def test_bot_core_modes(monkeypatch):
    pg_mod = types.ModuleType("pyautogui")
    pg_mod.moveTo = pg_mod.click = lambda *a, **k: None
    pg_mod.keyDown = pg_mod.keyUp = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "pyautogui", pg_mod)

    cv2_mod = types.ModuleType("cv2")
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)

    cap_mod = types.ModuleType("capture_utils")
    cap_mod.capture_screen = lambda select_region=False: np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "capture_utils", cap_mod)

    ocr_mod = types.ModuleType("ocr")
    ocr_mod.OcrEngine = lambda *a, **k: types.SimpleNamespace(extract_text=lambda img: "")
    monkeypatch.setitem(sys.modules, "ocr", ocr_mod)

    cv_mod = types.ModuleType("cv")
    cv_mod.CvEngine = lambda *a, **k: types.SimpleNamespace(detect_elements=lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "cv", cv_mod)

    ui_mod = types.ModuleType("ui")
    ui_mod.Ui = lambda *a, **k: types.SimpleNamespace(capture=lambda: np.zeros((10,10,3), dtype=np.uint8), execute=lambda c: None)
    monkeypatch.setitem(sys.modules, "ui", ui_mod)

    sm_mod = types.ModuleType("state_machine")
    sm_mod.FSM = lambda *a, **k: types.SimpleNamespace(on_event=lambda *a, **k: None, state=types.SimpleNamespace(name="IDLE"))
    sm_mod.Event = types.SimpleNamespace(START_MINING=1, DOCK=2)
    monkeypatch.setitem(sys.modules, "state_machine", sm_mod)

    ma_mod = types.ModuleType("mining_actions")
    ma_mod.MiningActions = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mining_actions", ma_mod)

    agent_mod = types.ModuleType("agent")
    class DummyPilot:
        def __init__(self, *a, **k):
            self.env = importlib.import_module("env").EveEnv()
        def bc_predict(self, obs):
            return 0
    agent_mod.AIPilot = DummyPilot
    monkeypatch.setitem(sys.modules, "agent", agent_mod)

    import env  # relies on patched modules above

    qt_mod = types.ModuleType("PySide6")
    class DummyApp:
        def __init__(self, args):
            pass
        def exec(self):
            return 0
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
        QApplication=DummyApp,
        QWidget=DummyWidget,
        QVBoxLayout=lambda *a, **k: None,
        QPushButton=lambda *a, **k: DummyWidget(),
        QTextEdit=lambda *a, **k: DummyWidget(),
        QLabel=lambda *a, **k: DummyWidget(),
        QLineEdit=lambda *a, **k: DummyWidget(),
    )
    qt_mod.QtWidgets = qt_widgets
    qt_mod.QtCore = types.ModuleType("QtCore")
    qt_mod.QtGui = types.ModuleType("QtGui")
    monkeypatch.setitem(sys.modules, "PySide6", qt_mod)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", qt_widgets)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qt_mod.QtCore)
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", qt_mod.QtGui)

    import importlib
    bot_core = importlib.import_module("bot_core")

    monkeypatch.setattr(sys, "argv", ["bot_core.py", "--mode", "gui"])
    monkeypatch.setattr(sys, "exit", lambda *a, **k: None)
    bot_core.main()

    monkeypatch.setattr(sys, "argv", ["bot_core.py", "--mode", "bc_inference"])
    bot_core.main()

