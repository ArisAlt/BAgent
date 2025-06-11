import os
import sys
import json
import types

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_replay_writes_accuracy(tmp_path, monkeypatch):
    log_file = tmp_path / 'log.jsonl'
    frame_file = tmp_path / 'frame.png'
    log_file.write_text(json.dumps({
        'frame': str(frame_file),
        'action': 'click_btn',
        'state': {'obs': [0.0, 0.0], 'foo': 1}
    }) + '\n')

    cv2_mod = types.ModuleType('cv2')
    calls = []
    cv2_mod.imread = lambda p: calls.append(('imread', p)) or 'img'
    cv2_mod.putText = lambda *a, **k: calls.append(('text', a[1]))
    cv2_mod.imshow = lambda *a, **k: calls.append(('show',))
    cv2_mod.waitKey = lambda d: calls.append(('wait', d)) or ord('q')
    cv2_mod.destroyAllWindows = lambda: calls.append(('destroy',))
    cv2_mod.FONT_HERSHEY_SIMPLEX = 0
    monkeypatch.setitem(sys.modules, 'cv2', cv2_mod)

    # dummy torch and sb3 before importing replay_session
    sys.modules['torch'] = types.ModuleType('torch')
    sb3_mod = types.ModuleType('stable_baselines3')
    sb3_mod.PPO = type('PPO', (), {'load': classmethod(lambda cls, p, env=None: object())})
    sys.modules['stable_baselines3'] = sb3_mod
    # stub pre_train_data and env modules to avoid heavy deps
    pre_mod = types.ModuleType('pre_train_data')
    pre_mod.BCModel = object
    sys.modules['pre_train_data'] = pre_mod
    env_mod = types.ModuleType('src.env')
    env_mod.EveEnv = object
    sys.modules['src.env'] = env_mod

    import replay_session

    class DummyEnv:
        def __init__(self):
            self.actions = [('click', 'btn')]
            self.action_space = types.SimpleNamespace(n=1)
            self.observation_space = types.SimpleNamespace(shape=(2,))
    monkeypatch.setattr(replay_session, 'EveEnv', DummyEnv)

    class DummyModel:
        def predict(self, obs, deterministic=True):
            return 0, None
    monkeypatch.setattr(replay_session, 'load_model', lambda p, env: (DummyModel(), 'ppo'))

    acc_file = tmp_path / 'acc.json'
    replay_session.replay(str(log_file), delay=1, model_path='m.zip', accuracy_out=str(acc_file))

    assert ('imread', str(frame_file)) in calls
    assert acc_file.exists()
    data = json.loads(acc_file.read_text())
    assert data['accuracy'] == 1.0
    assert calls[-1] == ('destroy',)
