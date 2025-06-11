import os
import sys
import json
import types

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_replay_loads_log(tmp_path, monkeypatch):
    log_file = tmp_path / 'log.jsonl'
    frame_file = tmp_path / 'frame.png'
    log_file.write_text(json.dumps({'frame': str(frame_file), 'action': 'test'}) + '\n')

    cv2_mod = types.ModuleType('cv2')
    calls = []
    cv2_mod.imread = lambda p: calls.append(('imread', p)) or 'img'
    cv2_mod.putText = lambda img, text, pos, font, scale, color, thickness: calls.append(('text', text))
    cv2_mod.imshow = lambda win, img: calls.append(('show', img))
    cv2_mod.waitKey = lambda d: calls.append(('wait', d)) or ord('q')
    cv2_mod.destroyAllWindows = lambda: calls.append(('destroy',))
    cv2_mod.FONT_HERSHEY_SIMPLEX = 0
    monkeypatch.setitem(sys.modules, 'cv2', cv2_mod)

    from replay_session import replay
    replay(str(log_file), delay=1)

    assert ('imread', str(frame_file)) in calls
    assert calls[-1] == ('destroy',)
