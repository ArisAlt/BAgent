# src/state_machine.py (version 0.2.0)
from enum import Enum, auto

class State(Enum):
    IDLE = auto()
    MINING = auto()
    COMBAT = auto()
    DOCKING = auto()
    MISSION = auto()
    EXPLORATION = auto()

class Event(Enum):
    START_MINING = auto()
    ENEMY_DETECTED = auto()
    DOCK = auto()
    UNDOCK = auto()
    MISSION_ACCEPTED = auto()
    EXPLORATION_FOUND = auto()
    ALERT_HOSTILE = auto()

class FSM:
    def __init__(self):
        self.state = State.IDLE

    def on_event(self, event):
        if self.state == State.IDLE:
            if event == Event.START_MINING:
                self.state = State.MINING
            elif event == Event.MISSION_ACCEPTED:
                self.state = State.MISSION
            elif event == Event.EXPLORATION_FOUND:
                self.state = State.EXPLORATION

        elif self.state == State.MINING:
            if event == Event.ENEMY_DETECTED:
                self.state = State.COMBAT
            elif event == Event.DOCK:
                self.state = State.DOCKING

        elif self.state == State.COMBAT:
            if event == Event.DOCK:
                self.state = State.DOCKING
            elif event == Event.ALERT_HOSTILE:
                self.state = State.COMBAT

        elif self.state == State.DOCKING:
            if event == Event.UNDOCK:
                self.state = State.IDLE

        elif self.state == State.MISSION:
            if event == Event.DOCK:
                self.state = State.DOCKING

        elif self.state == State.EXPLORATION:
            if event == Event.ENEMY_DETECTED:
                self.state = State.COMBAT

        return self.state
