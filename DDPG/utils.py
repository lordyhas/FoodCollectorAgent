import uuid

from mlagents_envs.side_channel import SideChannel


class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))
    def on_message_received(self, msg: bytes) -> None:
        pass