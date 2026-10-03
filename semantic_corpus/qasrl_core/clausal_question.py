"""A frame and its questioned slot (``labeling/ClausalQuestion.scala``, MIT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .frame import ArgStructure, ArgumentSlot, Frame


@dataclass(frozen=True, slots=True)
class ClausalQuestion:
    frame: Frame
    slot: ArgumentSlot

    @property
    def question_string(self) -> str:
        return self.frame.questions_for_slot(self.slot)[0]

    @property
    def clause_template(self) -> ArgStructure:
        return self.frame.structure.forget_animacy()

    def to_json(self) -> dict[str, Any]:
        return {"frame": self.frame.to_json(), "slot": self.slot.to_json()}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ClausalQuestion":
        return cls(Frame.from_json(data["frame"]), ArgumentSlot.from_json(data["slot"]))
