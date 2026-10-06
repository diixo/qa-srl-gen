"""Operation metadata independent of Django's model and migration state."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Operation:
    code: str
    label: str


class OperationRegistry:
    """An ordered, validated snapshot of the operation catalogue.

    The default snapshot is cached per process. Restart the server and worker
    after editing operations.json. Loading metadata never imports handlers.
    """

    def __init__(self, path):
        self.path = Path(path)
        try:
            rows = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot load operation registry {self.path}: {error}') from error
        if not isinstance(rows, list):
            raise ValueError(f'{self.path}: expected a JSON array of operations')
        operations = {}
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict) or set(row) != {'code', 'label'}:
                raise ValueError(f'{self.path}: entry {number} must contain code and label')
            code, label = row['code'], row['label']
            if not isinstance(code, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,23}', code):
                raise ValueError(f'{self.path}: invalid operation code at entry {number}')
            if code in operations:
                raise ValueError(f'{self.path}: duplicate operation code {code!r}')
            if not isinstance(label, str) or not label.strip() or label != label.strip():
                raise ValueError(f'{self.path}: invalid label for operation {code!r}')
            operations[code] = Operation(code, label)
        self._operations = operations

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls):
        return cls(Path(__file__).with_name('operations.json'))

    def choices(self, allowed=None):
        """Return form choices in catalogue order; None means all operations."""
        allowed = None if allowed is None else set(allowed)
        return [(operation.code, operation.label) for operation in self._operations.values()
                if allowed is None or operation.code in allowed]

    def require(self, code):
        if not isinstance(code, str) or code not in self._operations:
            raise ValueError(f'Unknown operation: {code!r}')
        return self._operations[code]

    def label(self, code):
        """Keep historical jobs readable even if their operation was removed."""
        operation = self._operations.get(code)
        return operation.label if operation is not None else code
