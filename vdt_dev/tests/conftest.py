from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# The local test environment may not have the full RL stack installed. These stubs are
# enough for importing `src.util`/`src.iql` in smoke tests because the exercised codepath
# does not call into Gym or D4RL.
sys.modules.setdefault("d4rl", types.ModuleType("d4rl"))
sys.modules.setdefault("gym", types.ModuleType("gym"))

