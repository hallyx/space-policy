"""Plugin import must register CLI choices before any model symbol is accessed."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_import_registers_cli_policy_choices_in_a_fresh_process():
    code = '''
import lerobot_policy_spacepolicy_act as plugin
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

expected = {
    "spline_act": "SplineACTPolicy",
    "base_aux_spline_act": "BaseAuxiliarySplineACTPolicy",
    "reaction_aware_act": "ReactionAwareACTPolicy",
}
# No configuration or model attributes have been requested from the package.
assert set(expected) <= set(PreTrainedConfig.get_known_choices())
for name, class_name in expected.items():
    assert get_policy_class(name).__name__ == class_name
plugin.register_policies()  # hooks must be idempotent
'''
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60,
                   capture_output=True, text=True)
