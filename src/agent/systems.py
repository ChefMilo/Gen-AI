"""Registers the agent systems with the experiment runner (evaluation.systems).

Listed in an experiment config's systems block, e.g. configs/final.yaml:

    systems:
      names: [r0, r1, r2, r3, r3-u]
      modules: [agent.systems]

Importing this module registers r1 (baselines.r1_single_shot) and r2, r3 and r3-u
(agent.system) through
evaluation.systems.register_system. This is the only place that registers them (the
agent.experiment CLI imports this module too). Nothing else imports it, so importing the
agent package never changes the runner's registry by itself. r3-u is R3 with the plan's
budget and no multiple-testing correction.
"""

from agent.system import AGENT_FACTORIES
from baselines.r1_single_shot import R1System
from evaluation.systems import register_system

FACTORIES = {"r1": R1System, **AGENT_FACTORIES}


def register() -> None:
    """Register every agent system; a no-op when they are already registered."""
    for name, factory in FACTORIES.items():
        register_system(name, factory, source=__name__)


register()
