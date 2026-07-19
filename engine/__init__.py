"""Engine subpackage.

Kept import-light on purpose: ``engine.steps.squilla_router`` imports
``decision_record``, which imports from ``engine.routing.calibration`` — and
importing any ``engine.routing.*`` submodule runs this ``__init__`` first.
Eagerly importing the step here would therefore create a circular import
(``calibration`` import → this ``__init__`` → step → ``decision_record``
mid-init). Import the step directly instead::

    from squilla_router_standalone.engine.steps.squilla_router import (
        apply_squilla_router, router_runtime_status,
    )
"""
