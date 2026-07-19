"""squilla_router_standalone — standalone extraction of OpenSquilla's model router.

Public API:

    from squilla_router_standalone import Router, SquillaRouterConfig

    router = Router(SquillaRouterConfig())
    result = await router.route("重写这个函数", session_key="s1")
    print(result.tier, result.model, result.source)

Run as a Claude Code proxy:

    python -m squilla_router_standalone --setup
    python -m squilla_router_standalone
"""

from squilla_router_standalone.api import Router, RoutingResult
from squilla_router_standalone.config import RouterBudgetConfig, SquillaRouterConfig

__all__ = [
    "Router",
    "RoutingResult",
    "SquillaRouterConfig",
    "RouterBudgetConfig",
]
