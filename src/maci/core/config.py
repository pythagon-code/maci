from types import SimpleNamespace

config = SimpleNamespace(
    dim=SimpleNamespace(
        actor_flow=128,
        critic_flow=128
    ),
    flow=SimpleNamespace(
        count=6,
        max_lifespan=6
    )
    transformer=SimpleNamespace(
        
    )
)