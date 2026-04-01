from train.ddpo_rl.algorithms.actor_critic import ActorCriticAlgorithm
from train.ddpo_rl.algorithms.reinforce import ReinforceAlgorithm


def build_algorithm(name: str, **kwargs):
    if name == "reinforce":
        return ReinforceAlgorithm(**kwargs)
    if name == "actor_critic":
        return ActorCriticAlgorithm(**kwargs)
    raise ValueError(f"Unsupported RL algorithm: {name}")
