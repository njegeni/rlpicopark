from gymnasium.envs.registration import register

register(
    id="PicoPark-v0",
    entry_point="PicoPark_v0.envs:PicoParkEnv",
)
