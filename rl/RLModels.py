from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize,
                                              unwrap_vec_normalize)

from rl.Car import RLCar


class RLModelHandler:
    """Handler class for training and testing RL models on the kart racing simulator."""

    def __init__(self, make_env):
        """
        :param make_env: The function to create the environment.
        """
        self.make_env = make_env
        self.sim = []

    def create_env(self):
        """Create a new environment instance and wrap it with Monitor for logging."""
        env = self.make_env()
        self.sim.append(env)
        return Monitor(env)

    def train(self, filename, from_model=None, nb_steps=1_000_000, n_envs=4):
        """
        Train a RL model and save it along with the VecNormalize environment.

        :param filename: The filename to save the model and VecNormalize environment.
        :param from_model: The filename of an existing model to continue training from (optional).
        :param nb_steps: The total number of training steps.
        :param n_envs: The number of parallel environments to use for training.
        """
        if from_model is None:
            # Create environment(s) for training
            if n_envs > 1:
                env = SubprocVecEnv([self.create_env for i in range(n_envs)])
            else:
                env = DummyVecEnv([self.create_env])

            # Wrap the environment with VecNormalize for observation and reward normalization
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            # Create a new PPO model
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
            )
        else:
            # Load the existing model and environment
            model, env = self.load(from_model, is_training=True, n_envs=n_envs)

        # Train the model
        model.learn(total_timesteps=nb_steps)

        # Save the trained model and VecNormalize environment
        model.save(filename)
        vec_normalize = unwrap_vec_normalize(env)
        if vec_normalize is not None:
            vec_normalize.save(filename + ".pkl")

        return model, env

    def load(self, filename, is_training=False, n_envs=1):
        """
        Load a RL model and its VecNormalize environment from files.
        :param filename: The filename to load the model and VecNormalize environment from.
        :param is_training: Whether the model will be used for further training.
        :param n_envs: The number of parallel environments to use if is_training is True.
        :return: The loaded model and environment.
        """
        if is_training and n_envs > 1:
            env = SubprocVecEnv([self.create_env for i in range(n_envs)])
        else:
            env = DummyVecEnv([self.create_env])

        env = VecNormalize.load(filename + ".pkl", env)
        env.training = is_training
        env.norm_reward = is_training

        model = PPO.load(filename, env=env)

        return model, env

    def test(self, rl_model_filename=None):
        """
        Test a RL model on the simulator

        :param rl_model_filename: The filename of the RL model and VecNormalize environment to load for testing
        """
        if rl_model_filename is not None:
            RLCar.set_model(self.load(rl_model_filename))

        if self.sim is None or len(self.sim) == 0:
            print("No simulation instance found. Creating a new one.")
            self.create_env()
        sim = self.sim[0]
        sim.run()
