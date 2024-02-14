from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from custom_dataset import TabletopOfflineDataset
from torch.utils.data import DataLoader

from src.iql import ImplicitQLearning
from src.policy import DiscretePolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy
from src.util import DEFAULT_DEVICE

def get_env_and_dataset(log, env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir), vars(args))
    log(f'Log dir: {log.dir}')

    #env, dataset = get_env_and_dataset(log, args.env_name, args.max_episode_steps)
    dataset = TabletopOfflineDataset(data_dir=args.data_dir, crop_size=args.crop_size, view='top')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    set_seed(args.seed)

    if args.deterministic_policy:
        policy = DeterministicPolicy(crop_size=args.crop_size)
    else:
        policy = DiscretePolicy(crop_size=args.crop_size)

    iql = ImplicitQLearning(
        qf=TwinQ(crop_size=args.crop_size),
        vf=ValueFunction(hidden_dim=args.hidden_dim),
        policy=policy,
        v_optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.v_learning_rate),
        q_optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.q_learning_rate),
        policy_optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.policy_learning_rate),
        max_steps=args.n_epochs * len(dataloader) // args.batch_size,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.n_epochs):
        with tqdm(dataloader) as bar:
            bar.set_description(f'Epoch {epoch}')
            for batch in bar:
                images = batch['image'].to(torch.float32).to(DEFAULT_DEVICE)
                patches = batch['patch'].to(torch.float32).to(DEFAULT_DEVICE)
                actions = batch['action'].to(DEFAULT_DEVICE)
                next_images = batch['next_image'].to(torch.float32).to(DEFAULT_DEVICE)
                rewards = batch['reward'].to(torch.float32).to(DEFAULT_DEVICE)
                terminals = batch['terminal'].to(DEFAULT_DEVICE)
                observations = [images, patches]
                next_observations = [next_images, None]
                losses = iql.update(observations, actions, next_observations, rewards, terminals)
                bar.set_postfix(losses)

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--crop-size', type=int, default=160)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--v-learning-rate', type=float, default=1e-4)
    parser.add_argument('--q-learning-rate', type=float, default=1e-5)
    parser.add_argument('--policy-learning-rate', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    main(parser.parse_args())