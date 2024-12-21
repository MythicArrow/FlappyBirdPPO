import jax
import jax.numpy as jnp
import optax
from flax import nnx
import numpy as np
import gym
from typing import Any, Tuple, List


# Hyperparameters
gamma = 0.99  # Discount factor
lr_actor = 0.0003  # Actor learning rate
lr_critic = 0.0003  # Critic learning rate
clip_ratio = 0.2  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization
max_episodes = 1000
max_steps_per_episode = 500

env = gym.make("FlappyBird-v0")

# Extract observation and action space information
state_size = env.observation_space.shape
action_size = env.action_space.n # jump and do nothing so 2 possible actions

# Actor-Critic Model with nnx
class ActorCritic(nnx.Module):
    action_size: int

    def __init__(self):
        self.shared_layer = nnx.Linear(64)
        self.policy_layer = nnx.Linear(self.action_size)
        self.value_layer = nnx.Linear(1)

    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jax.nn.relu(self.shared_layer(state))
        policy_logits = self.policy_layer(x)
        value = self.value_layer(x)
        return policy_logits, value


# PPO loss function
def ppo_loss_fn(
    params: Any,
    model: nnx.Module,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns: jnp.ndarray,
):
    logits, values = model.apply(params, states)

    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits)
    log_probs_for_actions = log_probs[jnp.arange(actions.shape[0]), actions]

    # Policy loss with clipping
    ratios = jnp.exp(log_probs_for_actions - old_log_probs)
    clipped_ratios = jnp.clip(ratios, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -jnp.mean(jnp.minimum(ratios * advantages, clipped_ratios * advantages))

    # Value loss
    value_loss = jnp.mean((returns - values.squeeze()) ** 2)

    # Total loss
    entropy = -jnp.mean(jax.nn.softmax(logits) * log_probs)
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    return total_loss


# Advantages calculation
def compute_advantages(rewards: List[float], values: List[float], gamma: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    discounted_sum = 0
    returns = []
    for r in rewards[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = jnp.array(returns)
    advantages = returns - jnp.array(values)
    return returns, (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)


# Training step
@jax.jit
def train_step(state, model, optimizer, states, actions, old_log_probs, advantages, returns):
    def loss_fn(params):
        return ppo_loss_fn(params, model, states, actions, advantages, old_log_probs, returns)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Training loop
model = ActorCritic(action_size)
key = jax.random.PRNGKey(42)
state = nnx.ModuleState(model.init(key, jnp.ones((state_size,))), optax.adam(learning_rate=lr_actor))

for episode in range(max_episodes):
    states, actions, rewards, values = [], [], [], []
    state_t = env.reset()

    for t in range(max_steps_per_episode):
        state_t = jnp.expand_dims(jnp.array(state_t, dtype=jnp.float32), axis=0)

        logits, value = model.apply(state.params, state_t)
        probs = jax.nn.softmax(logits)
        action = jax.random.choice(key, jnp.arange(action_size), p=probs[0])
        next_state, reward, done, _ = env.step(action.item())

        # Store experience
        states.append(state_t.squeeze())
        actions.append(action)
        rewards.append(reward)
        values.append(value.squeeze())

        state_t = next_state

        if done:
            # Compute advantages and returns
            returns, advantages = compute_advantages(rewards, values, gamma)

            # Convert to JAX arrays
            states = jnp.array(states)
            actions = jnp.array(actions)
            old_log_probs = jax.nn.log_softmax(model.apply(state.params, states)[0])[jnp.arange(actions.shape[0]), actions]

            # Update model
            for _ in range(epochs):
                state, loss = train_step(state, model, optimizer, states, actions, old_log_probs, advantages, returns)

            print(f"Episode {episode + 1}, Loss: {loss}")
            break    
