import numpy as np

class simple_bandit():
    def __init__(self, n, epsilon=0.1):
        self.size = n
        self.epsilon = epsilon
        self.N = [0 for _ in range(n)]
        self.Q = [0 for _ in range(n)]
        self.q_star = [np.random.normal(0, 1) for _ in range(n)]
        self.optimal_action = int(np.argmax(self.q_star))

    def choose(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.size)
        # 同点(タイ)の場合はランダムに解消する
        best = np.max(self.Q)
        return np.random.choice(np.flatnonzero(np.array(self.Q) == best))

    def bandit(self, a):
        return np.random.normal(self.q_star[a], 1)

    def update(self):
        a = self.choose()
        reward = self.bandit(a)
        self.N[a] += 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])
        return a, reward

    def step(self, steps):
        """steps回 update() を行い、各ステップの報酬と最適行動を選べたかどうかを記録して返す。

        図2.2(10本腕テストベッド)の再現に使うことを想定している。
        """
        rewards = np.zeros(steps)
        is_optimal = np.zeros(steps, dtype=bool)
        for t in range(steps):
            a, reward = self.update()
            rewards[t] = reward
            is_optimal[t] = (a == self.optimal_action)
        return rewards, is_optimal


def run_experiment(n_arms=10, epsilon=0.1, n_runs=2000, steps=1000):
    """異なる乱数のバンディット問題を n_runs 回生成し、それぞれ steps ステップ実行して
    平均報酬と最適行動選択率の推移を返す(図2.2相当の実験)。
    """
    all_rewards = np.zeros((n_runs, steps))
    all_optimal = np.zeros((n_runs, steps))
    for run in range(n_runs):
        bandit = simple_bandit(n_arms, epsilon=epsilon)
        rewards, is_optimal = bandit.step(steps)
        all_rewards[run] = rewards
        all_optimal[run] = is_optimal
    avg_reward = all_rewards.mean(axis=0)
    pct_optimal = all_optimal.mean(axis=0) * 100
    return avg_reward, pct_optimal


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(0)

    N_ARMS = 10
    N_RUNS = 2000
    STEPS = 1000
    EPSILONS = [0, 0.01, 0.1]

    results = {eps: run_experiment(N_ARMS, eps, N_RUNS, STEPS) for eps in EPSILONS}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    for eps, (avg_reward, pct_optimal) in results.items():
        label = "greedy (ε=0)" if eps == 0 else f"ε={eps}"
        ax1.plot(avg_reward, label=label)
        ax2.plot(pct_optimal, label=label)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.legend()

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("figure_2_2_reproduction.png", dpi=150)
    print("Saved plot to figure_2_2_reproduction.png")