import numpy as np


class StationaryBandit:
    """定常なバンディット問題のクラス。"""

    def __init__(self, n):
        super().__init__()
        self.size = n
        self.N = [0 for _ in range(n)]
        self.Q = [0 for _ in range(n)]
        self.q_star = [np.random.normal(0, 1) for _ in range(n)]
        self.optimal_action = int(np.argmax(self.q_star))

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
            is_optimal[t] = a == self.optimal_action
        return rewards, is_optimal


class NonStationaryBandit(StationaryBandit):
    """非定常なバンディット問題のクラス。"""

    def bandit(self, a):
        self.q_star += np.random.normal(0, 0.01, self.size)
        return np.random.normal(self.q_star[a], 1)

    def step(self, steps):
        """steps回 update() を行い、各ステップの報酬と最適行動を選べたかどうかを記録して返す。"""
        rewards = np.zeros(steps)
        is_optimal = np.zeros(steps, dtype=bool)
        for t in range(steps):
            a, reward = self.update()
            self.optimal_action = int(np.argmax(self.q_star))
            rewards[t] = reward
            is_optimal[t] = a == self.optimal_action
        return rewards, is_optimal


class GreedyMixin:
    """Greedy法を用いたバンディット問題のクラス。"""

    def choose(self):
        best = np.max(self.Q)
        return np.random.choice(np.flatnonzero(np.array(self.Q) == best))


class EpsilonGreedyMixin(GreedyMixin):
    """ε-greedy法を用いたバンディット問題のクラス。
    self.epsilonをもったクラスと組み合わせて使う。
    """

    def choose(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.size)
        else:
            return super().choose()


class UCBMixin:
    """UCB法を用いたバンディット問題のクラス。
    self.c, self.tをもったクラスと組み合わせて使う。
    """

    def choose(self):
        self.t += 1
        ucb_values = [
            self.Q[a] + self.c * np.sqrt(np.log(self.t) / (self.N[a] + 1e-5))
            for a in range(self.size)
        ]
        best = np.max(ucb_values)
        return np.random.choice(np.flatnonzero(np.array(ucb_values) == best))


class GradientMixin:
    """選好を用いたバンディット問題のクラス。
    self.piをもったクラスと組み合わせて使う。
    """

    def choose(self):
        return np.random.choice(range(self.size), p=self.pi)


class Stationary_Greedy(StationaryBandit, GreedyMixin):
    """定常なバンディット問題のクラス。"""


class Stationary_EpsilonGreedy(StationaryBandit, EpsilonGreedyMixin):
    """ε-greedy法を用いたバンディット問題のクラス。"""

    def __init__(self, n, epsilon=0.1):
        super().__init__(n)
        self.epsilon = epsilon


class NonStationary_EpsilonGreedy(NonStationaryBandit, EpsilonGreedyMixin):
    """非定常なバンディット問題のクラス。
    練習問題2.5: 一定ステップ幅を用いた更新式を用いる。
    """

    def __init__(self, n, epsilon=0.1, alpha=0.1):
        super().__init__(n)
        self.epsilon = epsilon
        self.alpha = alpha

    def update(self):
        a = self.choose()
        reward = self.bandit(a)
        self.Q[a] = self.Q[a] + self.alpha * (reward - self.Q[a])
        return a, reward


class NonStationary_UnbiasedTrace(NonStationaryBandit, EpsilonGreedyMixin):
    """非定常なバンディット問題のクラス。
    練習問題2.7: バイアスをなくすようなステップ幅を組み込んだ更新式を用いる。
    """

    def __init__(self, n, epsilon=0.1, alpha=0.1):
        super().__init__(n)
        self.epsilon = epsilon
        self.alpha = alpha
        self.O = [0]

    def update(self):
        a = self.choose()
        reward = self.bandit(a)
        self.N[a] += 1
        n = self.N[a]
        if len(self.O) <= n:
            O_next = self.O[-1] + self.alpha * (1 - self.O[-1])
            self.O.append(O_next)
        beta = self.alpha / self.O[n]
        self.Q[a] = self.Q[a] + beta * (reward - self.Q[a])
        return a, reward


class Nonstationary_UCB(NonStationaryBandit, UCBMixin):
    """UCB法を用いたバンディット問題のクラス。
    UCB法を用いて探索を行う。
    """

    def __init__(self, n, c=2):
        super().__init__(n)
        self.c = c
        self.t = 0


class Nonstationary_Gradient(NonStationaryBandit, GradientMixin):
    """選好を用いたバンディット問題のクラス。"""

    def __init__(self, n, alpha=0.1):
        super().__init__(n)
        self.alpha = alpha
        self.H = [0 for _ in range(n)]
        self.pi = [1 / n for _ in range(n)]
        self.average_reward = 0.0
        self.t = 0

    def update(self):
        a = self.choose()
        reward = self.bandit(a)
        self.t += 1
        self.average_reward += (reward - self.average_reward) / self.t
        baseline = self.average_reward
        for i in range(self.size):
            if i == a:
                self.H[i] += self.alpha * (reward - baseline) * (1 - self.pi[i])
            else:
                self.H[i] -= self.alpha * (reward - baseline) * self.pi[i]
        exp_H = np.exp(self.H)
        self.pi = exp_H / np.sum(exp_H)
        return a, reward


def run_experiment_2_2(n_arms=10, epsilon=0.1, n_runs=2000, steps=1000):
    """異なる乱数のバンディット問題を n_runs 回生成し、それぞれ steps ステップ実行して
    平均報酬と最適行動選択率の推移を返す(図2.2相当の実験)。
    """
    all_rewards = np.zeros((n_runs, steps))
    all_optimal = np.zeros((n_runs, steps))
    for run in range(n_runs):
        bandit = Stationary_EpsilonGreedy(n_arms, epsilon=epsilon)
        rewards, is_optimal = bandit.step(steps)
        all_rewards[run] = rewards
        all_optimal[run] = is_optimal
    avg_reward = all_rewards.mean(axis=0)
    pct_optimal = all_optimal.mean(axis=0) * 100
    return avg_reward, pct_optimal


def parameter_sweep(
    cls,
    param_name,
    param_values,
    fixed_kwargs=None,
    n_arms=10,
    n_runs=5,
    steps=200_000,
    last_n=100_000,
):
    """cls(n_arms, **{param_name: v, **fixed_kwargs}) を n_runs 回生成し、
    steps ステップ実行した上で、最後の last_n ステップの平均報酬を求める。

    param_values に含まれる各パラメータ値についてこれを行い、n_runs 回の平均を返す
    (図2.6のパラメータ研究に相当)。
    """
    fixed_kwargs = fixed_kwargs or {}
    scores = np.zeros(len(param_values))
    for i, v in enumerate(param_values):
        run_avgs = np.zeros(n_runs)
        for r in range(n_runs):
            bandit = cls(n_arms, **{param_name: v, **fixed_kwargs})
            rewards, _ = bandit.step(steps)
            run_avgs[r] = rewards[-last_n:].mean()
        scores[i] = run_avgs.mean()
    return scores


def run_experiment_2_11(n_arms=10, n_runs=5, steps=200_000, last_n=100_000):
    """練習問題2.11: 練習問題2.5で述べた非定常のケースについて、
    図2.6に類似したパラメータ研究を行う。

    一定のステップサイズ(alpha=0.1)を用いたε-貪欲アルゴリズムを含め、
    UCB・勾配バンディットについても、それぞれ自身のパラメータを振って比較する。
    各アルゴリズム・パラメータ設定について、200,000ステップ実行し、
    最後の100,000ステップにわたる平均報酬を性能指標として用いる。
    """
    epsilons = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]
    cs = [1 / 4, 1 / 2, 1, 2, 4]
    alphas = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2]

    eps_greedy_scores = parameter_sweep(
        NonStationary_EpsilonGreedy,
        "epsilon",
        epsilons,
        fixed_kwargs={"alpha": 0.1},
        n_arms=n_arms,
        n_runs=n_runs,
        steps=steps,
        last_n=last_n,
    )
    ucb_scores = parameter_sweep(
        Nonstationary_UCB,
        "c",
        cs,
        n_arms=n_arms,
        n_runs=n_runs,
        steps=steps,
        last_n=last_n,
    )
    gradient_scores = parameter_sweep(
        Nonstationary_Gradient,
        "alpha",
        alphas,
        n_arms=n_arms,
        n_runs=n_runs,
        steps=steps,
        last_n=last_n,
    )

    return {
        "epsilon-greedy (alpha=0.1)": (epsilons, eps_greedy_scores),
        "UCB": (cs, ucb_scores),
        "gradient bandit": (alphas, gradient_scores),
    }


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(0)

    print("=== 図2.2相当の実験(定常, epsilon比較) ===")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    for eps in [0, 0.01, 0.1]:
        avg_reward, pct_optimal = run_experiment_2_2(
            epsilon=eps, n_runs=2000, steps=1000
        )
        label = "greedy (ε=0)" if eps == 0 else f"ε={eps}"
        ax1.plot(avg_reward, label=label)
        ax2.plot(pct_optimal, label=label)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.legend()
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.legend()
    fig1.tight_layout()
    fig1.savefig("figure_2_2_reproduction.png", dpi=150)
    print("Saved plot to figure_2_2_reproduction.png")

    print("=== 練習問題2.11: 非定常ケースのパラメータ研究 ===")
    results = run_experiment_2_11(n_runs=5, steps=200_000, last_n=100_000)
    fig2, ax = plt.subplots(figsize=(8, 5))
    for label, (param_values, scores) in results.items():
        ax.plot(param_values, scores, marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Parameter value (log scale)")
    ax.set_ylabel(f"Average reward (last 100,000 of {200_000} steps)")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig("figure_2_11_nonstationary_parameter_study.png", dpi=150)
    print("Saved plot to figure_2_11_nonstationary_parameter_study.png")
