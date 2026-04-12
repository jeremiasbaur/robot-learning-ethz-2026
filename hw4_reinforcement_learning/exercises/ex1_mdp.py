import copy
import numpy as np


"""
  

1. What is the difference between policy iteration and value iteration in terms of their update procedures?

Policy iteration constantly switches between updating the value function based on the current policy and
then updating the policy based on the new value function by greedily picking the action giving the best value.

Value iteration only updates the value function and at the end of training greedely selects the policy actions based on the value function.
Both algorithms converge to the optimal policy.


2. What happens if the discount factor `gamma` is close to 0 or 1?

If it is close to 0, the future rewards are not really considered because they decay exponentally. With a factor close to 0, it is a one-step lookahead prediction because e.g.
gamma=0.1 becomes already 0.01 after two steps. The policy only optimizes for short-term performance.
If it is close to 1, the future rewards are considered and the policy optimizes also for long-term performance.

3. How does increasing the slip probability (`slip_chance`) affect the optimal policy?

- Compare the cases `slip_chance = 0.0`, `0.01`, and `0.2`.
For 0.0: there is no slip chance and hence the policy moves towards the goal in the fastest way possible.

For 0.01: in the row 2, column 1:5, we now have the optimal value of moving upward instead of sideways in the 0.0 case.
The reason for this is that the risk of falling of the cliff for these six tiles plus in each their downstream "journey" is too high and hence it chooses to go up.

For 0.2: Now all the actions in row 2, column 0:9 point upwards, because the reward of slipping of the cliff is negatively priced in.
Moreover, the row 1, column 0:7 also changed to upward movement for the same reason.  

- Why does the agent tend to behave more conservatively as stochasticity increases?

If it increases, the chance of taking a random action increases and hence the chance to drop of the cliff increases.
Therefore, the policy tries to move away from the cliff such that the chance of falling of the clip is minimized.

"""

class PolicyIteration:
    """
    Policy iteration for a finite tabular MDP.

    Attributes:
        env:
            Environment object. We assume:
                - env.n_states: int
                - env.n_actions: int
                - env.P[s][a] = [(prob, next_state, reward, done)]
        theta: float
            Convergence threshold for policy evaluation.
        gamma: float
            Discount factor.
        v: np.ndarray, shape (n_states,)
            State-value function.
        pi: np.ndarray, shape (n_states, n_actions)
            Stochastic policy. Each row pi[s] is a probability distribution
            over actions at state s.
    """

    def __init__(self, env, theta=1e-3, gamma=0.9):
        """Initialize policy iteration."""
        self.env = env
        self.theta = theta
        self.gamma = gamma

        # Initialize value function to zeros.
        self.v = np.zeros(self.env.n_states, dtype=float)

        # Initialize policy to uniform random.
        self.pi = np.ones((self.env.n_states, self.env.n_actions), dtype=float)
        self.pi /= self.env.n_actions

    def policy_evaluation(self):
        """
        Evaluate the current policy until convergence.

        Input:
            Uses:
                - self.pi: np.ndarray, shape (n_states, n_actions)
                - self.v:  np.ndarray, shape (n_states,)
                - self.env.P

        Output:
            Updates:
                - self.v: np.ndarray, shape (n_states,)
        """
        while True:
            max_diff = 0.0
            new_v = np.zeros_like(self.v)

            for s in range(self.env.n_states):
                qsa_list = []
                for a in range(self.env.n_actions):
                    qsa = 0.0
                    
                    # TODO: compute the updated value of state s under the current policy.
                    # 
                    # Suggested steps:
                    # 1. For each action a, compute the action-value under self.v
                    # 2. Weight q_pi(s, a) by pi[s][a]
                    # 3. Sum over all actions to obtain new_v[s]
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        qsa += (reward + self.gamma*self.v[next_state] * (1 - int(done)))*prob
                    
                    qsa *= self.pi[s, a]
                    qsa_list.append(qsa)

                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v

            if max_diff < self.theta: break


    def policy_improvement(self):
        """
        Improve the current policy greedily with respect to the current value function.

        Input:
            Uses:
                - self.v: np.ndarray, shape (n_states,)
                - self.env.P

        Output:
            Updates and returns:
                - self.pi: np.ndarray, shape (n_states, n_actions)

        We assign equal probability to all greedy actions (ties are allowed).
        """
        for s in range(self.env.n_states):
            qsa_list = []
            for a in range(self.env.n_actions):
                qsa = 0.0
                
                # TODO: compute qsa_list for all actions at state s
                for prob, next_state, reward, done in self.env.P[s][a]:
                    qsa += (reward + self.gamma*self.v[next_state]* (1 - int(done)))*prob
                qsa_list.append(qsa)

            max_q = max(qsa_list)
            num_best_actions = sum(np.isclose(qsa_list, max_q))
            self.pi[s] = [
                1.0 / num_best_actions if np.isclose(q, max_q) else 0.0
                for q in qsa_list
            ]

        return self.pi

    def policy_iteration(self):
        """
        Run policy iteration until the policy no longer changes.

        Input:
            Uses:
                - env.P
                - self.theta
                - self.gamma

        Output:
            v:  np.ndarray, shape (n_states,)
                Final converged value function.
            pi: np.ndarray, shape (n_states, n_actions)
                Final improved policy.

        Main loop:
            1. Policy evaluation
            2. Policy improvement
            3. Stop when the policy is unchanged
        """
        while True:
            old_pi = copy.deepcopy(self.pi)

            self.policy_evaluation()
            new_pi = self.policy_improvement()

            if np.allclose(old_pi, new_pi):
                break

        return self.v, self.pi


class ValueIteration:
    """
    Value iteration for a finite tabular MDP.

    Attributes:
        env:
            Environment object with tabular transition model env.P.
        theta: float
            Convergence threshold.
        gamma: float
            Discount factor.
        v: np.ndarray, shape (n_states,)
            State-value function.
        pi: np.ndarray, shape (n_states, n_actions)
            Greedy policy extracted from the converged value function.
    """

    def __init__(self, env, theta=1e-3, gamma=0.9):
        """Initialize value iteration."""
        self.env = env
        self.theta = theta
        self.gamma = gamma

        self.v = np.zeros(self.env.n_states, dtype=float)
        self.pi = np.zeros((self.env.n_states, self.env.n_actions), dtype=float)

    def value_iteration(self):
        """
        Run value iteration until convergence.

        Input:
            Uses:
                - self.v: np.ndarray, shape (n_states,)
                - self.env.P

        Output:
            Returns:
                - v:  np.ndarray, shape (n_states,)
                - pi: np.ndarray, shape (n_states, n_actions)
        """
        while True:
            max_diff = 0.0
            new_v = np.zeros_like(self.v)

            for s in range(self.env.n_states):
                qsa_list = []
                for a in range(self.env.n_actions):
                    qsa = 0.0
                    
                    # TODO: compute all action-values Q(s, a)
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        qsa += (reward + self.gamma*self.v[next_state]* (1 - int(done)))*prob
                    
                    qsa_list.append(qsa)

                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v
            
            if max_diff < self.theta:
                break

        self.get_policy()
        return self.v, self.pi

    def get_policy(self):
        """
        Extract a greedy policy from the converged value function.

        Input:
            Uses:
                - self.v: np.ndarray, shape (n_states,)

        Output:
            Updates:
                - self.pi: np.ndarray, shape (n_states, n_actions)
        """
        for s in range(self.env.n_states):
            qsa_list = []
            for a in range(self.env.n_actions):
                qsa = 0.0
                # TODO: compute qsa_list for all actions
                for prob, next_state, reward, done in self.env.P[s][a]:
                    qsa += (reward + self.gamma*self.v[next_state]* (1 - int(done)))*prob
                
                qsa_list.append(qsa)

            max_q = max(qsa_list)
            num_best_actions = sum(np.isclose(qsa_list, max_q))
            self.pi[s] = [
                1.0 / num_best_actions if np.isclose(q, max_q) else 0.0
                for q in qsa_list
            ]