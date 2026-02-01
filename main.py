from tabular_qlearning import train_tabular, test_tabular
from dqn_agent import train_dqn, test_dqn


def main():
    print("=== Training Tabular Q-Learning ===")
    q_table = train_tabular(episodes=50000)
    test_tabular(q_table, episodes=100)

    print("\n=== Training DQN ===")
    dqn_agent = train_dqn(episodes=500)
    test_dqn(dqn_agent, episodes=100)


if __name__ == "__main__":
    main()
