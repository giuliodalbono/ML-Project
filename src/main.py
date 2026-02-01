from tabular_qlearning import train_tabular, test_tabular
from dqn_agent import train_dqn, test_dqn


def main():
    print("=== Training Tabular Q-Learning ===")
    tab_agent = train_tabular(episodes=50000)
    print("\n=== Testing Tabular Q-Learning ===")
    test_tabular(tab_agent, episodes=100)

    print("\n=== Training DQN ===")
    dqn_agent = train_dqn(episodes=5000)
    print("\n=== Testing DQN ===")
    test_dqn(dqn_agent, episodes=100)


if __name__ == "__main__":
    main()
