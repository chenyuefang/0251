import seaborn as sns
import matplotlib.pyplot as plt


def show_cost(cost):
    sns.set_style("whitegrid")
    plt.plot(cost)
    plt.show()
