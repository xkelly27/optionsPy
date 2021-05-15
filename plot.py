import matplotlib.pyplot as plt


def custom_plot(table):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.xscale('linear')
    plt.xlabel("Spot Price (Future)")
    plt.yscale('linear')
    plt.ylabel("Return %")

    plt.plot(table)
    plt.show()
