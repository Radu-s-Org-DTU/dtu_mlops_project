import matplotlib.pyplot as plt


def plot_training_loss(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.title("Training Loss Over Iterations", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/train_losses.png")
    plt.show()


def plot_classification_per_class(correct_classifications, class_names, total_samples):
    colors = ["green", "red", "blue", "purple"]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, correct_classifications, color=colors)
    plt.title("Correct Classifications Per Class", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Correct Classifications", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Add numbers on each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            height - (height * 0.05),          # Slightly below the top of the bar
            f'{int(height)}',                  # Value to display
            ha='center', va='center', color='black', fontsize=20, fontweight='bold'
        )

    # Compute statistics
    correct_samples = sum(correct_classifications)
    incorrect_samples = total_samples - correct_samples
    accuracy = (correct_samples / total_samples) * 100

    # Add summary text below the plot
    figtext = (f'Total Test Samples: {total_samples} | '
               f'Correct Samples: {correct_samples} | '
               f'Incorrect Samples: {incorrect_samples} | '
               f'Accuracy: {accuracy:.2f}%')
    plt.figtext(0.5, -0.05, figtext, horizontalalignment='center', fontsize=10)

    # Save and show the plot
    plt.savefig("reports/figures/correct_classifications_per_class.png", bbox_inches='tight')
    plt.show()
