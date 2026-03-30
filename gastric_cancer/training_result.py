print("\n📊 FINAL METRICS FOR ALL CLIENTS\n")

for client, metrics in all_client_metrics.items():
    print(f"🔹 {client}")

    print(f"  Losses      : {np.round(metrics['losses'], 4)}")
    print(f"  Accuracies  : {np.round(metrics['accuracies'], 4)}")
    print(f"  Precisions  : {np.round(metrics['precisions'], 4)}")
    print(f"  Recalls     : {np.round(metrics['recalls'], 4)}")
    print(f"  F1 Scores   : {np.round(metrics['f1_scores'], 4)}")

    print("-" * 50)


import matplotlib.pyplot as plt

for client_id, metrics in all_client_metrics.items():
    num_epochs = len(metrics["losses"])
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(12, 6))

    # Plot Loss and Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics["losses"], label="Loss", color="red")
    plt.plot(epochs, metrics["accuracies"], label="Accuracy", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Loss & Accuracy")
    plt.legend()

    # Plot Precision, Recall, F1, Dice, Confidence
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics["precisions"], label="Precision", color="green")
    plt.plot(epochs, metrics["recalls"], label="Recall", color="purple")
    plt.plot(epochs, metrics["f1_scores"], label="F1-Score", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Other Metrics")
    plt.legend()

    plt.suptitle(f"Training Metrics for {client_id}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
