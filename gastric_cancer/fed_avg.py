import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import copy

# ==============================
# 🔹 FEDERATED AVERAGING
# ==============================
def fed_avg(models):
    global_dict = models[0].state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key]).float()

    for model in models:
        local_dict = model.state_dict()
        for key in global_dict.keys():
            global_dict[key] += local_dict[key].float() / len(models)

    return global_dict


# ==============================
# 🔹 LOAD GLOBAL MODEL
# ==============================
global_model = AttentionContrastiveModel(num_experts=2, top_k=1).to(device)
global_model.load_state_dict(fed_avg(local_models))
print("✅ Global model updated with FedAvg")


# ==============================
# 🔹 METRICS STORAGE
# ==============================
validation_losses = []
validation_accuracies = []


# ==============================
# 🔹 METRIC FUNCTION (CLASSIFICATION)
# ==============================
def compute_classification_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    acc = np.mean(preds == labels)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return acc, prec, rec, f1


# ==============================
# 🔹 EVALUATION FUNCTION 
# ==============================
def evaluate_model(model, val_loaders, device):
    model.eval()

    total_loss = 0
    total_samples = 0

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for client, dataloader in val_loaders.items():
            print(f"\n🔍 Evaluating {client}")

            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # ✅ FIX: unpack model output
                features, logits = model(images)

                loss = criterion(logits, labels)
                total_loss += loss.item()

                # ✅ metrics
                acc, prec, rec, f1 = compute_classification_metrics(logits, labels)

                batch_size = labels.size(0)
                total_samples += batch_size

                all_acc.append(acc)
                all_prec.append(prec)
                all_rec.append(rec)
                all_f1.append(f1)

    # ==========================
    # 📊 FINAL METRICS
    # ==========================
    avg_loss = total_loss / len(all_acc)
    avg_acc = np.mean(all_acc)
    avg_prec = np.mean(all_prec)
    avg_rec = np.mean(all_rec)
    avg_f1 = np.mean(all_f1)

    validation_losses.append(avg_loss)
    validation_accuracies.append(avg_acc)

    print("\n📊 GLOBAL VALIDATION RESULTS")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall: {avg_rec:.4f}")
    print(f"F1-score: {avg_f1:.4f}")

    return avg_loss, avg_acc, avg_prec, avg_rec, avg_f1


# ==============================
# 🔹 RUN EVALUATION
# ==============================
global_loss, global_accuracy, global_precision, global_recall, global_f1 = evaluate_model(
    global_model,
    val_loaders,
    device
)

print("\n✅ Final Global Model Performance")
print(f"Loss: {global_loss:.4f}")
print(f"Accuracy: {global_accuracy:.4f}")
print(f"Precision: {global_precision:.4f}")
print(f"Recall: {global_recall:.4f}")
print(f"F1-score: {global_f1:.4f}")
