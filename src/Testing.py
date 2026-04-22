from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import torch
import matplotlib.pyplot as plt

from Downstream_model import LinearClassifier
from Model import SSL
from Data import testloader,trainloader


def evaluate(model, testloader, device, class_names=None):
    model = model.to(device)
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for _, (x, label) in enumerate(testloader):
            x, label = x.to(device), label.to(device)
            pred      = model(x)
            predicted = pred.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.squeeze().cpu().numpy())

    # ── Accuracy ──────────────────────────────────────────
    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {acc:.2f}%\n")

    # ── Classification Report ─────────────────────────────
    cr = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"Classification Report:\n {cr}")

    # ── Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("../Confusion_Matrix.png", dpi=150)
    plt.show()


if __name__=="__main__":
    main_model = SSL()
    # main_chkpt = torch.load("../checkpoints/ssl_chkpt.pth")
    # main_model.load_state_dict(main_chkpt["model_state_dict"])
    encoder = main_model.encoder

    test_model = LinearClassifier(encoder,num_classes=7)
    down_chkpt = torch.load("../checkpoints/down_chkpt.pth")
    test_model.load_state_dict(down_chkpt["model_state_dict"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")
    classnames = ['actinic keratoses and intraepithelial carcinoma', 'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma',  'melanoma', 'melanocytic nevi',  'vascular lesions']
    short_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
    evaluate(test_model,testloader,device,classnames)