from sklearn.manifold import TSNE
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt

from Downstream_model import LinearClassifier
from Model import SSL,Encoder
from Data import testloader

def plot_tsne(encoder, dataloader, device, class_names=None):
    encoder.eval()
    all_h      = []
    all_labels = []
    encoder = encoder.to(device)

    with torch.no_grad():
        for _, (x, label) in enumerate(dataloader):
            x = x.to(device)
            h = encoder(x)                          # extract representations
            all_h.extend(h.cpu().numpy())
            all_labels.extend(label.squeeze().numpy())

    all_h      = np.array(all_h)
    all_labels = np.array(all_labels)

    # ── Run t-SNE ─────────────────────────────────────────
    print("Running t-SNE...")
    tsne      = TSNE(n_components=2, perplexity=30, random_state=42)
    h_2d      = tsne.fit_transform(all_h)

    # ── Plot ──────────────────────────────────────────────
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(np.unique(all_labels)))
    
    for idx, cls in enumerate(np.unique(all_labels)):
        mask = all_labels == cls
        label_name = class_names[idx] if class_names else f"Class {cls}"
        plt.scatter(h_2d[mask, 0], h_2d[mask, 1],
                    label=label_name, alpha=0.6, s=15, color=palette[idx])

    plt.title("t-SNE of Encoder Representations")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("../tSNE.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    classnames = ['actinic keratoses and intraepithelial carcinoma', 'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma',  'melanoma', 'melanocytic nevi',  'vascular lesions']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_model = SSL()
    encoder = main_model.encoder
    test_model = LinearClassifier(encoder,num_classes=7)
    down_chkpt = torch.load("../checkpoints/down_chkpt.pth")
    test_model.load_state_dict(down_chkpt["model_state_dict"])

    
    
    plot_tsne(encoder, testloader, device, class_names=classnames) 