import json
from pathlib import Path

def fix_notebook():
    notebook_path = Path("e:/image/code.ipynb")
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist.")
        return
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # Cell 1: imports and sys.path
    cell_1_src = [
        "import sys\n",
        "from pathlib import Path \n",
        "import matplotlib.pyplot as plt \n",
        "import torch\n",
        "from PIL import Image\n",
        "from torchvision import transforms\n",
        "\n",
        "# Add utils to sys.path to import models\n",
        "ROOT = Path.cwd()\n",
        "sys.path.append(str(ROOT / \"NST_Code\" / \"utils\"))\n",
        "from models import VGGEncoder"
    ]
    nb["cells"][0]["source"] = cell_1_src
    
    # Cell 2: correct typos in paths, filenames and torch.cuda.is_available
    cell_2_src = [
        "ROOT = Path.cwd()\n",
        "EXAMPLES = ROOT / \"NST_Code\" / \"examples\"\n",
        "VGG_WEIGHTS = ROOT / \"NST_Code\" / \"vgg_normalised.pth\"\n",
        "\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "image_size = 512\n",
        "\n",
        "image_paths = {\n",
        "    \"Original Brad Pitt\": EXAMPLES / \"brad_pitt.jpg\",\n",
        "    \"Stylized Brad Pitt A\": EXAMPLES / \"stylized_brad_pitt.jpg\", \n",
        "    \"Stylized Brad Pitt B\": EXAMPLES / \"stylized_brad_pitt (1).jpg\"\n",
        "}\n",
        "\n",
        "transform = transforms.Compose([\n",
        "    transforms.Resize((image_size, image_size)),\n",
        "    transforms.ToTensor(),\n",
        "])\n",
        "\n",
        "encoder = VGGEncoder(str(VGG_WEIGHTS), test=False).to(device).eval()\n",
        "\n",
        "print(\"Using device:\", device)"
    ]
    nb["cells"][1]["source"] = cell_2_src
    
    # Cell 3: correct syntax for encoder call and underscores for relu keys
    cell_3_src = [
        "def load_image(path):\n",
        "    image = Image.open(path).convert(\"RGB\")\n",
        "    tensor = transform(image).unsqueeze(0).to(device)\n",
        "    return image, tensor\n",
        "\n",
        "\n",
        "def show_images(paths_dict):\n",
        "    fig, axes = plt.subplots(1, len(paths_dict), figsize=(16, 5))\n",
        "    for ax, (title, path) in zip(axes, paths_dict.items()):\n",
        "        image = Image.open(path).convert(\"RGB\")\n",
        "        ax.imshow(image)\n",
        "        ax.set_title(title)\n",
        "        ax.axis(\"off\")\n",
        "    plt.tight_layout()\n",
        "\n",
        "\n",
        "def extract_features(tensor):\n",
        "    with torch.no_grad():\n",
        "        h1, h2, h3, h4 = encoder(tensor)\n",
        "        return {\n",
        "            \"relu1_1\": h1,\n",
        "            \"relu2_1\": h2,\n",
        "            \"relu3_1\": h3,\n",
        "            \"relu4_1\": h4,\n",
        "        }\n",
        "\n",
        "\n",
        "def activation_map(feature_tensor):\n",
        "    activation = feature_tensor[0].mean(dim=0).detach().cpu()\n",
        "    activation = activation - activation.min()\n",
        "    activation = activation / (activation.max() + 1e-8)\n",
        "    return activation"
    ]
    nb["cells"][2]["source"] = cell_3_src
    
    # Cell 5: correct the layers_to_show name ("relul_1" -> "relu1_1") and nest the plotting loops properly
    cell_5_src = [
        "layers_to_show = [\"relu1_1\", \"relu2_1\", \"relu3_1\", \"relu4_1\"]\n",
        "row_labels = [\"Input Image\"] + layers_to_show\n",
        "num_rows = len(row_labels)\n",
        "num_cols = len(image_paths)\n",
        "\n",
        "fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 3.6 * num_rows))\n",
        "\n",
        "for col, (name, path) in enumerate(image_paths.items()):\n",
        "    image = Image.open(path).convert(\"RGB\")\n",
        "    axes[0, col].imshow(image)\n",
        "    axes[0, col].set_title(name, fontsize=13)\n",
        "    axes[0, col].axis(\"off\")\n",
        "    \n",
        "    for row, layer in enumerate(layers_to_show, start=1):\n",
        "        axes[row, col].imshow(activation_map(feature_bank[name][layer]))\n",
        "        axes[row, col].axis(\"off\")\n",
        "\n",
        "for row, label in enumerate(row_labels):\n",
        "    axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=18)\n",
        "    \n",
        "plt.suptitle(\"VGG Content Features Across Original and Stylized Brad Pitt Images\", fontsize=16, y=1.02)\n",
        "plt.tight_layout()"
    ]
    nb["cells"][4]["source"] = cell_5_src

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully fixed code.ipynb!")

if __name__ == "__main__":
    fix_notebook()
