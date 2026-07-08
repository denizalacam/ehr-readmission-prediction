from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

from data_pipeline import (
    load_dataset,
    create_X_y,
    split_data,
    create_processed_feature_matrix,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------
# Make directories
# ---------------------------------------------------

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------
# Start of the PyTorch model training script
# ---------------------------------------------------

class ReadmissionDataset(Dataset):

    def __init__(
        self,
        features,
        labels,
    ):
        self.features = features
        self.labels = labels

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, index):
        """
        Return one sample and its label.
        """

        features = self.features[index]
        label = self.labels[index]

        return features, label

class ReadmissionModel(nn.Module):
    """
    Simple neural network for readmission prediction.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(19, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        Define how data flows through the network.
        """
        return self.network(x)

def model_forward_demo():
    """
    Send one batch through the neural network.
    """

    dataset = load_dataset()
    X, y = create_X_y(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = create_processed_feature_matrix(X_train, X_test)

    X_train_tensor = torch.tensor(
        X_train_processed.values,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32,
    )

    train_dataset = ReadmissionDataset(
        X_train_tensor,
        y_train_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

    model = ReadmissionModel()

    features, labels = next(iter(train_loader))

    predictions = model(features)

    # Convert labels from shape [64] to [64, 1]
    labels = labels.unsqueeze(1)

    # Define loss function for binary classification
    loss_fn = nn.BCEWithLogitsLoss()

    # Compute loss using raw logits
    loss = loss_fn(predictions, labels)

    probabilities = torch.sigmoid(predictions)

    print("\nBatch Feature Shape")
    print(features.shape)

    print("\nBatch Label Shape")
    print(labels.shape)

    print("\nPrediction Shape")
    print(predictions.shape)

    print("\nLoss")
    print(loss.item())

    print("\nFirst 5 Raw Predictions")
    print(predictions[:5])

    print("\nFirst 5 Probabilities")
    print(probabilities[:5])


def train_model():
    dataset = load_dataset()
    X, y = create_X_y(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_processed, X_test_processed, preprocessor = (
        create_processed_feature_matrix(X_train, X_test)
    )

    X_train_tensor = torch.tensor(
        X_train_processed.values,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32,
    )

    train_dataset = ReadmissionDataset(
        X_train_tensor,
        y_train_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    model = ReadmissionModel()

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    num_epochs = 10
    loss_history = []
    for epoch in range(num_epochs):
        total_loss = 0.0

        for features, labels in train_loader:
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()

            predictions = model(features)

            loss = loss_fn(predictions, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        loss_history.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Average Loss: {average_loss:.4f}"
        )
    
    evaluate_model(
    model,
    X_test_processed,
    y_test,
    )

    pytorch_metrics = evaluate_model(
        model,
        X_test_processed,
        y_test,
    )

    save_pytorch_model(model)

    save_pytorch_metrics(pytorch_metrics)

    save_loss_curve(loss_history)

def evaluate_model(model, X_test_processed, y_test):
    X_test_tensor = torch.tensor(
        X_test_processed.values,
        dtype=torch.float32,
    )

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32,
    ).unsqueeze(1)

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

    accuracy = accuracy_score(
        y_test_tensor.numpy(),
        predictions.numpy(),
    )

    auc = roc_auc_score(
        y_test_tensor.numpy(),
        probabilities.numpy(),
    )

    print("\nPyTorch Model Evaluation")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y_test_tensor.numpy(),
            predictions.numpy(),
        )
    )

    print("\nClassification Report")
    print(
        classification_report(
            y_test_tensor.numpy(),
            predictions.numpy(),
        )
    )
    
    report = classification_report(
    y_test_tensor.numpy(),
    predictions.numpy(),
    output_dict=True,
    )

    metrics = {
        "Accuracy": accuracy,
        "ROC-AUC": auc,
        "Precision": report["1.0"]["precision"],
        "Recall": report["1.0"]["recall"],
        "F1-Score": report["1.0"]["f1-score"],
    }

    return metrics



def save_pytorch_model(model):
    model_path = MODELS_DIR / "pytorch_mlp.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved PyTorch model to: {model_path}")


def save_pytorch_metrics(metrics):
    output_path = RESULTS_DIR / "pytorch_metrics.txt"

    with open(output_path, "w") as f:
        f.write("DeepEHR PyTorch MLP Results\n")
        f.write("=" * 40 + "\n\n")

        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\nSaved PyTorch metrics to: {output_path}")

def save_loss_curve(loss_history):

    plt.figure(figsize=(8,5))

    plt.plot(loss_history, marker="o")

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Average Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "training_loss.png",
        dpi=300,
    )

    plt.close()

    print("\nSaved training loss curve.")

def dataset_demo():
    """
    Test the custom Dataset.
    """

    dataset = load_dataset()

    X, y = create_X_y(dataset)

    X_train, X_test, y_train, y_test = split_data(X, y)

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = create_processed_feature_matrix(
        X_train,
        X_test,
    )

    X_train_tensor = torch.tensor(
        X_train_processed.values,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32,
    )

    train_dataset = ReadmissionDataset(
        X_train_tensor,
        y_train_tensor,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=False,
        )


    print("\nNumber of Batches")
    print(len(train_loader))

    # Get the first batch
    features, labels = next(iter(train_loader))

    print("\nBatch Feature Shape")
    print(features.shape)

    print("\nBatch Label Shape")
    print(labels.shape)

    print("\nFirst Batch Labels")
    print(labels[:10])
    # print("\nDataset Length")
    # print(len(train_dataset))

    # features, label = train_dataset[0]

    # print("\nFirst Sample Shape")
    # print(features.shape)

    # print("\nFirst Sample")
    # print(features)

    # print("\nFirst Label")
    # print(label)
        
def tensor_from_ehr_data():
    """
    Convert the processed EHR feature matrix into PyTorch tensors.
    """

    # Load the engineered dataset
    dataset = load_dataset()

    # Create features and target
    X, y = create_X_y(dataset)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocess the data
    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = create_processed_feature_matrix(
        X_train,
        X_test,
    )

    # Convert features to tensors
    X_train_tensor = torch.tensor(
        X_train_processed.values,
        dtype=torch.float32,
    )

    # Convert labels to tensors
    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32,
    )

    print("\nFeature Tensor Shape")
    print(X_train_tensor.shape)

    print("\nTarget Tensor Shape")
    print(y_train_tensor.shape)

    print("\nFeature Tensor Type")
    print(X_train_tensor.dtype)

    print("\nTarget Tensor Type")
    print(y_train_tensor.dtype)

    print("\nFirst Training Sample")
    print(X_train_tensor[0])

    print("\nFirst Label")
    print(y_train_tensor[0])


def tensor_from_dataframe_demo():
    """
    Convert our processed training data into PyTorch tensors.
    """

    # Load the engineered dataset
    dataset = pd.read_csv("data/processed/encounter_level_dataset.csv")

    # Reuse the functions we already wrote
    X, y = create_X_y(dataset)

    X_train, X_test, y_train, y_test = split_data(X, y)

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = create_processed_feature_matrix(
        X_train,
        X_test,
    )

    X_train_tensor = torch.tensor(
        X_train_processed.values,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32,
    )

    print("\nFeature Tensor")
    print(X_train_tensor)

    print("\nFeature Tensor Shape")
    print(X_train_tensor.shape)

    print("\nFeature Tensor Type")
    print(X_train_tensor.dtype)

    print("\nTarget Tensor Shape")
    print(y_train_tensor.shape)

    print("\nFirst Feature Row")
    print(X_train_tensor[0])

    print("\nFirst Target")
    print(y_train_tensor[0])





def main():
    
    train_model()
    #model_forward_demo()
    #dataset_demo()
    #tensor_from_ehr_data()


if __name__ == "__main__":
    main()
    