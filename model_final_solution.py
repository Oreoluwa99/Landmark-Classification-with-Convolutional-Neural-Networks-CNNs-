import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:
        super().__init__()
        self.my_model = nn.Sequential(
            nn.Conv2d(in_channels=3, kernel_size=3, padding=1, out_channels=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, kernel_size=3, padding=1, out_channels=32),            
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, kernel_size=3, padding=1, out_channels=64),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            nn.Conv2d(in_channels=64, kernel_size=3, padding=1, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=128, kernel_size=3, padding=1, out_channels=256),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, out_features=1024),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, out_features=512),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.my_model(x)
    

######################################################################################
#                                     TESTS
######################################################################################
import pytest
#from your_module import MyModel  # Import your custom model


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders  # Assuming you have a data module
    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel()

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor"

    assert out.shape[-1] == 50, f"Expected an output tensor with last dimension 50, got {out.shape[-1]}"
