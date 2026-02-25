import torch
import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


def export_model(output_path="toy_model.onnx", height=224, width=224):
    model = ToyNet()
    model.eval()

    dummy = torch.randn(1, 3, height, width)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {2: "height", 3: "width"}},
        opset_version=18,
    )

    print(f"Exported: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--out", type=str, default="toy_model.onnx")
    args = parser.parse_args()

    export_model(args.out, args.height, args.width)