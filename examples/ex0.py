import torch

def main():
    # Create a random tensor
    x = torch.rand(5, 3)
    print("Random tensor:")
    print(x)

    # Create a tensor of ones
    x = torch.ones(5, 3)
    print("\nTensor of ones:")
    print(x)

    # Create a tensor from a list
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    print("\nTensor from list:")
    print(x)

    # Create a 2D tensor
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 4.0, 5.0, 6.0]])
    print("\n2D tensor:")
    print(x)

    # Create a 3D tensor
    x = torch.tensor([[[0, 1], [2, 3]], [[3, 4], [5, 6]]])
    print("\n3D tensor:")
    print(x)

    # Create a 4D tensor
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 2, 2, 1)
    print("\n4D tensor:")
    print(x)

    # Print tensor properties
    tensor = torch.rand(3, 4)
    print("\nTensor properties:")
    print(f"Shape: {tensor.shape}")
    print(f"Datatype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Tensor:\n{tensor}")

    # Create an empty tensor and print its size
    x = torch.empty(3, 4, 5)
    print(f"\nSize of empty tensor: {x.numel()}")
    print(f"Size of dimension 1: {x.size(1)}")

if __name__ == "__main__":
    main()
