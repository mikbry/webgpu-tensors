import torch

def main():
    # Matrix multiplication
    x = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0],
                      [2.0, 2.0, 2.0, 2.0],
                      [3.0, 3.0, 3.0, 3.0]])
    y = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0],
                      [2.0, 2.0, 2.0, 2.0],
                      [3.0, 3.0, 3.0, 3.0]])
    print("Matrix multiplication:")
    print("x:")
    print(x)
    print("y:")
    print(y)
    result = torch.matmul(x, y)
    print("Result:")
    print(result)

    # Maximum element-wise with 0
    x = torch.tensor([[-3.0, -2.0, -1.0, 0.0],
                      [0.0, 1.0, 2.0, 3.0],
                      [0.0, 1.0, 2.0, 3.0],
                      [0.0, 1.0, 2.0, 3.0]])
    max_result = torch.maximum(x, torch.tensor(0.0))
    print("\nMaximum (element-wise) with 0:")
    print(max_result)

    # Test for maximum function
    test_tensor = torch.tensor([[-3.0, -2.0, -1.0, 0.0],
                                [0.0, 1.0, 2.0, 3.0],
                                [0.0, 1.0, 2.0, 3.0],
                                [0.0, 1.0, 2.0, 3.0]])
    mul = torch.matmul(test_tensor, x)
    max_result = torch.maximum(mul, torch.tensor(0.0))
    print("\nTest maximum:")
    print(max_result)

    # GPU tensor (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(5, 3, device=device)
    print(f"\nGPU tensor (on {device}):")
    print(x)

if __name__ == "__main__":
    main()
