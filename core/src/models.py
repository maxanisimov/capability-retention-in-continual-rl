import torch


def get_mnist_model(
    seed=42,
    device="cpu",
    output_dim=10,
    head=None,
    hidden_dim=128,
    dense_layers=1,
    final_bias=True,
):
    """Get a simple CNN model."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
    )
    if dense_layers == 1:
        model.append(torch.nn.Linear(576, output_dim, bias=final_bias))
    else:
        model.append(torch.nn.Linear(576, hidden_dim))
        model.append(torch.nn.ReLU())
        for _ in range(dense_layers - 2):
            model.append(torch.nn.Linear(hidden_dim, hidden_dim))
            model.append(torch.nn.ReLU())
        model.append(torch.nn.Linear(hidden_dim, output_dim, bias=final_bias))

    if head is not None:
        model.append(head)
    return model.to(device)


def get_fully_connected_model(
    input_dim,
    seed=0,
    device="cuda",
    output_dim=10,
    head=None,
    hidden_dim=128,
    dense_layers=1,
    final_bias=True,
):
    torch.manual_seed(seed)
    model = torch.nn.Sequential()
    model.append(torch.nn.Flatten())
    if dense_layers == 1:
        model.append(torch.nn.Linear(input_dim, output_dim, bias=final_bias))
    else:
        model.append(torch.nn.Linear(input_dim, hidden_dim))
        model.append(torch.nn.ReLU())
        for _ in range(dense_layers - 2):
            model.append(torch.nn.Linear(hidden_dim, hidden_dim))
            model.append(torch.nn.ReLU())
        model.append(torch.nn.Linear(hidden_dim, output_dim, bias=final_bias))

    if head is not None:
        model.append(head)
    return model.to(device)
