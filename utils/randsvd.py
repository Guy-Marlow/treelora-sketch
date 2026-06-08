import torch
import numpy as np


def rand_svd(M: np.ndarray | torch.Tensor, target_rank: int, oversampling: int):
    """Randomized SVD returning (B_hat, A_hat) such that B_hat @ A_hat ≈ M.

    B_hat: [m, target_rank]
    A_hat: [target_rank, n]
    """
    if isinstance(M, np.ndarray):
        omega = np.random.randn(M.shape[1], target_rank + oversampling)
        Y = M @ omega
        Q = np.linalg.qr(Y).Q
        M_bar = np.transpose(Q) @ M
        U_bar, S, Vh = np.linalg.svd(M_bar)
        S = np.diag(S)
        U = (Q @ U_bar)[:, 0:target_rank]
        S_root = np.power(S, 0.5)[0:target_rank, 0:target_rank]
        return U @ S_root, S_root @ Vh[0:target_rank, :]
    else:
        omega = torch.randn(M.shape[1], target_rank + oversampling, device=M.device)
        Y = M @ omega
        Q, _ = torch.linalg.qr(Y)
        M_bar = Q.t() @ M
        U_bar, S, Vh = torch.linalg.svd(M_bar)
        S = torch.diag(S)
        U = (Q @ U_bar)[:, 0:target_rank]
        S_root = torch.pow(S, 0.5)[0:target_rank, 0:target_rank]
        return U @ S_root, S_root @ Vh[0:target_rank, :]


def main():
    from plot_utils import plot_xy  # only needed for standalone benchmarking

    rows, cols, p = 256, 4096, 2

    torch.random.manual_seed(42)
    mat = torch.randn([rows, cols]).cuda()

    ranks, accs = [], []
    for i in range(rows):
        B_hat, A_hat = rand_svd(mat, i, p)
        acc = torch.linalg.norm(mat - B_hat @ A_hat, ord='fro')
        ranks.append(i)
        accs.append(acc.item())

    plot_xy(ranks, accs, "Target Rank vs. Frob Acc", "Target Rank", "Acc", './plot.png')


if __name__ == "__main__":
    main()