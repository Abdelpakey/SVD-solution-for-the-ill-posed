# SVD-solution-for-the-ill-posed
## Credit to @Hippogriff

This can happens either matrix is severely ill conditioned or because the singular values are very close or equal to each other.
There are following solution to the problem:

For ill conditioned case, you can compute the condition number of the matrix on cpu and if the condition number is very large, then you cannot do much. In this case, you can simply trivialize the solution.
If the singular values are close to each other, you need to safe guard your back prop, that is, you need to write a new back ward pass. You can use the custom_svd function that replaces the torch's svd function.


    def compute_grad_V(U, S, V, grad_V):
        N = S.shape[0]
        K = svd_grad_K(S)
        S = torch.eye(N).cuda(S.get_device()) * S.reshape((N, 1))
        inner = K.T * (V.T @ grad_V)
        inner = (inner + inner.T) / 2.0
        return 2 * U @ S @ inner @ V.T


    def svd_grad_K(S):
        N = S.shape[0]
        s1 = S.view((1, N))
        s2 = S.view((N, 1))
        diff = s2 - s1
        plus = s2 + s1

    # TODO Look into it
    eps = torch.ones((N, N)) * 10**(-6)
    eps = eps.cuda(S.get_device())
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff

    # gaurd the matrix inversion
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus

    ones = torch.ones((N, N)).cuda(S.get_device())
    rm_diag = ones - torch.eye(N).cuda(S.get_device())
    K = K_neg * K_pos * rm_diag
    return K


    class CustomSVD(Function):
        """
        Costum SVD to deal with the situations when the
        singular values are equal. In this case, if dealt
        normally the gradient w.r.t to the input goes to inf.
        To deal with this situation, we replace the entries of
        a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
        to high value.
        Note: only applicable for the tall and square matrix and doesn't
        give correct gradients for fat matrix. Maybe transpose of the
        original matrix is requires to deal with this situation. Left for
        future work.
        """
        @staticmethod
        def forward(ctx, input):
            # Note: input is matrix of size m x n with m >= n.
            # Note: if above assumption is voilated, the gradients
            # will be wrong.
            try:
                U, S, V = torch.svd(input, some=True)
            except:
                import ipdb; ipdb.set_trace()

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input

customsvd = CustomSVD.apply
