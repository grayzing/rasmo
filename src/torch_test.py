import torch
def main():
    k = torch.randn(8,5)
    print(k)

    j = torch.softmax(k.view(-1), 0).reshape(8,5)
    print(j)

    print(sum(j))

    d = torch.distributions.Categorical(j)
    print(d.sample([1,1]).squeeze())
main()