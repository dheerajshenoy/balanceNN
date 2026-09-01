#include "BalanceNN.hpp"

#include <cstring>
#include <string>

int
main(int argc, char **argv)
{
    std::string policy_path;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--policy") == 0 && i + 1 < argc)
            policy_path = argv[++i];
    }
    BalanceNN balanceNN(policy_path);
    balanceNN.run();
}
