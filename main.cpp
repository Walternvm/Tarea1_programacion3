#include "Tensor.h"
#include <ctime>

using namespace std;

int main(){
    srand(time(nullptr));

    Tensor T1 = Tensor::random({1000, 20, 20}, 1.0, 5.0);
    T1.print_shape();

    Tensor T2 = T1.view({1000, 400});
    T2.print_shape();

    Tensor W1 = Tensor::random({400, 100}, 1.0, 5.0);
    Tensor T3_pre = matmul(T2, W1);
    T3_pre.print_shape();

    Tensor b1 = Tensor::random({1, 100}, 1.0, 5.0);
    Tensor ones_exp1 = Tensor::ones({1000, 1});
    Tensor b1_full = matmul(ones_exp1, b1);
    Tensor T3 = T3_pre + b1_full;
    T3.print_shape();

    ReLU relu;
    Tensor T4 = T3.apply(relu);
    T4.print_shape();

    Tensor W2 = Tensor::random({100, 10}, 1.0, 5.0);
    Tensor T5_pre = matmul(T4, W2);
    T5_pre.print_shape();

    Tensor b2 = Tensor::random({1, 10}, 1.0, 5.0);
    Tensor ones_exp2 = Tensor::ones({1000, 1});
    Tensor b2_full = matmul(ones_exp2, b2);
    Tensor T5 = T5_pre + b2_full;
    T5.print_shape();

    Sigmoid sigmoid;
    Tensor T6 = T5.apply(sigmoid);
    T6.print_shape();

    return 0;
}
