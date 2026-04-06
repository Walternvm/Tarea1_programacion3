#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cmath>

class Tensor;

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

//Clase Tensor
class Tensor{
private:
    double* data;
    std::vector<size_t> shape;
    size_t total_size;
public:
    //Constructores
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    //Destructor
    ~Tensor();

    //Funciones amigas
    friend class ReLU;
    friend class Sigmoid;
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    //Metodos estaticos
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor random(const std::vector<size_t>& shape, double min, double max);
    static Tensor arange(double start, double end);

    //Metodo apply
    Tensor apply(const TensorTransform& transform) const;

    //Sobrecarga de operadores
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double n) const;

    //View y Unsqueeze
    Tensor view(const std::vector<size_t>& new_shape);
    Tensor unsqueeze(size_t position);

    //Concatenar
    static Tensor concat(const std::vector<Tensor>& tensors, size_t dim);

    //Imprimir
    void print_shape() const;
};

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

#endif //TENSOR_H
