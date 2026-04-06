#include "Tensor.h"

using namespace std;

//Constructores
Tensor::Tensor(const vector<size_t>& shape, const vector<double>& values){
    if (shape.empty() || shape.size() > 3){
        throw invalid_argument("El tensor debe tener entre 1 y 3 dimensiones");
    }

    size_t total = 1;
    for (size_t i = 0; i < shape.size(); i++){
        if (shape[i] == 0){
            throw invalid_argument("Las dimensiones no pueden ser cero");
        }
        total *= shape[i];
    }

    if (total != values.size()){
        throw invalid_argument("El numero de valores no coinciden con el shape");
    }

    this->shape = shape;
    this->total_size = total;
    this->data = new double[total_size];

    for (size_t i = 0; i < total_size; i++){
        this->data[i] = values[i];
    }
}

//Copia
Tensor::Tensor(const Tensor& other){
    this->shape = other.shape;
    this->total_size = other.total_size;
    this->data = new double[this->total_size];
    
    for (size_t i = 0; i < this->total_size; i++){
        this->data[i] = other.data[i];
    }
}

//Movimiento
Tensor::Tensor(Tensor&& other) noexcept{
    this->data = other.data;
    this->shape = other.shape;
    this->total_size = other.total_size;

    other.data = nullptr;
    other.shape.clear();
    other.total_size = 0;
}

//Asignacion Copia
Tensor& Tensor::operator=(const Tensor& other){
    if (this != &other){
        delete[] this->data;
        this->shape = other.shape;
        this->total_size = other.total_size;
        this->data = new double[this->total_size];

        for (size_t i = 0; i < this->total_size; i++){
            this->data[i] = other.data[i];
        }
    }
    return *this;
}

//Asignacion Movimiento
Tensor& Tensor::operator=(Tensor&& other) noexcept{
    if (this != &other){
        delete[] this->data;
        this->data = other.data;
        this->shape = other.shape;
        this->total_size = other.total_size;

        other.data = nullptr;
        other.shape.clear();
        other.total_size = 0;
    }
    return *this;
}

//Destructor
Tensor::~Tensor(){
    delete[] data;
}

//Metodo apply
Tensor Tensor::apply(const TensorTransform& transform) const{
    return transform.apply(*this);
}

//Zeros
Tensor Tensor::zeros(const vector<size_t>& shape){
    size_t total = 1;
    for (size_t i = 0; i < shape.size(); i++){
        total *= shape[i];
    }

    vector<double> values(total, 0.0);

    return Tensor(shape, values);
}

//Ones
Tensor Tensor::ones(const vector<size_t>& shape){
    size_t total = 1;
    for (size_t i = 0; i < shape.size(); i++){
        total *= shape[i];
    }

    vector<double> values(total, 1.0);

    return Tensor(shape, values);
}

//Random
Tensor Tensor::random(const vector<size_t>& shape, double min, double max){
    size_t total = 1;
    for (size_t i = 0; i < shape.size(); i++){
        total *= shape[i];
    }

    vector<double> values(total);
    for (size_t i = 0; i < total; i++){
        double r = (double)rand() / (RAND_MAX + 1.0);
        values[i] = min + r * (max - min);
    }

    return Tensor(shape, values);
}

//Arange
Tensor Tensor::arange(double start, double end){
    vector<double> values;
    for (double i = start; i < end; i++){
        values.push_back(i);
    }

    return Tensor({values.size()}, values);
}

//ReLU
Tensor ReLU::apply(const Tensor& t) const{
    vector<double> values(t.total_size);
    for (size_t i = 0; i < t.total_size; i++){
        if (t.data[i] > 0.0){
            values[i] = t.data[i];
        }
        else{
            values[i] = 0.0;
        }
    }

    return Tensor(t.shape, values);
}

//Sigmoid
Tensor Sigmoid::apply(const Tensor& t) const{
    vector<double> values(t.total_size);
    for (size_t i = 0; i < t.total_size; i++){
        values[i] = 1.0/(1.0 + exp(-t.data[i]));
    }

    return Tensor(t.shape, values);
}

//Suma
Tensor Tensor::operator+(const Tensor& other) const{
    if (this->shape != other.shape){
        throw invalid_argument("Las dimensiones deben ser iguales para sumar");
    }

    vector<double> values(this->total_size);

    for (size_t i = 0; i < this->total_size; i++){
        values[i] = this->data[i] + other.data[i];
    }

    return Tensor(this->shape, values);
}

//Resta
Tensor Tensor::operator-(const Tensor& other) const {
    if (this->shape != other.shape) {
        throw invalid_argument("Las dimensiones deben ser iguales para restar");
    }

    vector<double> values(this->total_size);

    for (size_t i = 0; i < this->total_size; i++) {
        values[i] = this->data[i] - other.data[i];
    }

    return Tensor(this->shape, values);
}

//Multiplicacion
Tensor Tensor::operator*(const Tensor& other) const {
    if (this->shape != other.shape) {
        throw invalid_argument("Las dimensiones deben ser iguales para multiplicar");
    }

    vector<double> values(this->total_size);

    for (size_t i = 0; i < this->total_size; i++) {
        values[i] = this->data[i] * other.data[i];
    }

    return Tensor(this->shape, values);
}

//Multiplicacion por un escalar
Tensor Tensor::operator*(double n) const {
    vector<double> values(this->total_size);
    for (size_t i = 0; i < this->total_size; i++) {
        values[i] = this->data[i] * n;
    }

    return Tensor(this->shape, values);
}

//View
Tensor Tensor::view(const vector<size_t>& new_shape){
    if (new_shape.empty() || new_shape.size() > 3) {
        throw invalid_argument("El tensor debe tener entre 1 y 3 dimensiones");
    }

    size_t new_total = 1;
    for (size_t i = 0; i < new_shape.size(); i++) {
        if (new_shape[i] == 0) {
            throw invalid_argument("Las dimensiones no pueden ser cero");
        }
        new_total *= new_shape[i];
    }

    if (new_total != this->total_size) {
        throw invalid_argument("El numero de valores no coincide con el nuevo shape");
    }

    Tensor result(move(*this));
    result.shape = new_shape;

    return result;
}

//Unsqueeze
Tensor Tensor::unsqueeze(size_t position){
    if (this->shape.size() >= 3) {
        throw invalid_argument("No se pueden tener mas de 3 dimensiones");
    }

    if (position > this->shape.size()) {
        throw invalid_argument("Posicion invalida para unsqueeze");
    }

    Tensor result(move(*this));
    result.shape.insert(result.shape.begin() + position, 1);

    return result;
}

//Concatenar
Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) {
        throw invalid_argument("Lista de tensores vacias");
    }

    const vector<size_t>& base_shape = tensors[0].shape;
    if (base_shape.size() > 3) {
        throw invalid_argument("No se pueden tener mas de 3 dimensiones");
    }

    if (dim >= base_shape.size()) {
        throw invalid_argument("Dimension invalida");
    }

    vector<size_t> new_shape = base_shape;
    new_shape[dim] = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        const vector<size_t>& current_shape = tensors[i].shape;
        if (current_shape.size() != base_shape.size()) {
            throw invalid_argument("Todos los tensores deben tener las mismas dimensiones");
        }

        for (size_t j = 0; j < base_shape.size(); j++) {
            if (j != dim && current_shape[j] != base_shape[j]) {
                throw invalid_argument("Las formas de los tensores no coinciden para concatenar");
            }
        }

        new_shape[dim] += current_shape[dim];
    }

    size_t new_total = 1;
    for (size_t i = 0; i < new_shape.size(); i++) {
        new_total *= new_shape[i];
    }

    vector<double> new_values(new_total, 0.0);
    size_t outer_size = 1;
    for (size_t i = 0; i < dim; i++) {
        outer_size *= new_shape[i];
    }

    size_t inner_size = 1;
    for (size_t i = dim + 1; i < new_shape.size(); i++) {
        inner_size *= new_shape[i];
    }

    size_t output_index = 0;
    for (size_t i = 0; i < outer_size; i++) {
        for (size_t j = 0; j < tensors.size(); j++) {
            size_t element_to_copy = tensors[j].shape[dim] * inner_size;
            size_t input_index = i * element_to_copy;
            for (size_t k = 0; k < element_to_copy; k++) {
                new_values[output_index++] = tensors[j].data[input_index + k];
            }
        }
    }

    return Tensor(new_shape, new_values);
}

//Producto Punto
Tensor dot(const Tensor& a, const Tensor& b){
    if (a.shape.size() != 1 || b.shape.size() != 1) {
        throw invalid_argument("El producto punto solo aplica para tensores 1D");
    }

    if (a.shape != b.shape) {
        throw invalid_argument("Los tensores deben tener el mismo tamano");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.total_size; i++) {
        sum += a.data[i] * b.data[i];
    }
    
    return Tensor({1}, {sum});
}

//Multiplicacion de matrices tensores 2D
Tensor matmul(const Tensor& a, const Tensor& b){
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw invalid_argument("Solo soporta tensores de 2 dimensiones");
    }

    size_t rows_a = a.shape[0];
    size_t cols_a = a.shape[1];
    size_t rows_b = b.shape[0];
    size_t cols_b = b.shape[1];

    if (cols_a != rows_b) {
        throw invalid_argument("Dimensiones incompatibles para multiplicacion de matrices");
    }

    vector<double> result_values(rows_a * cols_b, 0.0);

    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_b; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_a; k++) {
                size_t index_a = i * cols_a + k;
                size_t index_b = k * cols_b + j;
                
                sum += a.data[index_a] * b.data[index_b];
            }
            result_values[i * cols_b + j] = sum;
        }
    }

    return Tensor({rows_a, cols_b}, result_values);
}

//Imprimir dimensiones
void Tensor::print_shape() const {
    cout << "Dimensiones: [";
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i < shape.size() - 1) {
            cout << " x ";
        }
    }
    cout << "]" << endl;
}
