# NOTE: Classes below are copied from 4_optim-student.ipynb
# and consolidated into this module for reuse.

class Tensor:
    
    """ stores a single scalar Tensor and its gradient """

    def __init__(self, data, _children=(), _op=''):

        self.data = data
        self.grad = 0.0

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        out._prev = set([self, other])
        return out

    def __mul__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, [self, other], '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):

        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
            
        out._backward = _backward

        return out

    def relu(self):
        # FIXME: implement relu
        out = Tensor(self.data if self.data > 0 else 0, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def build_topo(self, visited=None, topo=None):
        if self not in visited:
            visited.add(self)
            for child in self._prev:
                child.build_topo(visited=visited, topo=topo)
            topo.append(self)
        return topo

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        topo = self.build_topo(topo=topo, visited=visited)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        pass

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0


class SGD(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate

    def step(self):
        for param in self.params:
            param.data -= self.learning_rate * param.grad

class RMSProp(Optimizer):
    def __init__(self, params, learning_rate=0.01, decay=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.decay = decay
        self.cache = [Tensor(0.0) for _ in params]

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode RMSProp.
        La méthode RMSProp (Root Mean Square Propagation) est une méthode d'optimisation adaptative 
        qui divise le taux d'apprentissage par une moyenne exponentielle glissante des carrés des gradients passés.
        """
        for p, c in zip(self.params, self.cache):
            c.data = self.decay * c.data + (1 - self.decay) * p.grad**2
            p.data -= (self.learning_rate * p.grad) / (np.sqrt(c.data) + 1e-8)

class Adagrad(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.cache = [Tensor(0.0) for _ in params]

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode Adagrad.
        La méthode Adagrad (Adaptive Gradient) est une méthode d'optimisation adaptative 
        qui divise le taux d'apprentissage par la racine carrée de la somme cumulée des carrés des gradients passés.
        """
        for p, c in zip(self.params, self.cache):
            c.data += p.grad**2
            p.data -= (self.learning_rate * p.grad) / (np.sqrt(c.data) + 1e-8)

class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0 for _ in params]
        self.v = [0.0 for _ in params]
        self.t = 0

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode Adam.
        La méthode Adam (Adaptive Moment Estimation) est une méthode d'optimisation adaptative 
        qui utilise une moyenne exponentielle glissante des gradients passés et de leurs carrés 
        pour estimer la première et la deuxième moments des gradients.
        """
        self.t += 1

        for p, m, v in zip(self.params, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * p.grad
            v = self.beta2 * v + (1 - self.beta2) * p.grad**2

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            p.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

class AdamW(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [0.0 for _ in params]
        self.v = [0.0 for _ in params]
        self.t = 0

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode AdamW.
        La méthode AdamW (Adaptive Moment Estimation with Weight Decay) est une variante de la méthode Adam 
        qui ajoute une régularisation L2 (ou weight decay) aux mises à jour des paramètres.
        """
        # FIXME
        pass


class Momentum(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [Tensor(0.0) for _ in params]

    def step(self):
        # FIXME
        pass


class Adadelta(Optimizer):
    def __init__(self, params, rho=0.95, eps=1e-8):
        super().__init__(params)
        self.rho = rho
        self.eps = eps
        self.cache = [0.0 for _ in params]
        self.delta = [0.0 for _ in params]

    def step(self):
        """
        Mise a jour des parametres en utilisant la methode Adadelta.
        Adadelta adapte automatiquement le taux d'apprentissage par
        une moyenne exponentielle des carres des gradients et des deltas.
        """
        for i, p in enumerate(self.params):
            self.cache[i] = self.rho * self.cache[i] + (1 - self.rho) * (p.grad ** 2)
            update = - (np.sqrt(self.delta[i] + self.eps) / (np.sqrt(self.cache[i] + self.eps))) * p.grad
            self.delta[i] = self.rho * self.delta[i] + (1 - self.rho) * (update ** 2)
            p.data += update

class LRScheduler:
    def __init__(self, optimizer, initial_lr):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.iteration = 0  # Ou self.epoch = 0 si basé sur les époques
    
    def update_lr(self, metrics=None):
        self.iteration += 1
        self.lr = self.initial_lr / (1 + 0.01 * self.iteration) 

    def step(self, metrics=None):
        self.update_lr(metrics)
        self.optimizer.learning_rate = self.lr

class LRSchedulerOnPlateau(LRScheduler):
    def __init__(self, optimizer, initial_lr, patience=10, factor=0.1, min_lr=1e-6, mode='min', threshold=1e-4):
        super().__init__(optimizer, initial_lr)
        self.patience = patience  # Nombre d'époques sans amélioration avant de réduire le taux
        self.factor = factor  # Facteur de réduction du taux d'apprentissage
        self.min_lr = min_lr  # Valeur minimale du taux d'apprentissage
        self.mode = mode  # 'min' : réduire le taux quand la métrique cesse de diminuer, 'max' : inverse
        self.threshold = threshold  # Seuil pour déterminer la réduction de la métrique
        self.best_metric = float('inf') if mode == 'min' else float('-inf')  # Meilleure métrique observée
        self.num_bad_epochs = 0  # Nombre d'époques sans amélioration de la métrique

    def update_lr(self, metric):
        if self.mode == 'min':
            if metric < self.best_metric - self.threshold:
                self.best_metric = metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        else:
            if metric > self.best_metric + self.threshold:
                self.best_metric = metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            print("Reducing learning rate.")
            self.lr = max(self.lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0
            self.optimizer.learning_rate = self.lr
