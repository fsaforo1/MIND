# MIND
An implementation of MIND

- [Inductive Mutual Information Estimation, A Convex Maximum-Entropy Copula Approach](https://arxiv.org/abs/2102.13182) by Yves-Laurent Kom Samo 


## CopulaBatchGenerator

The `CopulaBatchGenerator` class generates batches of copula samples for training a maximum-entropy copula model. It ensures that the generated samples are diverse and representative of the underlying copula structure in the input data. The class is designed to be used as a data generator during the training process.

## CopulaBatchGenerator
The CopulaLearner class encapsulates the logic for training a maximum-entropy copula model. It handles the setup of the copula model, optimizer, loss function, and training process. By calling the fit method with input data, the user can train the copula model efficiently, with the flexibility to specify various training parameters. The class is designed to provide a convenient interface for training copula models for specific datasets.



## The Mutual Information Objective

- **Objective Function:** The `MINDLoss` class implements a custom loss function that calculates the mutual information between two random variables using the copula representation. The loss function is defined as: 

  $$\text{MIND Loss} = -E_P(T(x, y)^T\theta) + \log E_Q(e^{T(x, y)^T\theta})$$

  Where:
  - $T(x, y)$ represents copula samples generated from the copula model.
  - $\theta$ represents parameters of the copula model.
  - $E_P$ and $E_Q$ represent expectations under distributions $P$) and $Q$, respectively.

- **Usage in CopulaLearner:**
  - The `MINDLoss` class is utilized in the `CopulaLearner` during the compilation of the copula model. When the copula model is compiled, the `MINDLoss` instance is set as the loss function. This means that during training, the copula model aims to minimize the mutual information calculated by the `MINDLoss` function.

**CopulaLearner Class:**

- **Initialization and Compilation:**
  - In the `CopulaLearner` class, an instance of the `MINDLoss` class is created and set as the loss function for the copula model during the initialization.
  - The copula model is compiled using the Adam optimizer and the `MINDLoss` as the loss function.

- **Training Process:**
  - During training (when the `fit` method of `CopulaLearner` is called), the copula model learns to minimize the mutual information calculated by the `MINDLoss` class. The copula samples generated during the training process are used to compute the mutual information, and the model's parameters are adjusted to minimize this value.
  - The copula model learns to capture the dependencies between variables in the dataset in such a way that the mutual information, as calculated by the `MINDLoss`, is minimized. This leads to the copula model effectively capturing the underlying relationships in the data.


In summary, the `MINDLoss` class provides the specific loss function that quantifies the mutual information in the copula model. The `CopulaLearner` class utilizes this loss function during the training process to guide the copula model's parameter updates, ensuring that the model learns to represent the dependencies in the data as captured by the `MINDLoss`. The combination of these two classes enables the effective training of a copula model for capturing the complex relationships in the given dataset.

---