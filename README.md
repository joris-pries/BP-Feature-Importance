# The bp_feature_importance package

bp_feature_importance is a Python package for determining the Berkelmans-Pries Feature Importance.

## Paper

This package is an implementation of the ideas from 'The Berkelmans-Pries Feature Importance Method: A Generic Measure of Informativeness of Features', where we introduce a new feature importance function, which improves explainability. In this paper, we first discuss what properties an ideal dependency should have. Then, it is shown that no commonly used dependency function satisfies these requirements. We introduce a new dependency function and prove that it does satisfy all requirements, making it an ideal candidate to use as dependency function.

### Citation

If you have used the bp_feature_importance package, please also cite: https://arxiv.org/abs/2301.04740

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package

```bash
pip install bp_feature_importance
```

---

### Windows users

```bash
python -m pip install bp_feature_importance
```

<!-- ```bash
python -m pip install bp_dependency
``` -->

or

```bash
py -m pip install bp_feature_importance
```

<!-- ```bash
py -m pip install bp_dependency
``` -->

## How to use:

This package can be used to determine the Berkelmans-Pries Feature Importance for a given dataset. For the theoretical formulation, see our paper. This package provides one main function: `bp_feature_importance`. We will now explain this function.

### bp_feature_importance

This function is used to determine the Berkelmans-Pries Feature Importance.

```bash
from bp_feature_importance import bp_feature_importance
```

#### Input

* `dataset` (array_like): MxK array containing M samples of K variables.
* `X_indices` (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
* `Y_indices` (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
* `binning_indices` (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
* `binning_strategy` (dictionary or number or str, optional): Default is `auto`. See numpy.histogram_bin_edges. Input a dictionary if for each binning index a specific strategy should be applied.
* `midway_binning` (bool, optional): Determines if the dataset is binned using the index of the bin (False) or the midway of the bin (True). Default is False.
* sequence_strategy (str): Default is `exhaustive`, which samples all possible sequences. Alternative option is `random`, which samples a random sequence every time
* stopping_strategy (int): Default is `None`. Used to stop earlier after x rounds.
* print_stats (bool): Default is `False`. Print statistics

#### Output

The function `bp_feature_importance` gives the following output:

* `dict`: The Berkelmans-Pries Feature Importance for each index in X_indices

#### Example

Let the dataset be given by:

| $X_0$ | $X_1$ | $Y$ |
| :-----: | ------- | :---: |
|    0    | 0       |   0   |
|    1    | 0       |   1   |
|    0    | 1       |   1   |
|    1    | 1       |   0   |

where each row is as likely to be drawn. We now determine the feature importance values using:

```python
import numpy as np
dataset, X_indices, Y_indices = (np.array([[0,0,0], [1,0,1], [0,1,1], [1,1,0]]), [0,1], [2])
print(bp_feature_importance(dataset, X_indices, Y_indices))
```

with output:

```python
{0: 0.5, 1: 0.5}
```

This output is desirable, as $Y$ is the XOR function of $X_0$ and $X_1$

#### Warning

When $Y$ is constant (either immediately or by binning), the feature importance is undefined and NaN is returned with a warning. As example:

```python
dataset, X_indices, Y_indices = (np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]), [0,1], [2])
print(bp_feature_importance(dataset, X_indices, Y_indices))
```

with output

```
nan
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
