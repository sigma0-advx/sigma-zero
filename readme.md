# σ-zero: Gradient-based Optimization of ℓ0-norm Adversarial Examples

This code is the official PyTorch implementation of the **σ-zero: Gradient-based Optimization of ℓ0-norm Adversarial Examples** (ICLR 2025). 

![Image1](git_images/sigma-zero_optimization.jpg)

The leftmost plot shows an instance of σ-zero’s execution on a two-dimensional problem. The initial point x (red dot) is modified via gradient descent to find the adversarial example x* (green star) while minimizing the number of perturbed features (i.e., the ℓ0 norm of the perturbation). The gray lines surrounding x demarcate regions where the ℓ0 norm is minimized. The rightmost plot shows the adversarial images (top row) and the corresponding perturbations (bottom row) found by σ-zero during the three steps highlighted in the leftmost plot, alongside their prediction and ℓ0 norm.
## Dependencies and Reproducibility

- Python ≥ 3.11.*
- PyTorch ≥ 2.0.*
- torchvision ≥ 0.15.*
- RobustBench ≥ 1.1
- adversarial-library ≥ 0.2.0
- torchattacks ≥ 3.5.1

In order to improve the reproducibility of our experiments, we released our anaconda environment, containing all dependencies and corresponding SW versions. 
The environment can be installed by running the following command: 

```shell
conda env create -f env.yml
```
Once the environment is created, we can use it by typing `conda activate sigmazero`.

## Code Folding

The code is structured as follows: 

- **configs/**, where experimental configurations are stored.
- **data/**, where datasets are downloaded and stored.
- **imagenet/val/**, where the validation set of imagenet needs to be stored.
- **models/**, where models are downloaded and stored.
- **results/**, contains the results of the experiments created by the main script.
- **utils/**, contains wrappers for model and attacks.
- **datasets.py**, used to load datasets.
- **model.py**, used to download models to test.
- **sigma_zero.py**, contains the official implementation of σ-zero attack.
- **utilities.py**, contains useful function to run experiments.
- **main.py**, executes experiment in `{args.config}` on device `{args.device}`.


### Running Experiments 
To execute an experiment where a number of attacks generate adversarial examples for a selected model, firstly a configuration must be created:
```json
{
    "seed": 1233,
    "experiments": [
        {
            "attack": {
                "name": "sigma_zero",
                "params": {
                    "steps":100
                }
            },
            "dataset": "mnist",
            "model": "smallcnn_ddn",
            "n_samples": 100,
            "batch_size": 16
        }
    ]
}
```
Experiments is an array that can contain different attack configurations like the one above.
Experiment are then run by calling the main function:
```shell
python main.py --device=cpu --config=configs/config_single_attack.json
```
After having executed the main function, a folder structure inside **results/**" will be created containing
results, salient statistics and some resulting adversarial images.

### Usage
The σ-zero attack is implemented as a function, so it can be called directly in the following way:
```python
from sigma_zero_attack import sigma_zero
adv_samples = sigma_zero(model=model, inputs=inputs, labels=labels)
```
with required parameters:
- `model`: the model that produces logits with inputs in $[0, 1]$;
- `inputs`: the samples to attack in $[0, 1]$;
- `labels`: the ground-truth labels for the samples.

and more which are optional:
- `steps`: number of iteration steps made by the attack;
- `lr`: learning rate parameter for the optimizer;
- `sigma`: $\sigma$ parameter for the $\ell_0$-norm approximation;
- `threshold`: initial value for the dynamic thresholding;
- `verbose`: flag used to show informations during the optimization process;
- `epsilon_budget`: threshold for the early stopping mechanism to stop the optimization of an adversarial examples when an adversarial examples with a lower perturbation budget than $\epsilon$ is found;
- `grad_norm`: which norm is used to perform the normalization of the gradients.

## Citation

If you use the **$\sigma$-zero**, then consider citing our [paper](https://arxiv.org/pdf/2402.01879) accepted at ICLR 2025:

```bibtex
@inproceedings{cina2024sigma,
  title={$\sigma$-zero: Gradient-based Optimization of 𝓁-norm Adversarial Examples},
  author={Cin{\`a}, Antonio Emanuele and Villani, Francesco and Pintor, Maura and Sch{\"o}nherr, Lea and Biggio, Battista and Pelillo, Marcello},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025},
}
```

## Acknowledgements
The authors would like to thank the contributors of [adversarial-library](https://github.com/jeromerony/adversarial-library), [RobustBench](https://github.com/RobustBench/robustbench) and [Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) for having facilitated the development of this project.

$\sigma$-zero has been partially developed with the support of European Union’s [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu), Horizon Europe, grant agreement No. 101070617, and [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu), Horizon Europe, grant agreement No. 101120393.

<img src="git_images/sec4AI4sec.png" alt="sec4ai4sec" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/elsa.jpg" alt="elsa" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/FundedbytheEU.png" alt="europe" style="width:240px;" />



