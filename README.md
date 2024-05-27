# DD3D-EIT

Three distinct networks are trained: a VAE $\mathbb{VAE}$, a mapper $\Xi$, and a material classifier $\Upsilon$.

The final architecture of the reconstruction network is defined by

$$
\Gamma := \Xi \circ \Psi : \mathbf{u} \mapsto \mathbf{z} \mapsto \hat{\gamma}
$$

in parallel with the material classification network

$$
\Upsilon : \mathbf{u} \mapsto m
$$

Here, $\mathbf{u}$ represents the EIT data, and $\hat{\gamma}$ is the reconstructed conductivity in the 3D domain by the final reconstruction network architecture."

## Hyperparametertuning $\beta$-VAE

<div style="text-align: center;">
<img src="images/vae_hpt.png" alt="Last 25 VAE hyperparameter tunings with accuracy history of position and volume error (1.5 whisker rule). The three dashed lines mark the three best VAEs, with model 21 being the best." width="600"/>
</div>


## Hyperparametertuning Mapper $\Xi$


| **VAE** | **Iteration** | **Predictable (%)** | **Median volume error (%)** | **Median position error (%)** |
|---------|---------------|---------------------|-----------------------------|-------------------------------|
| 21      | 1             | 92.19               | 0.16                        | 4.14                          |
| 21      | 2             | 90.94               | 0.16                        | 4.31                          |
| 21      | 3             | 92.09               | 0.14                        | 4.39                          |
| 21      | 4             | 92.77               | 0.15                        | 4.12                          |
| 21      | 5             | 54.65               | -0.10                       | 8.73                          |
| 21      | 6             | 92.54               | 0.09                        | 4.72                          |
| 21      | 7             | 92.58               | 0.14                        | 4.58                          |
| 21      | 8             | 96.01               | 0.13                        | 4.55                          |
| 21      | 9             | 90.83               | 0.12                        | 4.44                          |
