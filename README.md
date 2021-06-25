# InteL-VAEs
Codes for paper &lt;InteL-VAEs: Adding Inductive Biases to Variational Auto-Encoders via Intermediary Latents>. InteL-VAE is a simple and effective method for learning VAEs with controllable inductive biases by using an intermediary set of latent variables. This allows us to overcome the limitations of the standard Gaussian prior assumption. In particular, it allows us to impose desired properties like sparsity or clustering on learned representations, and incorporate prior information into the learned model.
![Model Graph](https://github.com/NingMiao/InteL-VAEs/blob/main/model.png)

## Usages
### To try low dimensional datasets, 
	run *_low_dim.ipynb in Jupyter notebook.
### To train inteL-VAEs and other baselines,
	run train_*.sh 
Hyper-parameters can be changed in .sh files.
### To run downstream tasks,
	run downstream_*.sh
Please run downstream tasks after training corresponding VAEs.

## Requirements
 - Tensorflow `>= 2.2.0`
 - sklearn (Only for downstream tasks.)
 - Pillow (PIL)
 - fid_score (Only for calculating FID scores.)
