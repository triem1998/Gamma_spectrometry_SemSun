# Semi-blind Spectral unmixing based on manifold learning (SemSun)
This is the code for spectral unmixing in gamma-ray spectrometry with spectral deformation.
The gamma-ray spectrum can be deformed by physical phenomena such as attenuation, the Compton scattering and fluorescence. The database used in this article contains different characteristic spectra for each radionuclide (called spectral signatures) based on different thicknesses of a steel sphere.
As the thickness varies, the spectral signatures of all radionuclides are deformed. For example, the figure below shows the spectral signature of <sup>133</sup>Ba as a function of thickness.
![ ](illustrations/spectre_Ba133.png)

The main idea behind this algorithm is to use a particular machine learning model, called IAE, to model spectral deformation. The key to this method is to use nonlinear interpolation of predefined points (called anchor points) to describe the spectral signature.
![ ](illustrations/iae_schema.PNG)
In this work, the IAE model is based on the CNN architecture. Two versions of the IAE model are proposed: the individual model that learns independently for each radionuclide, and a joint model that captures correlations between spectral variability for all radionuclides. The file [IAE_CNN_joint_gamma_spectrometry.ipynb](notebooks/IAE_CNN_joint_gamma_spectrometry.ipynb) is an example of using IAE to learn spectral deformations. 

When the IAE model is already trained to capture the shape and variability of the spectral signature of all radionuclides, it is included in an unmixing procedure as constraints on the spectral signatures. Based on an observed spectrum, the hybrid spectral unmixing jointly estimates the spectral signatures and the counting vector according to the likelihood function of Poisson distribution.
The notebook file [Evaluation_unmixing_gamma_spectrometry.ipynb](notebooks/Evaluation_unmixing_gamma_spectrometry.ipynb) explains how to use this code for spectral unmixing.


The code is organized as follows:
-  The Code folder contains the source code for the [IAE](codes/IAE_CNN_TORCH_Oct2023.py) and the [hybrid spectral unmixing algorithm](codes/unmixing_optim_constraint_CNN_joint.py)
-  The Data folder contains the dataset of 96 spectral signatures of 4 radionuclides: <sup>57</sup>Co, <sup>60</sup>Co,<sup>133</sup>Ba and <sup>137</sup>Cs as a function of steel thickness.
-  The Notebooks folder contains two jupyter notebook files for training an IAE model and using SemSun to estimate the spectral signature and counting
      - The Models folder contains the pre-trained IAE model.
      - The Data folder contains the results of the evaluation of 1000 Monte Carlo simulations for the hybrid algorithm.
## Package requirements
The codes are implemented using Pytorch. Please follow this link to install it correctly: [Pytorch (Cuda or CPU)](https://pytorch.org/get-started/locally/)
