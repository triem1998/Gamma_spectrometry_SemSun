# Semi-blind Spectral unmixing based on manifold learning (SemSun)
This is the code for spectral unmixing of gamma-spectrometry with spectral deformation.
Gamma-spectrum can be deformed by the physical phenomena like attenuation, Compton scatering and fluorescence. The database used in this paper contain different spectrum caractiristic for each radionuclide (called spectral signature) based on a sphere of steel with different thickness.
When thickness is varied, the spectral signatures of all radionuclides are deformed. for example, the figure below represents the evolution of spectral signature of $^{133}$Ba in function of thickness.
![alt text](https://github.com/triem1998/Gamma_spectrometry_SemSun/illustrations/spectre_Ba133.PNG?raw=true)

The main idea of this algorithme is to use a Machine learning particulier, called IAE, to model the spectral deformation.
