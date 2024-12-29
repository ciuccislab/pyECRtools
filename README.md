# pyECRtools: A Python Toolbox for Electrical Conductivity Relaxation (ECR) Analysis

**pyECRtools** is a Python translation of the original MATLAB-based ECRtools (available at https://github.com/ciuccislab/ECRtools/). This Python implementation provides the same powerful functionality for analyzing Electrical Conductivity Relaxation (ECR) data. Developed by Francesco Ciucci at the University of Bayreuth, this toolbox provides comprehensive tools for researchers working with ECR measurements.

## Key Features

* **Data Analysis:** Tools for plotting and analyzing ECR experimental data
* **Parameter Extraction:** Extract surface exchange coefficients (k) and diffusion coefficients (D) from ECR measurements
* **Statistical Analysis:** Assess fit quality through posterior asymptotic confidence regions
* **Sensitivity Analysis:** Evaluate and establish the sensitivity of ECR measurements
* **Synthetic Data:** Generate exact and synthetic ECR responses with optional Gaussian noise
* **Multiple Fitting Methods:** Various optimization approaches for fitting ECR data

## Installation

1. Clone or download the repository from https://github.com/ciuccislab/pyECRtools/
2. Using Anaconda:
   ```bash
   conda create -n ECR matplotlib scipy spyder
   conda activate ECR
   ```
3. Add the pyECRtools directory to your Python path or navigate to the directory when running scripts

## Demos

The package includes six comprehensive demos:

1. **Demo 1:** Generate exact and synthetic ECR responses
2. **Demo 2:** Compute ECR sensitivity with respect to k and D
3. **Demo 3:** Calculate asymptotic confidence regions and optimization parameters
4. **Demo 4:** Fit ECR data with confidence intervals and residuals
5. **Demo 5:** Compare various fitting methods for ECR data
6. **Demo 6:** Run synthetic experiments with confidence analysis

To run any demo:
```bash
python -m demos.demo_[number]
```

## Citations

If you use pyECRtools in your research, please cite:

1. Ciucci, F. (2013). Electrical conductivity relaxation measurements: Statistical investigations using sensitivity analysis, optimal experimental design and ECRTOOLS. *Solid State Ionics*, 239, 28-40. https://doi.org/10.1016/j.ssi.2013.03.020

2. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Assessing the identifiability of k and D in electrical conductivity relaxation via analytical results and nonlinearity estimates. *Solid State Ionics*, 270, 18-32. https://doi.org/10.1016/j.ssi.2014.11.026

3. Effat, M. B., Quattrocchi, E., Wan, T. H., Saccoccio, M., Belotti, A., & Ciucci, F. (2017). Electrical Conductivity Relaxation in the Nonlinear Regime. *Journal of The Electrochemical Society*, 164(14), F1671. https://doi.org/10.1149/2.1241714jes

## Support

For inquiries and support, please contact: francesco.ciucci@uni-bayreuth.de

## Dependencies

* Python
* matplotlib
* scipy
* spyder

## License

This project is distributed without any warranty; use it at your own risk.

## Contributing

Contributions and improvements to pyECRtools are welcome! Please feel free to fork the repository and submit pull requests.

---
Repository: https://github.com/ciuccislab/pyECRtools/
