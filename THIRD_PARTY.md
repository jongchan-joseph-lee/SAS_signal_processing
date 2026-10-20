# Sources and separate licenses

## PoSSM public data

- Raw GDF: https://www.marine-geo.org/tools/files/31898
- SLC HDF5: https://www.marine-geo.org/tools/files/31899
- Reference GeoTIFF (not used in this execution): https://www.marine-geo.org/tools/files/31900
- Supporting layout products: https://www.marine-geo.org/tools/files/31901
- Repository-stated license: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States: https://creativecommons.org/licenses/by-nc-sa/3.0/us/
- Repository investigator attribution: Philtron, Johnson and Brown; dataset DOIs https://doi.org/10.60521/331898 (raw) and https://doi.org/10.60521/331899 (SLC).
- Acknowledge the data creators and the Marine Geoscience Data System (MGDS). This bundle retains the supplied README, data-package description, GDF format and scatterer/object locations. Exact downloaded files and SHA-256 hashes are in `references/download_manifest.json`.
- The data-package document cites Johnson & Brown, *SAS simulations with procedural texture and the point-based sonar scattering model*, OCEANS 2018; Abawi, *Kirchhoff scattering from non-penetrable targets modeled as an assembly of triangular facets*, JASA 140(3), 2016; Bellettini & Pinto, *Design and experimental results of a 300-kHz synthetic aperture sonar optimized for shallow-water operations*, IEEE JOE 34(3); Gerg et al., *GPU acceleration for synthetic aperture sonar image reconstruction*, Global Oceans 2020.
- Fully model-derived underwater acoustic data, no added noise, ideal linear navigation. The scene01 object types are lobster traps, rocks and cylinders, not verified mines.

To preserve source restrictions, the derived results, source-layout CSVs, plots, weights and report in this delivery are distributed under CC BY-NC-SA 3.0. The MIT license applies only to original program code. Official PDFs retain their own attribution.

## Research references (ideas; no copied model weights/code)

- Gerg & Monga, *Real-Time, Deep Synthetic Aperture Sonar (SAS) Autofocus*, 2021: https://arxiv.org/abs/2103.10312 — motivates phase-estimation autofocus. Our PhaseNet architecture/training is original and is not a reproduction of that paper.
- Reed et al., *Neural Volumetric Reconstruction for Coherent Synthetic Aperture Sonar*, 2023: https://arxiv.org/abs/2306.09909 — physics-aware neural reconstruction; not implemented here.
- *An in-air synthetic aperture sonar dataset of target scattering in environments of varying complexity*, Scientific Data 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC11538465/ — measured in-air data, not part of the executed PoSSM experiment.

## Dependencies

NumPy, SciPy, Numba, h5py, matplotlib, requests and PyTorch remain under their respective upstream licenses. No third-party pretrained model weights are included.

## v3 classical autofocus and low-data detector

- Wahl, Eichel, Ghiglia, Jakowatz, *Phase gradient autofocus—a robust tool for high resolution SAR phase correction*, 1994, https://doi.org/10.1109/7.303752 .
- Evers & Jackson, *A Generalized Phase Gradient Autofocus Algorithm*, 2019, https://doi.org/10.1109/TCI.2019.2899453 ; author dissertation https://scholar.afit.edu/etd/2240/ . The present strongest-pixel adjacent-ping local estimator is an adaptation, not the full Max-SDR paper reproduction.
- Henrik Forsten, author notes and experimental implementation: https://hforsten.com/synthetic-aperture-radar-autofocus-and-calibration.html . Consulted for BP-contribution interpretation; no source code copied.
- scikit-learn1.8.0 RandomForestClassifier (BSD-3-Clause), https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html ; joblib1.5.3 (BSD-3-Clause). v3 trains its own low-data baseline, no external weights. Derived RF weights retain PoSSM noncommercial share-alike restrictions.
- v3 does not retrain the v1 PhaseNet. requirements_v3 omits PyTorch because no neural network is used in this stage.
