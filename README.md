leodecay
==============================

### Description

Tools for downloading, preprocessing, and analyzing space weather and LEO satellite orbital decay data.
 
### Project Organization
    ├── LICENSE                                    <- License file for the project.
    ├── README.md                                  <- README for this specific sub-project.
    ├── requirements.txt                           <- Defines Python dependencies for the analysis environment.
    ├── .gitignore                                 <- Specifies intentionally untracked files to ignore.
    ├── configs/                                   <- Configuration data classes.
    ├── data/                                      <- Methods for downloading datasets (e.g., OMNI, LATIS, POD, TLE).
    ├── merging/                                   <- Scripts to merge datasets into a consolidated form for modeling.
    ├── pipeline/                                  <- Scripts for dataset compilation.
    ├── preprocessing/                             <- Scripts for cleaning and transforming individual datasets using configuration pipelines.
    ├── postprocessing/                            <- Maneuver postprocessing.
    ├── utils/                                     <- General-purpose helper functions.
    └── visualization/                             <- Scripts for with plotting methods for data and results.

### Reproducibility

1. Clone the repository.
2. Install dependencies:
   e.g., pip install -r requirements.txt
3. Install the package:
   e.g., pip install -e .

### Contact

For questions or feedback, please contact: <br>
**Vanessa Mercea (PhD Student, University of Bern, Switzerland)* – [vanessa-maria.mercea@unibe.ch](mailto:vanessa-maria.mercea@unibe.ch)<br>
**ORCID* [0000-0001-5252-9393](https://orcid.org/0000-0001-5252-9393)<br>

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>

