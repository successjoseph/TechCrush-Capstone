# TechCrush-Capstone — PhytoScan Plant Disease Classifier

**Language:** Python (Streamlit + TensorFlow) · **Visibility:** Public · **License:** No license file included — all rights reserved by default

## Table of Contents
- [About](#about)
- [Repository Map](#repository-map)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Contributing](#contributing)
- [Authors and License](#authors-and-license)

## About

TechCrush-Capstone is a bootcamp/course capstone project: a Streamlit web app called **PhytoScan** that classifies plant leaf images into one of 38 crop/disease categories using a trained Keras CNN (`plant_disease_cnn.keras`, tracked via Git LFS). A helper script in the repo (`getfolders.py`) hardcodes a local path reading `"...\Capstone Project Group 6 C6 AIML Track A\plantvillage-dataset\raw\color"`, confirming this was built as a group capstone (Group 6, "C6 AIML Track") using the well-known PlantVillage dataset, consistent with the repo name suggesting a TechCrush bootcamp course project. The app is a single-file, multi-section Streamlit application with a dark, custom-styled UI, presenting an overview, the dataset, the model architecture, training results, a live image classifier, and a team page — i.e., it doubles as both a working demo and a presentation of the group's ML pipeline.

## Repository Map

```
personal.py                the entire Streamlit app (1358 lines): custom CSS/theme, and six sidebar
                            sections — Overview, Dataset, Model Architecture, Training Results,
                            Live Classifier, Team
plant_disease_cnn.keras     the trained Keras CNN model (tracked via Git LFS, ~128 MB)
getfolders.py               one-off dev utility that lists subfolder names of a local PlantVillage
                            dataset directory (not used by the app itself; hardcoded local path)
requirements.txt             streamlit, tensorflow, numpy, pillow — plus commented links to a
                              Google Doc and a Colab notebook used during development
.gitattributes               configures Git LFS tracking for *.keras files
```

## Prerequisites

- Python 3 with `streamlit`, `tensorflow`, `numpy`, and `pillow` installed (per `requirements.txt`).
- Git LFS, to correctly pull `plant_disease_cnn.keras` (tracked via `.gitattributes`: `*.keras filter=lfs diff=lfs merge=lfs -text`).

## Installation

```bash
git clone https://github.com/successjoseph/TechCrush-Capstone.git
cd TechCrush-Capstone
git lfs pull            # fetch the ~128 MB model file if not already present
pip install -r requirements.txt
```

## Usage

```bash
streamlit run personal.py
```

The app opens with a sidebar radio menu (`personal.py` line 628) with six pages: "Overview," "Dataset," "Model Architecture," "Training Results," "Live Classifier," and "Team." On the Live Classifier page, a user uploads a leaf image (`st.file_uploader`); the app runs it through the loaded model (`model.predict`) and reports the predicted crop and condition (via a `parse_class()` helper that splits class names like `Tomato___Early_blight` into crop + condition) along with the top prediction confidences.

The development notebook and write-up referenced during training are linked as comments in `requirements.txt` (a Google Doc and a Google Colab notebook) but are not included in the repository itself.

## Model

`plant_disease_cnn.keras` is loaded via `tf.keras.models.load_model('./plant_disease_cnn.keras')` and cached with `@st.cache_resource`. It classifies images into 38 classes covering 14 crops (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper (bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato), each either "healthy" or a specific disease (e.g. `Apple___Apple_scab`, `Tomato___Tomato_Yellow_Leaf_Curl_Virus`, `Orange___Haunglongbing_(Citrus_greening)`). The app includes a `DISEASE_INFO` dictionary with short, human-readable descriptions and general treatment notes for each non-healthy condition, and a "Training Results" page presenting the model's reported training metrics (referenced in-app; the underlying training notebook itself is not part of this repo).

## Contributing

This is a public bootcamp/group capstone project. There is no explicit contribution guide in the repo; treat any external contribution interest as a matter for the author/team to decide on a case-by-case basis.

## Authors and License

**Author:** successjoseph (github.com/successjoseph), submitted as part of a group project ("Capstone Project Group 6 C6 AIML Track A," per `getfolders.py`) — other team members are not identified in the code itself.

**License:** No license file included — all rights reserved by default.
