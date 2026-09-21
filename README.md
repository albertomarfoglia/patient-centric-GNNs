# Patient-Centric GNNs

This repository contains the graph-based clinical prediction experiments
described in the MEDS-OWL work. It provides the training pipeline for
patient-level outcome prediction using Relational Graph Convolutional Networks
(RGCNs) on MEDS-RDF representations.

The MIMIC-IV experiments are organized around multiple clinical prediction
tasks and can be configured through a single YAML file. The pipeline supports
repeated patient subsampling, stratified cross-validation, and the collection
of predictive-performance and computational measurements.

---

## Repository Structure

The main components used by the MIMIC-IV experiments are organized as follows:

```text
.
├── main_mimic.py                  # Main entry point for MIMIC-IV experiments
├── experiments.yaml               # Experiment/task configuration
├── requirements.txt               # Python dependencies
│
├── configs/
│   ├── datasets/
│   │   └── mimic.py               # MIMIC-IV dataset configuration
│   ├── experiment.py              # Experiment configuration
│   ├── model.py                   # Model hyperparameters
│   └── formats/
│       └── MEDSFormat.py          # MEDS graph/data format
│
├── models/
│   └── binary/
│       └── rgcn.py                # RGCN model used for binary prediction
│
├── pipelines/
│   ├── preprocess_pipeline.py     # Data preprocessing
│   └── train_pipeline.py          # Training and cross-validation
│
└── ...
````

The experiments use the following external TeamHeKA components:

* [MEDS2RDF Python Library](https://github.com/TeamHeKA/meds2rdf)
* [MEDS-OWL Ontology](https://github.com/TeamHeKA/meds-ontology)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/albertomarfoglia/patient-centric-GNNs.git
cd patient-centric-GNNs
```

### 2. Create a virtual environment

A dedicated environment is recommended.

Using Conda:

```bash
conda create -n patient-gnn python=3.11.14
conda activate patient-gnn
```

Alternatively, create a standard Python virtual environment.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `meds2rdf` requires `polars<0.20`. Using a dedicated environment
> helps avoid dependency conflicts with other projects.

The MIMIC experiments also use the
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
Sentence Transformer model. The model is loaded automatically by
`main_mimic.py` and cached locally.

---

## MIMIC-IV Experimental Setup

The main MIMIC-IV entry point is:

```text
main_mimic.py
```

Run the complete configured experiment with:

```bash
python main_mimic.py
```

The script:

1. loads the experiment definitions from `experiments.yaml`;
2. creates the model and experiment configurations;
3. loads the MIMIC-IV MEDS data;
4. preprocesses each requested task;
5. trains the RGCN model using the configured number of patient subsamples;
6. evaluates the model using stratified 10-fold cross-validation.

For the current configuration, the main script uses:

```python
ModelConfig(
    lr=5e-3,
    embed_dim=32,
)
```

and:

```python
ExperimentConfig(
    folds=10,
    time_option="TS",
    include_text=False,
    data_mode=MEDSFormat(),
    model_type=RGCNet,
)
```

Thus, all configured experiments use:

* 10-fold cross-validation;
* the `TS` time option;
* MEDS-format graph data;
* the `RGCNet` model;
* 32-dimensional embeddings;
* no text features.

---

## Experiment Configuration

The set of experiments is defined in:

```text
experiments.yaml
```

The configuration is organized into groups such as:

```yaml
experiments:
  mortality:
    ...
  los:
    ...
  readmission:
    ...
  phenotyping:
    ...
```

Each experiment contains three fields:

```yaml
- task: <task_name>
  sample_size: <number_of_patients>
  num_of_samples: <number_of_patient_subsamples>
```

### `task`

The name of the clinical prediction task.

For example:

```yaml
task: first_24_in_hospital_mortality
```

The task name is passed to `MimicConfig` and determines the corresponding
MIMIC-IV cohort and labels.

### `sample_size`

The number of patients included in each patient subsample.

For example:

```yaml
sample_size: 1000
```

A different value can be specified for individual tasks.

### `num_of_samples`

The number of independent patient subsamples generated and evaluated for the
task.

For example:

```yaml
num_of_samples: 5
```

causes the training pipeline to run five independent patient subsamples.

This is implemented in `main_mimic.py` through:

```python
ExperimentConfig(
    ...
    dataset_samples=exp["num_of_samples"],
)
```

and subsequently used by `run_train_pipeline()`.

---

## Current MIMIC-IV Configuration

The current `experiments.yaml` contains the following tasks.

### Mortality

| Task                                    | Patients per subsample | Number of subsamples |
| --------------------------------------- | ---------------------: | -------------------: |
| `first_24_in_hospital_mortality`        |                   1000 |                    5 |
| `first_24_in_icu_mortality`             |                   1000 |                    1 |
| `first_48_in_hospital_mortality`        |                   1000 |                    1 |
| `first_48_in_icu_mortality`             |                   1000 |                    1 |
| `post_hospital_discharge_mortality_30d` |                   1000 |                    1 |
| `post_hospital_discharge_mortality_1y`  |                   1000 |                    1 |

### Length of stay

| Task                       | Patients per subsample | Number of subsamples |
| -------------------------- | ---------------------: | -------------------: |
| `first_48_in_hospital_los` |                   1000 |                    1 |
| `first_48_in_icu_los`      |                   1000 |                    1 |

### Readmission

| Task              | Patients per subsample | Number of subsamples |
| ----------------- | ---------------------: | -------------------: |
| `readmission_30d` |                   1000 |                    1 |

### Phenotyping

| Task                                         | Patients per subsample | Number of subsamples |
| -------------------------------------------- | ---------------------: | -------------------: |
| `ckd_in_diabetics_within_5y_of_kidney_panel` |                   1000 |                    1 |
| `myocardial_infarction_1-5y_phenotyping`     |                    886 |                    1 |

The first 24-hour in-hospital mortality task is therefore configured with five
independent patient subsamples, while the remaining tasks currently use one
subsample.

---

## Changing the Experiments

The experiment configuration can be modified directly in
`experiments.yaml`.

For example, to evaluate the first 24-hour in-hospital mortality task on five
independent subsamples of 1,000 patients:

```yaml
experiments:
  mortality:
    - task: first_24_in_hospital_mortality
      sample_size: 1000
      num_of_samples: 5
```

To increase the number of patient subsamples:

```yaml
- task: first_24_in_hospital_mortality
  sample_size: 1000
  num_of_samples: 10
```

To change the number of patients per subsample:

```yaml
- task: first_24_in_hospital_mortality
  sample_size: 2000
  num_of_samples: 5
```

The cross-validation configuration is currently controlled in
`main_mimic.py`:

```python
folds=10
```

Thus, changing `num_of_samples` changes the number of independent patient
subsamples, while changing `folds` changes the number of cross-validation
folds performed within each subsample.

---

## MIMIC-IV Data Configuration

The MIMIC-IV experiment currently expects the MEDS-formatted data to be
available through the paths configured in `main_mimic.py`:

```python
dataset_cfg = MimicConfig(
    source_dir=Path("../meds-to-owl-examples/MIMIC/exp3-0.9"),
    labels_dir=Path("../meds-to-owl-examples/MIMIC/exp0-0.9"),
    num_patients=exp["sample_size"],
    task=exp["task"],
)
```

Therefore, before running `main_mimic.py`, the corresponding MEDS-formatted
MIMIC-IV input data and task labels must be available at these locations, or
the paths must be changed to match the local installation.

The repository does not distribute the MIMIC-IV dataset.

---

## Experimental Pipeline

For each configured task, `main_mimic.py` performs the following sequence:

```text
experiments.yaml
       |
       v
ExperimentConfig + MimicConfig
       |
       v
run_preprocess_pipeline()
       |
       v
MEDS-formatted task data
       |
       v
run_train_pipeline()
       |
       +--> patient subsample 0 --> 10-fold CV
       |
       +--> patient subsample 1 --> 10-fold CV
       |
       +--> ...
       |
       +--> patient subsample N --> 10-fold CV
       |
       v
aggregated results
```

Each cross-validation fold is trained and evaluated independently. The
training pipeline also records computational measurements such as training
duration and estimated emissions.

---

## Related Repositories

This repository contains the downstream graph-learning experiments. The RDF
conversion and ontology components are maintained separately by TeamHeKA:

* [MEDS2RDF](https://github.com/TeamHeKA/meds2rdf)
* [MEDS-OWL](https://github.com/TeamHeKA/meds-ontology)

The MEDS2RDF library is used to transform MEDS datasets into MEDS-RDF
representations, while MEDS-OWL provides the ontology underlying the semantic
representation.

---

## References

* Marfoglia, A. et al. (2026). *Clinical Data Goes MEDS? Let's OWL make sense of it.*   [https://arxiv.org/abs/2601.04164](https://arxiv.org/abs/2601.04164)
* [MEDS2RDF Python Library](https://github.com/TeamHeKA/meds2rdf)
* [MEDS-OWL Ontology](https://github.com/TeamHeKA/meds-ontology)
* Jhee, J. H. et al. (2025). *Predicting Clinical Outcomes from Patient Care
  Pathways Represented with Temporal Knowledge Graphs.*
  [https://doi.org/10.1007/978-3-031-94575-5_16](https://doi.org/10.1007/978-3-031-94575-5_16)


<!--

---

 ## Citation

If you use this repository or the accompanying experiments, please cite the
associated publication:

```bibtex
@article{marfoglia2026medsowl,
  title   = {MEDS-OWL: An Ontology-Driven Framework for Clinical Outcome Prediction with Graph Learning},
  author  = {Marfoglia, Alberto and Carbonaro, Antonella and Coulet, Adrien},
  year    = {2026}
}
``` -->

---

## License

This project is licensed under the [LICENSE](LICENSE) file.