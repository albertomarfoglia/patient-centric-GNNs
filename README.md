# Patient Centric GNNs

TODO


## Repository Structure

```
TODO
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/albertomarfoglia/patient-centric-GNNs.git
cd meds-to-owl-examples
```

2. (Recommended) Create a virtual environment:

```bash
# ----- Using conda -----
conda create -n venv python=3.11.14   # Replace "venv" and Python version as needed
conda activate venv
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

> ⚠ **Note:** `meds2rdf` requires `polars<0.20`. Using a dedicated environment avoids conflicts with other projects.

## References

* [MEDS2RDF Python Library](https://github.com/TeamHeKA/meds2rdf)
* [MEDS-OWL Ontology](https://github.com/TeamHeKA/meds-ontology)
* [Jhee, J.H. et al. (2025). "Predicting Clinical Outcomes from Patient Care Pathways Represented with  Temporal Knowledge Graphs"](https://doi.org/10.1007/978-3-031-94575-5_16).

## License

This project is licensed under the [LICENSE](LICENSE) file.

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@misc{marfoglia2026clinicaldatagoesmeds,
      title={Clinical Data Goes MEDS? Let's OWL make sense of it}, 
      author={Alberto Marfoglia and Jong Ho Jhee and Adrien Coulet},
      year={2026},
      eprint={2601.04164},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.04164}, 
}
```
