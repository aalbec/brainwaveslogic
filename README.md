# Brainwaves Logic Project

## Project Overview

**Brainwaves Logic** is a hybrid fuzzy-neural system designed to process EEG signals to control video games like **Doom** using brain waves. The system leverages fuzzy logic to interpret ambiguous EEG data and neural networks to extract complex features, providing a seamless brain-computer interface (BCI) experience.

## Features

- **Fuzzy Logic Integration**: Handles uncertainty and ambiguity in brainwave data by fuzzifying EEG signals.
- **Neural Network Processing**: Extracts and classifies patterns from EEG signals to facilitate accurate control.
- **Game Control**: Translates brain activity into commands for in-game movements and actions.
  
## Getting Started

### Prerequisites

To run the project locally, you will need the following:

- **Python 3.7+**
- Required Python libraries (install via `pip`):

  ```bash
  pip install -r requirements.txt
  ```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aalbec/brainwaveslogic.git
   ```

2. Navigate to the project directory:

   ```bash
   cd brainwaveslogic
   ```

3. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the System

You can run the system using either simulated or pre-recorded EEG data to test the hybrid processing pipeline.

1. **Preprocess EEG Data**: Use the provided scripts to preprocess raw EEG signals.
2. **Fuzzification**: Convert EEG signals into fuzzy sets using the `fuzzify.py` script.
3. **Neural Network**: Train or use pre-trained neural networks for feature extraction and classification.
4. **Game Interface**: Map the processed EEG signals to game control actions.

To run the system, use the following command:

```bash
python main.py
```

### Example EEG Dataset

For testing, you can use public EEG datasets like:

- [PhysioNet EEG Datasets](https://physionet.org/content/eegmmidb/1.0.0/)
- [GitHub User meagmohit](https://github.com/meagmohit/EEG-Datasets#eeg-datasets)

## Folder Structure

```
brainwaveslogic/
│
├── data/                # EEG datasets or preprocessed data
├── models/              # Trained neural network models
├── scripts/             # Preprocessing, fuzzification, and training scripts
├── LICENSE.txt          # Licensing information
├── README.md            # Project documentation
├── requirements.txt     # Required Python dependencies
└── main.py              # Main entry point for running the system
```

## License

### Proprietary Code License

The proprietary portions of this project are licensed for commercial use. Redistribution and modification of the proprietary code are prohibited without explicit permission. For commercial licensing inquiries, please contact <michael.diatta@gmail.com>.

### Third-Party Open-Source Licenses

This project includes components from third-party libraries such as scikit-fuzzy and scikit-image, which are licensed under the BSD 3-Clause License. See the `LICENSE.txt` file for more details.

## Contributing

Contributions are welcome! Please reach out via <michael.diatta@gmail.com> if you're interested in collaborating or submitting a pull request.

## Contact

For questions, licensing inquiries, or support, please contact:

- **aalbec**
- **Email**: <michael.diatta@gmail.com>

---

### Requirements File

You have already provided the requirements as follows, so make sure they are in the `requirements.txt` file:

```txt
numpy==2.0.1
matplotlib==3.9.1.post1
scikit-fuzzy==0.4.2
```
