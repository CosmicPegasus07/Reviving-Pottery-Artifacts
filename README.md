# Reviving Pottery Artifacts: Generative AI Image Inpainting for Restoration

## Project Overview

This project focuses on restoring historical artifacts and manuscripts using cutting-edge generative artificial intelligence techniques. By leveraging deep learning and generative models, we aim to reconstruct damaged or missing parts of artifacts, bringing timeless history back to life.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [How It Works](#how-it-works)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

## Features

- Generative AI-based artifact restoration using advanced models like GANs and diffusion models
- Pretrained models for pottery 
- Support for training on custom datasets
- Visualization tools for comparing original and restored artifacts
- Scalable architecture for adding new restoration techniques

## Prerequisites

- Python 3.10+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- Flask (for API server)
- OpenCV (for image processing)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Timeless-Reconstruction.git
   cd Timeless-Reconstruction

2. Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate #If MacOs use source venv/bin/activate 

3. Install the required packages:

   ```bash
   pip install -r requirements.txt

4. Run FlaskApp:

   ```bash
   cd FlaskApp
   python app.py

or

5. Run Jupiter Notebook:

   ```bash
   cd Notebook Implementation
   mv Pottery Inpainting.ipynb FlaskApp
   Run the notebook in the FlaskApp folder

## Project Structure
```text
Reviving Pottery Artifacts/
│── FlaskApp/                   # Main Flask application directory
│   │── __pycache__/            # Compiled Python files
│   │── .ipynb_checkpoints/     # Jupyter Notebook checkpoints
│   │── deepfillv2/             # DeepFill v2 model (for image inpainting)
│   │── examples/               # Example images or test cases
│   │── input/                  # Input images or data
│   │── model/                  # Model files or pre-trained weights
│   │── output/                 # Output results (processed images, logs, etc.)
│   │── templates/              # HTML templates for Flask frontend
│   │── app.py                  # Main Flask application
│   │── config.py               # Configuration settings
│   │── create_mask.py          # Script to generate masks for inpainting
│   │── edge_detection_mask.py   # Script for edge detection-based masking
│   │── inpaint.py              # Core inpainting script
│
│── Notebook Implementation/    # Jupyter notebooks for model testing
│   │── Pottery Inpainting.ipynb # Notebook demonstrating pottery inpainting
│
│── .gitignore                  # Git ignore file
│── LICENSE                     # Project license
│── README.md                   # Project documentation
│── requirements.txt             # Dependencies for the project
```
## How It Works

1. **Data Preparation**: Input images of damaged artifacts are preprocessed for compatibility with restoration models.

2. **Generative Modeling**: 
   - Generative Adversarial Networks (GANs) or diffusion models reconstruct damaged or missing regions.
   - Models are pre-trained on datasets of historical artifacts and can be fine-tuned for specific use cases.

3. **Restoration Process**:
   - Input images undergo preprocessing for normalization and artifact segmentation.
   - The restoration model generates the reconstructed artifact.
   - Post-processing ensures that restored artifacts maintain historical accuracy and visual quality.
  
4. **Evaluation**: Metrics like PSNR, SSIM, and perceptual similarity are used to assess the quality of restorations.

5. **Real-Time API**: The Flask API allows users to instantly upload images and receive restored versions.

## Contributing

We welcome contributions to improve this project. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

Please make sure your code follows our coding standards and includes the appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or concerns, please open an issue on GitHub or contact the project maintainer at afeef2001kashif@gmail.com | srushti.vp10@gmail.com                
