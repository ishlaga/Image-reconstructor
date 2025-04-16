# 🧠 Image-reconstructor

This project demonstrates how **Principal Component Analysis (PCA)** can be used to reconstruct 16x16 grayscale images from the **USPS handwritten digit dataset** by reducing the dimensionality of the data and then reconstructing it from the reduced representation.




---

## 📊 Dataset Details

- **File:** `USPS.mat`
- **Shape:** `(3000, 256)` – 3000 images, each represented as a flattened 16x16 grayscale digit.
- **Each row** of the matrix represents one image (digit between 0 and 9).

## 🚀 How to Run

1. Ensure the dataset `USPS.mat` is placed inside the `data/` folder.
2. Complete your implementations inside `solution.py`.
3. From the `code/` directory, run the following command:

```bash
python main.py
