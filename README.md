# Desk Sorter: CS111 Assignment 2 bonus

A Streamlit app that runs my Teachable Machine model (Pen / Phone / Water bottle)
on new photos and shows the predicted class with its confidence.

## 1. Export the model from Teachable Machine

1. Open the project, then click **Export Model**.
2. Go to the **Tensorflow** tab and select **Keras**.
3. Click **Download my model**. You get `converted_keras.zip`.
4. Unzip it. You get two files: `keras_model.h5` and `labels.txt`.
   Leaving them inside the unzipped `converted_keras/` folder is fine too.

## 2. Put the files in the `model/` folder

```
desk-sorter/
├── app.py
├── requirements.txt
├── .streamlit/config.toml     (colours and theme)
└── model/
    ├── keras_model.h5         <- put it here
    └── labels.txt             <- put it here
```

Keep the file names as they are. If the files aren't there, the app shows these steps
instead of crashing.

## 3. Run it

Use Python 3.11. TensorFlow doesn't install on 3.14.

```bash
cd desk-sorter
python3.11 -m venv .venv          # only the first time
source .venv/bin/activate
pip install -r requirements.txt   # only the first time
streamlit run app.py
```

Streamlit opens it in your browser at http://localhost:8501.

## What's in the app

- **Try a photo**: upload a picture or take one with the webcam. Shows the guess,
  its confidence, and a bar for every class. If confidence is under the cut-off
  set in the sidebar (60% by default), it flags the guess for a person to check.
- **Run my test set**: drop the 10 test photos for each class into its own box.
  The app gives accuracy (correct ÷ total × 100), the most common mistake, and a
  confusion table, all comparable with the observation log.
- **Prediction log**: every photo tried this session. Tick whether each guess was
  right. Use it for the "test end to end with at least 3 inputs" step.
