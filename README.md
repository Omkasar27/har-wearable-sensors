# Wearable Sensor Data Analytics: Activity Recognition using 1D-CNN and LSTM Networks

Recognising 12 physical activities from accelerometer and gyroscope signals recorded by three body-worn sensors (hand, chest, ankle). Deep learning models (1D-CNN, LSTM, CNN-LSTM) are compared against a classical Random Forest baseline under a strict subject-wise evaluation, meaning the test person is never seen during training.

Course project, Deep Learning Applications (Human Activity Recognition module).

## Demo

A Streamlit app loads the trained models, shows real windows from an unseen person with each model's prediction, and accepts an uploaded sensor recording.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

[PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) (Reiss and Stricker, 2012), licensed CC BY 4.0. This project uses a Kaggle copy by Diego Silva França (`diegosilvadefrana/fisical-activity-dataset`): one CSV, 8 subjects, 2.86 million rows, 12 activities after removing the "transient" label. The dataset is not included in this repository because of its size.

## Method

1. **Cleaning.** Dropped the transient label (32% of rows). Kept 18 channels (accelerometer and gyroscope at hand, chest, ankle). Temperature, magnetometer and heart rate were excluded because they can identify a person rather than an activity.
2. **Windowing.** Downsampled 100 Hz to 50 Hz, then cut each contiguous segment into windows of 128 samples (2.56 s) with 50% overlap. Result: 14,975 windows of shape 128 x 18.
3. **Subject-wise split.** Train on subjects 2, 3, 4, 6, 7, 8. Validate on subject 1. Test on subject 5. The scaler is fitted on training subjects only.
4. **Baseline.** 222 hand-crafted features per window (statistics, spectral features, acceleration magnitude) with Random Forest and RBF-SVM.
5. **Deep models.** 1D-CNN, LSTM and CNN-LSTM, with class weights, dropout, batch normalisation, gradient clipping, and random scaling plus noise augmentation.
6. **Analysis.** Per-class scores, confusion matrices, sensor placement study, leave-one-subject-out (LOSO) evaluation.

## Results

Test on the unseen subject 5:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest (baseline) | 0.899 | 0.901 |
| 1D-CNN with augmentation | 0.863 | 0.865 |
| CNN-LSTM | 0.854 | 0.834 |
| LSTM | 0.813 | 0.797 |

Leave-one-subject-out, mean over 8 subjects:

| Model | Mean accuracy | Std |
|---|---|---|
| Random Forest | 0.903 | 0.023 |
| 1D-CNN with augmentation | 0.780 | 0.161 |

Sensor placement (1D-CNN, subject 5): chest alone 0.819, hand alone 0.701, ankle alone 0.633, hand + chest 0.855, all three 0.863.

## Findings

- The Random Forest was the most reliable model and won on all 8 subjects in LOSO.
- With only 6 to 7 training subjects, the networks partly learn person-specific signal patterns. Training accuracy reached about 96% while accuracy on unseen people stayed near 80 to 86%. Random scaling and noise augmentation raised 1D-CNN test accuracy from 0.826 to 0.863.
- Subject 8 is an outlier for the 1D-CNN (0.41 accuracy in LOSO), while the Random Forest scored 0.90 on the same person.
- The chest is the most informative single sensor, and hand + chest nearly matches all three.
- The hardest activities for every model are standing and sitting, and vacuum cleaning and ironing.

## Repository contents

```
app.py                  Streamlit demo
requirements.txt        Python dependencies
.streamlit/config.toml  Theme for the demo
meta.json               Scaler values, class names, result tables
demo_windows.npz        Sample windows from the unseen test subject
cnn1d_aug.keras         Trained 1D-CNN
cnn_lstm.keras          Trained CNN-LSTM
lstm.keras              Trained LSTM
notebook.ipynb          Full pipeline: EDA, preprocessing, models, analysis
```

## Limitations

- One test subject and one random seed for the main comparison, and 8 subjects in total.
- No timestamps in the Kaggle copy, so a 100 Hz sampling rate is assumed.
- The deep models were lightly tuned.
- Sensor placement matches PAMAP2 (dominant wrist, chest, ankle). The models will not transfer to a phone in a pocket. Not for medical use.

## Future work

3D rotation augmentation, gravity removal, feeding hand-crafted features into the network, subject-adaptive fine-tuning, and validation on other datasets such as MHEALTH or UCI HAR.

## Author

[Your name], [your college or course], [year]

## Reference

A. Reiss and D. Stricker. Introducing a New Benchmarked Dataset for Activity Monitoring. 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.