
# Chronos-2 Fine-Tuning Project Documentation

## 1. Project Overview

This project focuses on fine-tuning the Chronos-2 time-series forecasting model using Low-Rank Adaptation (LoRA).

The assigned task was to fine-tune Chronos-2 and evaluate its forecasting performance. To assess the effect of fine-tuning, the fine-tuned model was compared with the original zero-shot Chronos-2 model and a seasonal-naive baseline.

**Models evaluated:**
- Chronos-2 Zero-Shot
- Chronos-2 Fine-Tuned with LoRA
- Seasonal-Naive Baseline

The implementation was carried out using Python and Google Colab.

## 2. Dataset

The project uses the hourly version of the Electricity dataset from the Monash time-series dataset collection.

| Property | Description |
|---|---|
| Dataset | Electricity |
| Number of series | 321 |
| Frequency | Hourly |
| Time period | 2012–2014 |
| Observations per series | 26,304 |
| Measurement unit | kW |
| Missing values | None |

Each series represents an individual electricity customer.

### Data preparation

The dataset was divided chronologically into training, validation, and test sets for each series.

| Split | Observations per series |
|---|---:|
| Training | 18,412 |
| Validation | 3,946 |
| Test | 3,946 |

The chronological split ensured that future observations were not used during model training.

The prepared splits were saved in `dataset_splits.npz` for reuse across the different notebooks.

## 3. Implementation Workflow

The implementation was organised into four notebooks to separate data preparation, model evaluation, fine-tuning, and final testing.

### Notebook 01: Data Preparation

`01_Chronos2_Data_Preparation.ipynb`

Tasks:
- Load the hourly Electricity dataset.
- Inspect the dataset structure and individual time series.
- Prepare chronological training, validation, and test splits.
- Save the prepared splits for subsequent notebooks.

### Notebook 02: Zero-Shot Evaluation

`02_Chronos2_Zero_Shot_Evaluation.ipynb`

The original Chronos-2 model was evaluated without additional training.

Tasks:
- Load the pretrained `amazon/chronos-2` model.
- Generate forecasts using the test data.
- Evaluate forecasting performance using MAE, RMSE, and MASE.
- Implement a seasonal-naive baseline.
- Compare zero-shot Chronos-2 against the baseline.

### Notebook 03: LoRA Fine-Tuning

`03_Chronos2_LoRA_Fine_Tuning.ipynb`

Chronos-2 was fine-tuned using Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique.

**Training configuration**

| Parameter | Value |
|---|---:|
| Base model | `amazon/chronos-2` |
| Fine-tuning method | LoRA |
| Training steps | 500 |
| Batch size | 4 |
| Learning rate | 0.00001 |
| Context length | 168 |
| Prediction length | 24 |
| Best checkpoint | Step 200 |

The validation loss was used to select the best checkpoint. The selected checkpoint was saved in Google Drive.

### Notebook 04: Final Evaluation

`04_Chronos2_Final_Evaluation.ipynb`

The fine-tuned model was evaluated on the held-out test data and compared with the previously obtained zero-shot and seasonal-naive results.

Tasks:
- Load the saved LoRA checkpoint.
- Generate forecasts using the same test evaluation protocol.
- Calculate per-series MAE, RMSE, and MASE.
- Combine the results from all three forecasting approaches.
- Calculate aggregate performance metrics and customer-level comparisons.
- Save the final evaluation results.

## 4. Forecasting and Evaluation Configuration

The same forecasting setup was used for the three approaches.

| Parameter | Configuration |
|---|---:|
| Context length | 168 hours |
| Forecast horizon | 24 hours |
| Forecast frequency | Every 24 hours |
| Forecast quantile | 0.5 (median) |
| Evaluation series | 321 |
| Forecast windows per series | 164 |
| Total forecast windows | 52,644 |

A rolling-origin evaluation approach was used.

For each forecast window, the model received the most recent 168 observations and generated predictions for the next 24 hours. After each forecast, the observed values were incorporated into the available history before the next forecasting window.

This resulted in 52,644 forecast windows and 1,263,456 hourly predictions per model.

The final 10 observations in each test series were not included because they did not form a complete 24-hour forecast window.

### Evaluation metrics

Three metrics were used:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values.
- **Root Mean Squared Error (RMSE):** Measures prediction errors while giving greater weight to larger errors.
- **Mean Absolute Scaled Error (MASE):** Measures forecast error relative to a seasonal-naive scaling factor calculated from the training data.

For MASE, a seasonal period of 24 hours was used to account for daily seasonality.

## 5. Final Evaluation Results

The following table presents the aggregate test performance across the 321 electricity customer series.

| Metric | Chronos-2 Zero-Shot | Chronos-2 LoRA | Seasonal Naive |
|---|---:|---:|---:|
| Mean MAE | 170.4269 | 170.7693 | 210.5993 |
| Median MAE | 43.1387 | 43.1383 | 54.5094 |
| Mean RMSE | 323.5805 | 324.0206 | 374.2500 |
| Median RMSE | 70.7897 | 70.7994 | 95.8477 |
| Mean MASE | 0.8181 | 0.8179 | 1.0412 |
| Median MASE | 0.8233 | 0.8230 | 1.0177 |

### Customer-level comparison

The per-series MAE comparison produced the following results:

| Comparison | Result |
|---|---:|
| LoRA wins over zero-shot | 208 series |
| Zero-shot wins over LoRA | 113 series |
| Ties | 0 |
| LoRA wins over seasonal naive | 318 series |
| Zero-shot wins over seasonal naive | 318 series |

### Key observations

1. Both Chronos-2 approaches performed better than the seasonal-naive baseline on 318 out of 321 customer series.
2. Zero-shot Chronos-2 achieved slightly lower mean MAE and mean RMSE than the LoRA fine-tuned model.
3. The LoRA model achieved marginally lower median MAE and lower mean and median MASE.
4. Although LoRA performed better on more individual customer series, its aggregate mean MAE was slightly higher than that of the zero-shot model.
5. The difference between the two Chronos-2 approaches was very small. The results therefore do not demonstrate a substantial overall improvement from LoRA fine-tuning under the evaluation configuration used.

### Notable series

Two customer series were particularly notable:

- **T183:** The test observations were all zero despite a non-zero historical training mean. This produced an unusual evaluation case and should be considered when interpreting the aggregate results.
- **T314:** Both Chronos-2 approaches recorded particularly large forecasting errors on this series.

These cases were retained in the final evaluation.

## 6. Computational Performance

The LoRA fine-tuning and inference processes were measured separately.

| Measurement | Result |
|---|---:|
| Fine-tuning time | 2.73 minutes |
| Peak training GPU memory | 1.41 GB |
| Test inference time | 1.58 minutes |
| Test forecast windows | 52,644 |
| Inference batch size | 32 |

The implementation was run in Google Colab using an NVIDIA Tesla T4 GPU.

The zero-shot evaluation was performed sequentially, whereas the final LoRA inference used batching. Their recorded execution times should therefore not be treated as a direct speed comparison.

## 7. Saved Outputs

The final evaluation outputs were saved in Google Drive under:

`Chronos2_Finetune/Final_Results/`

| File | Description |
|---|---|
| `final_test_comparison.csv` | Per-customer comparison of the three forecasting approaches |
| `final_test_summary.csv` | Aggregate evaluation metrics |
| `lora_test_predictions.npz` | Saved LoRA test predictions |
| `computational_results.json` | Fine-tuning and inference measurements |

The fine-tuned model checkpoint was saved separately under:

`Chronos2_LoRA_Checkpoint/finetuned-ckpt`

## 8. Conclusion

Chronos-2 was successfully fine-tuned using LoRA and evaluated on the hourly Electricity dataset.

The final evaluation showed that both the original zero-shot model and the LoRA fine-tuned model outperformed the seasonal-naive baseline on most customer series.

However, the difference between zero-shot and LoRA performance was marginal. LoRA achieved better results on more individual series and slightly lower median MAE and MASE, while zero-shot Chronos-2 retained a small advantage in mean MAE and RMSE.

The implementation demonstrates the fine-tuning workflow, checkpoint selection, forecasting evaluation, and comparison of a fine-tuned time-series model with its pretrained version and a baseline.

## 9. Implementation Structure

The project was organised as follows:

```text
Chronos2-Electricity-Forecasting/
│
├── notebooks/
│   ├── 01_chronos2_data_preparation.ipynb
│   ├── 02_chronos2_zero_shot_evaluation.ipynb
│   ├── 03_chronos2_lora_fine_tuning.ipynb
│   └── 04_chronos2_final_evaluation.ipynb
│
├── results/
│   ├── final_test_comparison.csv
│   ├── final_test_summary.csv
│   └── computational_results.json
│
└── PROJECT_DOCUMENTATION.md
```
