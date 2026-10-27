# Predictive maintenance Streamlit app

https://predictive-maintenance-app-afbemfbsnoojxdfb7jbtes.streamlit.app/

An interactive interface for the saved Random Forest from the student's Colab notebook. Enter five sensor readings, press **Predict and explain**, and view the model score, failure alert, and exact SHAP explanations. The saved cutoff is 0.3. No retraining is performed.

## Interactive views

- **Score gauge:** displays the current prediction against the fixed 0.3 alert cutoff.
- **SHAP explanation:** switch between a waterfall and signed contribution bars. Hover for exact readings and contributions; drag to zoom; double-click to reset. Red contributions raise the score and blue contributions lower it. Download the exact readings, contributions, baseline, final score and cutoff as CSV.
- **What-if simulator:** select one of the five raw readings and move its slider. The scenario score, alert and SHAP explanation update automatically. Other raw readings stay fixed, while power and temperature difference are recalculated. The response curve samples up to 61 values and shows the original and selected scenarios.
- **Reference importance:** an interactive mean-absolute-SHAP chart across the 100 saved training reference records, plus a selectable feature scatter plot. This sample is different from the 200 validation examples used for the notebook's global chart, so rankings can differ.

The main input form updates only when **Predict and explain** is pressed. The result is labelled **Last submitted prediction**. A scenario does not overwrite those submitted readings. Selecting an example clears the earlier result until the example is submitted.

## Files

- `app.py`: the Streamlit interface and feature calculations.
- `predictive_maintenance.joblib`: the original uploaded model bundle, unchanged.
- `requirements.txt`: the package versions used to test the app, including SHAP and Plotly.
- `README.md`: these instructions.

## Model and validation notes

- The uploaded Random Forest records scikit-learn **1.6.1** as its training version. Keep that exact pin in requirements.txt.
- The original notebook did not record all training-package versions. Other pins here are the versions used for the deployment verification, not a claim that they all match the original Colab environment.
- The five raw readings use kelvins, rpm, Nm and minutes. Power is torque × rpm × 2π/60, and temperature difference is process temperature − air temperature.
- Column order is taken from the saved bundle, and the failure class is located using the model's class labels. Prediction uses score >= saved cutoff.
- The app warns when an input falls outside the ranges of the notebook's 6,000 fitting records. Ranges were derived using the notebook's recorded random seeds and splits.
- Test precision is 79.7%, recall 75.0%, F1 77.3%, and AP 0.828. The initial baseline also used the test set, so it was not completely unseen throughout development.
- The model score is not a calibrated real-world probability. The synthetic dataset supports classification at the recorded observation, not a proven failure warning lead time.
- SHAP uses the bundled background and `ExactExplainer`, evaluating the saved model's failure score across the 128 feature coalitions for its seven inputs. The independent masker uses exact equality for its invariant optimization, so nearby values across a tree split are not treated as equal. The failure-class index is obtained from the model's labels. Each explanation is checked: reference score + sum of contributions must equal the model score within an absolute tolerance of 0.000001.
- Exact SHAP was chosen after validation found reconstruction discrepancies with TreeExplainer for this model and runtime. The saved model, predictions and cutoff are unchanged; no residual is hidden or redistributed to force the chart to match. Attribution values can differ slightly from the notebook's TreeExplainer output.
- The reference score is the average model output over the saved background, not the dataset's failure prevalence. SHAP explains model behavior, and related features can share attribution. What-if changes describe hypothetical model outputs rather than causal effects.
- The explainer and reference-sample calculations are cached. The first explanation after a server start can take longer because SHAP must initialize. Optional reference-sample calculations run only when that view is selected.
- Only load trusted joblib files. This app loads its bundled model and does not accept model uploads from site visitors.

## Official references

- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit forms: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form
- Scikit-learn model persistence: https://scikit-learn.org/stable/model_persistence.html
- SHAP ExactExplainer: https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html
- Streamlit interactive Plotly charts: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
- Plotly waterfall charts: https://plotly.com/python/waterfall-charts/
- Changing the deployment's Python version: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python
