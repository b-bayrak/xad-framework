"""
    Experiment scripts that reproduce the results of the paper.

    Experiments:
        - exp_1_detection_performance.py: Detection performance across architectures
        - exp_2_calibration_quality.py: Calibration quality at different alpha levels
        - exp_3_uncertainty_effectiveness.py: Uncertainty effectiveness
        - exp_4_counterfactual_quality.py: Counterfactual explanation quality
        - exp_5_gace_ablation.py: GA-CE loss-term ablation
        - exp_6_cp_ablation.py: Threshold policies and the temporal term of the score
        - exp_7_policy_sensitivity.py: Sensitivity to the ACI step size and the window size
        - exp_8_classical_baselines.py: Isolation Forest, LOF and One-Class SVM baselines

    Usage:
        python experiments/run_all_experiments.py --dataset skab --data_path ./data/SKAB/
"""
