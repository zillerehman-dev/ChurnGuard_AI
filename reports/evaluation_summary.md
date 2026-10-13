# Evaluation summary

## CV comparison (train split, 5-fold OOF)
| Model                               |   PR-AUC |   ROC-AUC |   Precision |   Recall |     F1 |   TN |   FP |   FN |   TP |
|:------------------------------------|---------:|----------:|------------:|---------:|-------:|-----:|-----:|-----:|-----:|
| XGBoost (scale_pos_weight)          |   0.6859 |    0.8626 |      0.5369 |   0.8167 | 0.6479 | 3086 | 1053 |  274 | 1221 |
| XGBoost (SMOTE)                     |   0.6827 |    0.8625 |      0.6042 |   0.7003 | 0.6487 | 3453 |  686 |  448 | 1047 |
| Random Forest (class weights)       |   0.6793 |    0.8622 |      0.5584 |   0.7773 | 0.6499 | 3220 |  919 |  333 | 1162 |
| Logistic Regression (class weights) |   0.6760 |    0.8585 |      0.5295 |   0.8094 | 0.6402 | 3064 | 1075 |  285 | 1210 |
| Logistic Regression (SMOTE)         |   0.6754 |    0.8578 |      0.5417 |   0.7993 | 0.6458 | 3128 | 1011 |  300 | 1195 |
| Dummy (prior)                       |   0.2653 |    0.4999 |      0.0000 |   0.0000 | 0.0000 | 4139 |    0 | 1495 |    0 |

## Calibration (OOF)
| method   |   brier |   log_loss |
|:---------|--------:|-----------:|
| none     |  0.1534 |     0.4561 |
| sigmoid  |  0.1298 |     0.4008 |
| isotonic |  0.1291 |     0.4190 |
Chosen: isotonic

## Threshold: 0.32 (break-even 0.333)

## Validation policies (train OOF)
| policy           |   customers_contacted |   expected_cost |   saving_vs_no_offer |
|:-----------------|----------------------:|----------------:|---------------------:|
| Model @ 0.32     |                  2076 |     270680.0000 |           28320.0000 |
| Model @ 0.50     |                  1206 |     273980.0000 |           25020.0000 |
| Contact everyone |                  5634 |     321980.0000 |          -22980.0000 |
| No offers        |                     0 |     299000.0000 |               0.0000 |

## Test metrics
| index     |   PR-AUC |   ROC-AUC |   Precision |   Recall |     F1 |       TN |       FP |       FN |       TP |
|:----------|---------:|----------:|------------:|---------:|-------:|---------:|---------:|---------:|---------:|
| @selected |   0.6724 |    0.8538 |      0.5458 |   0.7487 | 0.6313 | 802.0000 | 233.0000 |  94.0000 | 280.0000 |
| @0.50     |   0.6724 |    0.8538 |      0.6817 |   0.5267 | 0.5943 | 943.0000 |  92.0000 | 177.0000 | 197.0000 |

## Test policy comparison (estimates under assumptions)
| policy           |   customers_contacted |   expected_cost |   saving_vs_no_offer |
|:-----------------|----------------------:|----------------:|---------------------:|
| Model @ 0.32     |                   513 |      68260.0000 |            6540.0000 |
| Model @ 0.50     |                   289 |      68760.0000 |            6040.0000 |
| Contact everyone |                  1409 |      80540.0000 |           -5740.0000 |
| No offers        |                     0 |      74800.0000 |               0.0000 |

Reporting-only CLTV proxy: 84 expected retained churners x mean CLTV of churners $4,085 = $343,169 estimated lifetime value retained, before offer costs of $10,260.

## Top SHAP features
| index            |   mean_abs_shap |
|:-----------------|----------------:|
| Contract         |          0.7351 |
| Dependents       |          0.5248 |
| Tenure Months    |          0.5132 |
| Monthly Charges  |          0.2339 |
| Online Security  |          0.2261 |
| Internet Service |          0.2253 |
| Total Charges    |          0.2012 |
| Payment Method   |          0.1981 |
| Tech Support     |          0.1879 |
| Multiple Lines   |          0.1217 |