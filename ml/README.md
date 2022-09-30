```python
import pandas as pd
```


```python
df = pd.read_csv('./data/application_train.csv')
```

## Configuration submodule


```python
from ml_processor.configuration import config
```

<span style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8;">
    <b>add_path</b>(lib_path=None) <br>
</span><br>
<p style="margin-left:25px">
    Add path to working path <br>
</p>
<p style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4;">
    <span style="border-bottom: 1px dotted #000"><b>Parameter</b></span>
</p>
<p style="margin-left:25px">
    <ul style="margin-left:50px">
        <li>binning_process (object) – A BinningProcess instance</li><br>
        <li>estimator (object) – A supervised learning estimator with a fit and predict method that provides information about feature coefficients through a coef_ attribute. For binary classification, the estimator must include a predict_proba method.</li><br>
        <li>scaling_method (str or None (default=None)) – The scaling method to control the range of the scores. Supported methods are “pdo_odds” and “min_max”. Method “pdo_odds” is only applicable for binary classification. If None, no scaling is applied.</li><br>
        <li>scaling_method_params (dict or None (default=None)) – Dictionary with scaling method parameters. If scaling_method="pdo_odds" parameters required are: “pdo”, “odds”, and “scorecard_points”. If scaling_method="min_max" parameters required are “min” and “max”. If scaling_method=None, this parameter is not used.</li>
    </ul>
</p>

<span></span>
<span></span>
<span></span>



```python

```
