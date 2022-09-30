```python
import pandas as pd
import numpy as np
```


```python
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows', 100)

pd.set_option('display.float_format', lambda x:'{:.4f}'.format(x))
```


```python
df = pd.read_csv('./data/application_train.csv')
```


```python
df.columns = map(lambda x: x.lower(), df.columns)
```

## Configuration submodule


```python
from ml_processor.configuration import config
```

<span style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8;">
    <b>add_path</b>(lib_path=None) <br>
</span>
<p style="padding-left:25px">
    Add path to working path <br>
</p>
<p style="padding-left:5px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4">
    Parameter
</p>
<span></span>
<span></span>
<span></span>



```python

```
