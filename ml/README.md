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

<div class="module">
    <div class="module-details">
        <p style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8;">
            <span style="font-size: 16px; font-weight:bold">config</span>( )
        </p>
        <p style="margin-left:25px">
            Perform basic configurations and logging
        </p>
        <p style="">
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; 
                         padding-left: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li>None</li>
            </ul>
        </p>
    </div>
    <!-- add path -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding-left: 5px">
                <span style="font-size: 16px; font-weight:bold">add_path</span> (lib_path=None)
            </span>
        </p>
        <p style="margin-left:25px">
            Add path to current home path
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding-left: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>lib_path</b> (string) - Path to add to home path</li>
            </ul>
        </p>
    </div>
</div>






```python

```
