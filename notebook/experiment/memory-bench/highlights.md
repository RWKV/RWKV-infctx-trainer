## RWKV model highlights

|      model      | Largest 100% match | Input of Largest 90% match | Number of tokens for 90% match | 90% match: % | Input for largest match count | Largest match count | Largest match count % |
|:---------------:|:------------------:|:--------------------------:|:------------------------------:|:------------:|:-----------------------------:|:-------------------:|:---------------------:|
|    Raven 1B5    |         10         |             50             |              46.0              |   92.000000  |              140              |          58         |       41.428571       |
|     Raven 3B    |         10         |             60             |              54.0              |   90.000000  |              140              |          83         |       59.285714       |
|     Raven 7B    |         50         |             75             |              68.0              |   90.666667  |              210              |         113         |       53.809524       |
|    Raven 14B    |         75         |             150            |              136.0             |   90.666667  |              210              |         157         |       74.761905       |
| EchoB 1B4 (L96) |         105        |             250            |              232.0             |   92.800000  |              375              |         263         |       70.133333       |

## RWKV WaveNet Channel mix highlights

|             model            | Largest 100% match | Input of Largest 90% match | Number of tokens for 90% match | 90% match: % | Input for largest match count | Largest match count | Largest match count % |
|:----------------------------:|:------------------:|:--------------------------:|:------------------------------:|:------------:|:-----------------------------:|:-------------------:|:---------------------:|
|  TokenShiftA 1B4 (L12-D2560) |         35         |             300            |              273.0             |   91.000000  |              650              |         415         |       63.846154       |
| TokenShiftB 430M (L24-D1024) |         120        |             300            |              272.0             |   90.666667  |              625              |         409         |       65.440000       |
|  TokenShiftC 1B5 (L24-D2048) |         300        |             650            |              587.0             |   90.307692  |              1000^            |         709         |       70.900000       |
|  TokenShiftD 1B4 (L96-D1024) |         150        |             1000^          |              904.0             |   90.4       |              1000^            |         904         |       90.4            |

> Note: that the benchmark currently only measures up to 1k tokens
