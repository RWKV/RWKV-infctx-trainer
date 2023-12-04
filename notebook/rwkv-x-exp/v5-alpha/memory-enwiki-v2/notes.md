## Model measured performance.

Historical spreadsheet: https://docs.google.com/spreadsheets/d/18Q8JpQvUIRM5X2AlzVn5lMGQyTE-mPdQq5KDTU36N1E/edit#gid=1302009394

## Summary

| Model Type      | Number of layers | Theoretical limit | Measured limit                   | Measured 90% limit   |
|-----------------|------------------|-------------------|----------------------------------|----------------------|
| Raven 1.5B      | 24               | 48                | ~ 10 tokens (finetuned at 30)    | ~ 40 (finetuned 90)  |
| Raven 3B        | 32               | 64                | ~ 10 tokens (finetuned at 35)    | ~ 60 (finetuned 90)  |
| Raven 7B        | 32               | 64                | ~ 50 tokens                      | ~ 75                 |
| Raven 14B       | 40               | 80                | ~ 75 tokens                      | ~ 150                |
| Memory-B (1.4B) | 96               | 192               | ~ 105 tokens                     | ~ 250                |
