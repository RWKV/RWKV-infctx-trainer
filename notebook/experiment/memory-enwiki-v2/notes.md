## Model measured performance.

Historical spreadsheet: https://docs.google.com/spreadsheets/d/18Q8JpQvUIRM5X2AlzVn5lMGQyTE-mPdQq5KDTU36N1E/edit#gid=1302009394

## Summary

| Model Type     | Number of layers | Theoretical limit | Measured limit                       |
|----------------|------------------|-------------------|--------------------------------------|
| Raven 1.5B     | 24               | 48                | ~ 10 tokens (can be finetuned to 30) |
| Raven 3B       | 32               | 64                | ~ 10 tokens (can be finetuned to 35) |
| Raven 7B       | 32               | 64                | ~ 50 tokens                          |
| Raven 14B      | 40               | 80                | ~ 75 tokens                          |
| Memory-Model-B | 96               | 192               | ~ 50 tokens (still training ... )    |