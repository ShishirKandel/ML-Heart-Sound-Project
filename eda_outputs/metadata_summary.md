# CirCor 2022 — Metadata EDA summary

Total patients in public training split: 942

## Murmur class balance (patient-level)
Murmur
Absent     695
Present    179
Unknown     68

## Outcome class balance (patient-level)
Outcome
Normal      486
Abnormal    456

## Murmur × Outcome cross-tab
Outcome  Abnormal  Normal  All
Murmur                        
Absent        263     432  695
Present       150      29  179
Unknown        43      25   68
All           456     486  942

## Sex distribution
Sex
Female    486
Male      456

## Age category distribution
Age
Child         664
Infant        126
NaN            74
Adolescent     72
Neonate         6

## Height / Weight summary
           Height      Weight
count  826.000000  837.000000
mean   110.800242   23.632756
std     30.000607   15.453337
min     35.000000    2.300000
25%     89.000000   12.500000
50%    115.000000   20.400000
75%    133.000000   31.200000
max    180.000000  110.800000
