# Сравнительный анализ моделей (COCO2017)

## Сводная таблица метрик

| Модель | Время (ms) | mAP50-95 | mAP50 | mAP75 | mAP small | mAP medium | mAP large | AR1 | AR100 |
|---|---|---|---|---|---|---|---|---|---|
| **LibreYOLO9t** | 25.31 | 0.377 | 0.5279 | 0.4021 | 0.1794 | 0.415 | 0.5388 | 0.3137 | 0.5602 |
| **LibreYOLO9s** | 24.18 | 0.4597 | 0.624 | 0.497 | 0.2525 | 0.5066 | 0.6353 | 0.3534 | 0.6189 |
| **LibreYOLO9m** | 39.62 | 0.5038 | 0.669 | 0.5471 | 0.3245 | 0.557 | 0.6676 | 0.3756 | 0.6551 |
| **LibreYOLO9c** | 70.40 | 0.5175 | 0.6867 | 0.5602 | 0.3515 | 0.5727 | 0.6725 | 0.3829 | 0.6666 |
| **LibreYOLO9E2Et** | 16.43 | 0.3384 | 0.4843 | 0.3596 | 0.1467 | 0.3729 | 0.4754 | 0.2887 | 0.5075 |
| **LibreYOLO9E2Es** | 24.59 | 0.4146 | 0.5766 | 0.4462 | 0.2115 | 0.4582 | 0.5682 | 0.3293 | 0.5693 |
| **LibreYOLO9E2Em** | 40.84 | 0.4763 | 0.6455 | 0.5146 | 0.3009 | 0.529 | 0.6209 | 0.3608 | 0.6263 |
| **LibreYOLO9E2Ec** | 73.10 | 0.4879 | 0.6609 | 0.5271 | 0.3186 | 0.5384 | 0.6241 | 0.364 | 0.6351 |
| **LibreYOLOXn** | 4.27 | 0.2551 | 0.4117 | 0.2656 | 0.08 | 0.2707 | 0.4071 | 0.2401 | 0.4174 |
| **LibreYOLOXt** | 5.06 | 0.3263 | 0.5024 | 0.3467 | 0.1378 | 0.3586 | 0.4939 | 0.2833 | 0.4827 |
| **LibreYOLOXs** | 10.28 | 0.4028 | 0.5913 | 0.437 | 0.2284 | 0.4485 | 0.5379 | 0.325 | 0.5743 |
| **LibreYOLOXm** | 16.24 | 0.4668 | 0.6533 | 0.5092 | 0.2902 | 0.5165 | 0.6163 | 0.3619 | 0.6265 |
| **LibreYOLOXl** | 26.03 | 0.497 | 0.6792 | 0.5403 | 0.3331 | 0.5486 | 0.6497 | 0.3782 | 0.6504 |
| **LibreYOLOXx** | 42.94 | 0.5103 | 0.692 | 0.5536 | 0.346 | 0.5632 | 0.6693 | 0.3804 | 0.6587 |
| **LibreDEIMn** | 77.19 | 0.4051 | 0.573 | 0.4347 | 0.1994 | 0.4432 | 0.6069 | 0.3219 | 0.5887 |
| **LibreDEIMs** | 158.50 | 0.4656 | 0.6339 | 0.5041 | 0.2633 | 0.5016 | 0.6533 | 0.3539 | 0.6545 |
| **LibreDEIMm** | 337.04 | 0.509 | 0.6817 | 0.5497 | 0.3175 | 0.5521 | 0.692 | 0.3779 | 0.6861 |
| **LibreDEIMl** | 897.15 | 0.5275 | 0.7024 | 0.5727 | 0.3295 | 0.5781 | 0.7091 | 0.3843 | 0.7041 |
| **LibreDEIMx** | 1789.67 | 0.5446 | 0.7214 | 0.5906 | 0.345 | 0.6002 | 0.729 | 0.3937 | 0.7146 |
| **LibreDEIMv2n** | 88.67 | 0.4017 | 0.567 | 0.4339 | 0.1949 | 0.4418 | 0.5961 | 0.3217 | 0.5803 |
| **LibreDEIMv2s** | 527.40 | 0.4352 | 0.6013 | 0.4712 | 0.248 | 0.4788 | 0.6164 | 0.3442 | 0.6286 |
| **LibreDEIMv2m** | 570.11 | 0.4625 | 0.6271 | 0.4996 | 0.2694 | 0.4998 | 0.6492 | 0.3592 | 0.6554 |
| **LibreDEIMv2l** | 741.22 | 0.5223 | 0.6952 | 0.5669 | 0.3173 | 0.5757 | 0.7268 | 0.3817 | 0.6957 |
| **LibreDEIMv2x** | 990.79 | 0.5475 | 0.7232 | 0.5953 | 0.3479 | 0.6006 | 0.7487 | 0.3942 | 0.706 |

## Графики Precision-Recall

### LibreYOLO9t
![PR Curve LibreYOLO9t](images/PR_LibreYOLO9t.png)

### LibreYOLO9s
![PR Curve LibreYOLO9s](images/PR_LibreYOLO9s.png)

### LibreYOLO9m
![PR Curve LibreYOLO9m](images/PR_LibreYOLO9m.png)

### LibreYOLO9c
![PR Curve LibreYOLO9c](images/PR_LibreYOLO9c.png)

### LibreYOLO9E2Et
![PR Curve LibreYOLO9E2Et](images/PR_LibreYOLO9E2Et.png)

### LibreYOLO9E2Es
![PR Curve LibreYOLO9E2Es](images/PR_LibreYOLO9E2Es.png)

### LibreYOLO9E2Em
![PR Curve LibreYOLO9E2Em](images/PR_LibreYOLO9E2Em.png)

### LibreYOLO9E2Ec
![PR Curve LibreYOLO9E2Ec](images/PR_LibreYOLO9E2Ec.png)

### LibreYOLOXn
![PR Curve LibreYOLOXn](images/PR_LibreYOLOXn.png)

### LibreYOLOXt
![PR Curve LibreYOLOXt](images/PR_LibreYOLOXt.png)

### LibreYOLOXs
![PR Curve LibreYOLOXs](images/PR_LibreYOLOXs.png)

### LibreYOLOXm
![PR Curve LibreYOLOXm](images/PR_LibreYOLOXm.png)

### LibreYOLOXl
![PR Curve LibreYOLOXl](images/PR_LibreYOLOXl.png)

### LibreYOLOXx
![PR Curve LibreYOLOXx](images/PR_LibreYOLOXx.png)

### LibreDEIMn
![PR Curve LibreDEIMn](images/PR_LibreDEIMn.png)

### LibreDEIMs
![PR Curve LibreDEIMs](images/PR_LibreDEIMs.png)

### LibreDEIMm
![PR Curve LibreDEIMm](images/PR_LibreDEIMm.png)

### LibreDEIMl
![PR Curve LibreDEIMl](images/PR_LibreDEIMl.png)

### LibreDEIMx
![PR Curve LibreDEIMx](images/PR_LibreDEIMx.png)

### LibreDEIMv2n
![PR Curve LibreDEIMv2n](images/PR_LibreDEIMv2n.png)

### LibreDEIMv2s
![PR Curve LibreDEIMv2s](images/PR_LibreDEIMv2s.png)

### LibreDEIMv2m
![PR Curve LibreDEIMv2m](images/PR_LibreDEIMv2m.png)

### LibreDEIMv2l
![PR Curve LibreDEIMv2l](images/PR_LibreDEIMv2l.png)

### LibreDEIMv2x
![PR Curve LibreDEIMv2x](images/PR_LibreDEIMv2x.png)

