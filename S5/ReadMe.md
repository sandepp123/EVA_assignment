# Analysis of S5 solution

**Step 1** : Chose 99% accuracy by making the classroom notebooks **CODE9** under 10000 parameter, and wanted to check if I can reach 99%. As only thing compensate for less parameter model is data. **Code9** has image augmentation so used it.

**Step2** : Tried to have 99.4% accuracy, by optimizing my core model, the architecture, tried to leverage extra 300-400 unused parameters without using any LR and using experimented with dropout value and. Got 99.36% accuracy

**Step3**: Used LR and lowered the dropouts and reached the results 99.4 although it was attained in 11-12th epoch but final value got to be 99.4.

