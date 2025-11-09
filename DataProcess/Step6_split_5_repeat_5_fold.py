"""
Data splitting for 5-repeat 5-fold cross-validation.

Inputs:
- Step5_refilter_categorical_for_deeplearning.csv

Outputs:
- Iter_<N_ITER>_Folds_<N_FOLDS>.json
- iter01_folds.json, ..., iter<N_ITER>_folds.json
- Subject_occurrence_count_<N_ITER>iter_<N_FOLDS>fold.csv

Description:
This script performs repeated stratified 5-fold cross-validation on the
top/bottom 10% fluid intelligence subset (already encoded as 'fluid_2_p10').

For each iteration:
- Randomly samples N_PER_CLASS subjects from each class (fluid_2_p10 = 0 and 1),
  using a chunk-based scheme to improve coverage across iterations.
- Performs stratified K-fold splitting (N_FOLDS) on the sampled subjects.
- Saves train/validation subject IDs (eid) for each fold into JSON files.
"""

import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ---------- Config ----------
root_path = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB/FinalFinal_Version_1108'
csv_path = os.path.join(root_path, 'Step5', 'Step5_refilter_categorical_for_deeplearning.csv')

save_root = os.path.join(root_path, 'Step6')
os.makedirs(save_root, exist_ok=True)

N_PER_CLASS = 1500   # number of subjects sampled per class in each iteration
N_ITER = 5           # number of repeated iterations
N_FOLDS = 5          # number of CV folds
SEED = 2025

json_all_path = os.path.join(save_root, f"Iter_{N_ITER}_Folds_{N_FOLDS}.json")
count_csv_path = os.path.join(save_root, f"Subject_occurrence_count_{N_ITER}iter_{N_FOLDS}fold.csv")

# ---------- Load ----------
df = pd.read_csv(csv_path)
if not {'eid', 'fluid_2_p10'}.issubset(df.columns):
    raise ValueError("Required columns: 'eid', 'fluid_2_p10'.")

# Keep only columns needed for splitting (saves memory)
df = df[['eid', 'fluid_2_p10']].copy()

# ---------- Pool of IDs by class ----------
ids0_all = df.loc[df['fluid_2_p10'] == 0, 'eid'].unique().tolist()
ids1_all = df.loc[df['fluid_2_p10'] == 1, 'eid'].unique().tolist()

rng = np.random.RandomState(SEED)
rng.shuffle(ids0_all)
rng.shuffle(ids1_all)

# Split into N_ITER chunks for round-robin coverage
def chunks(lst, n):
    return np.array_split(np.array(lst, dtype=object), n)

chunks0 = chunks(ids0_all, N_ITER)  # list of length N_ITER (each element: np.array)
chunks1 = chunks(ids1_all, N_ITER)

def sample_for_iter(chunks_c, all_ids_c, iter_idx, n_pick, rnd):
    """
    For a given class and iteration, first use the iter_idx-th chunk as a base set.
    If the chunk size is smaller than n_pick, fill the remainder from the full pool
    (preferring IDs not in the base, allowing duplicates only if necessary).
    """
    base = chunks_c[iter_idx].tolist()
    need = n_pick - len(base)
    if need <= 0:
        # If the chunk is larger than needed, subsample without replacement
        return rnd.choice(base, size=n_pick, replace=False).tolist()

    # Remainder candidates: all IDs not in 'base'
    rest = [e for e in all_ids_c if e not in set(base)]
    if len(rest) >= need:
        extra = rnd.choice(rest, size=need, replace=False).tolist()
    else:
        # If not enough unique IDs, allow replacement to fill up
        extra = rnd.choice(all_ids_c, size=need, replace=True).tolist()
    return base + extra

# ---------- Build 5 repeated splits ----------
iterations = []  # each element: dict(iteration, class0_count, class1_count, folds[...])

for it in range(N_ITER):
    # Sample N_PER_CLASS per class for this iteration
    pick0 = sample_for_iter(chunks0, ids0_all, it, N_PER_CLASS, rng)
    pick1 = sample_for_iter(chunks1, ids1_all, it, N_PER_CLASS, rng)
    iter_ids = np.array(pick0 + pick1, dtype=object)
    rng.shuffle(iter_ids)

    sub = df[df['eid'].isin(iter_ids)].copy()
    y = sub['fluid_2_p10'].values

    # Stratified K-fold splitting
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + it)
    folds = []
    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(sub, y), start=1):
        tr_eids = sub.iloc[tr_idx]['eid'].tolist()
        te_eids = sub.iloc[te_idx]['eid'].tolist()
        folds.append({
            "fold": fold_id,
            "train_eid": tr_eids,
            "valid_eid": te_eids
        })

    iterations.append({
        "iteration": it + 1,
        "class0_count": int((sub['fluid_2_p10'] == 0).sum()),
        "class1_count": int((sub['fluid_2_p10'] == 1).sum()),
        "folds": folds
    })

# ---------- Save JSON splits ----------
# 1) Save all iterations into a single JSON file
with open(json_all_path, "w") as f:
    json.dump(
        {
            "meta": {
                "n_iter": N_ITER,
                "n_folds": N_FOLDS,
                "per_class": N_PER_CLASS,
                "seed": SEED
            },
            "iterations": iterations
        },
        f,
        indent=2
    )
print(f"Saved: {json_all_path}")

# 2) Optionally, also save per-iteration JSON files
for item in iterations:
    ipath = os.path.join(save_root, f"iter{item['iteration']:02d}_folds.json")
    with open(ipath, "w") as f:
        json.dump(item, f, indent=2)
    print(f"Saved: {ipath}")
# ---------- Sanity check + subject occurrence counting (validation-based) ----------
with open(json_all_path, 'r') as f:
    ids = json.load(f)

valid_total = []
subject_counter = {}  # counts how many times each subject appears in VALIDATION sets

for iteridx in range(N_ITER):
    iter_wise = []
    print(f"\n===== Iteration {iteridx + 1} =====")
    for foldidx in range(N_FOLDS):
        iteration_fold = ids['iterations'][iteridx]['folds'][foldidx]

        # We treat 'valid_eid' as the test set for this fold
        valid_eid = iteration_fold['valid_eid']

        iter_wise.extend(valid_eid)
        valid_total.extend(valid_eid)

        # ---- Count appearance frequency across all VALIDATION sets ----
        for eid in valid_eid:
            subject_counter[eid] = subject_counter.get(eid, 0) + 1

        uniq_numb = len(np.unique(valid_eid))
        print(
            f"Iter {iteridx + 1} / Fold {foldidx + 1} / "
            f"Subjects: {len(valid_eid)} / Unique in fold: {uniq_numb}"
        )

    uniq_iter = len(np.unique(iter_wise))
    print(f"→ Iter {iteridx + 1} total unique validation subjects: {uniq_iter}")

valid_total_unique = len(np.unique(valid_total))
print(f"\n=== Unique validation subjects across all {N_ITER} iterations: {valid_total_unique}")

# ---------- Save subject occurrence count (validation-based) ----------
count_df = pd.DataFrame(list(subject_counter.items()), columns=["eid", "valid_count"])
count_df.to_csv(count_csv_path, index=False)

print(f"\nSaved validation-based subject occurrence counts to: {count_csv_path}")
print(f"Total subjects tracked: {len(count_df)}")
print(count_df.describe())
