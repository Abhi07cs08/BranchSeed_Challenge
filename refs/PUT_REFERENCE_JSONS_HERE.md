# Where the data goes

Download the dataset from Google Drive and unpack it so this folder looks like:

    data/
      subject001/
        orig1.nii
        mask1.nii
      subject002/
        orig2.nii
        mask2.nii
      ...

`run_all.sh` globs `data/*/` for one `*orig*.nii*` and one `*mask*.nii*` per folder, so
the exact filenames inside each subject folder don't matter as long as one contains
"orig" and one contains "mask". Compressed `.nii.gz` works too.

Put the dev-subset reference JSONs in `refs/` (one per case, named by case id) so
`make score` can use them. Nothing in `data/` or `refs/` is tracked by git.
