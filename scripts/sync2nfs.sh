rsync -avm --include='*/' --include='*.py' --include='uv.lock' --include='pyproject.toml' --exclude='*' ~/ml/kaggle/src/ /mnt/mlarena/src/
rsync -avm --include='*/' --include='*.py' --include='uv.lock' --include='pyproject.toml' --exclude='*' ~/ml/kaggle/scripts/ /mnt/mlarena/scripts/
rsync -av ~/ml/kaggle/uv.lock ~/ml/kaggle/pyproject.toml /mnt/mlarena/
