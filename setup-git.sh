# nbstripout removes output from .ipynb files
pip install nbstripout
nbstripout --install

# writes version file based on the commit count for every commit 
cat <<'EOF' > .git/hooks/post-commit
#!/usr/bin/env bash
commit=$(git rev-list HEAD --count)
version=$(cat version)
if [ $commit != $version ]; then
  echo "commit number $commit"
  printf $commit > version
  git add version
  git commit --amend -C HEAD --no-verify
fi
EOF
chmod +x .git/hooks/post-commit