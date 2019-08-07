# nbstripout removes output from .ipynb files
pip install nbstripout
nbstripout --install

# writes version file based on the commit count for every commit 
cat <<'EOF' > .git/hooks/post-commit
#!/usr/bin/env bash
set -e
commit=$(git describe --long --tags --dirty --match '[0-9]*\.[0-9]*')
version=$(cat version)
if [ $commit != $version ]; then
  echo "commit number $commit"
  echo $commit > version
  git add version
  git commit --amend -C HEAD --no-verify
fi
EOF
chmod +x .git/hooks/post-commit