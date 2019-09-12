set -e  # exit on error
set -u  # raise error if variables undefined

master-rev(){
  git ls-remote $1 HEAD | head -n 1 | awk '{ print $1}'
}